# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Any, List, Optional, Union

from deprecate import void
from torch import Tensor

from pytorch_lightning.loops.base import Loop
from pytorch_lightning.loops.optimization.manual_loop import _OUTPUTS_TYPE as _MANUAL_LOOP_OUTPUTS_TYPE
from pytorch_lightning.loops.optimization.manual_loop import ManualOptimization
from pytorch_lightning.loops.optimization.optimizer_loop import _OUTPUTS_TYPE as _OPTIMIZER_LOOP_OUTPUTS_TYPE
from pytorch_lightning.loops.optimization.optimizer_loop import OptimizerLoop
from pytorch_lightning.loops.utilities import _get_active_optimizers
from pytorch_lightning.trainer.supporters import TensorRunningAccum

_OUTPUTS_TYPE = List[Union[_OPTIMIZER_LOOP_OUTPUTS_TYPE, _MANUAL_LOOP_OUTPUTS_TYPE]]


class IPL_TrainingBatchLoop(Loop[_OUTPUTS_TYPE]):
    """Runs over a single batch of data."""

    def __init__(self) -> None:
        super().__init__()
        self.accumulated_loss: Optional[Tensor] = None
        self.running_loss: TensorRunningAccum = TensorRunningAccum(window_length=20)
        # the current split index when the batch gets split into chunks in truncated backprop through time
        self.split_idx: Optional[int] = None
        self.optimizer_loop = OptimizerLoop()
        self.manual_loop = ManualOptimization()

        self._outputs: _OUTPUTS_TYPE = []
        self._remaining_splits: Optional[List[Any]] = None

    @property
    def done(self) -> bool:
        """Returns if all batch splits have been processed already."""
        return len(self._remaining_splits) == 0

    def connect(
            self, optimizer_loop: Optional["Loop"] = None, manual_loop: Optional[ManualOptimization] = None
    ) -> None:
        if optimizer_loop is not None:
            self.optimizer_loop = optimizer_loop
        if manual_loop is not None:
            self.manual_loop = manual_loop

    def reset(self) -> None:
        """Resets the loop state."""
        self._outputs = []

    def on_run_start(self, batch: Any, batch_idx: int):
        """Splits the data into tbptt splits.
        Args:
            batch: the current batch to run the trainstep on
            batch_idx: the index of the current batch
        """
        void(batch_idx)
        self._remaining_splits = list(enumerate(self._tbptt_split_batch(batch)))

    def advance(self, batch, batch_idx):
        """Runs the train step together with optimization (if necessary) on the current batch split.
        Args:
            batch: the current batch to run the training on (this is not the split!)
            batch_idx: the index of the current batch
        """
        void(batch)
        self.split_idx, split_batch = self._remaining_splits.pop(0)
        # let logger connector extract current batch size
        self.trainer.logger_connector.on_train_split_start(self.split_idx, split_batch)

        # choose which loop will run the optimization
        if self.trainer.lightning_module.automatic_optimization:
            optimizers = _get_active_optimizers(self.trainer.optimizers, self.trainer.optimizer_frequencies, batch_idx)
            outputs = self.optimizer_loop.run(split_batch, optimizers, batch_idx)
        else:
            outputs = self.manual_loop.run(split_batch, batch_idx)
        if outputs:
            # automatic: can be empty if all optimizers skip their batches
            # manual: #9052 added support for raising `StopIteration` in the `training_step`. If that happens,
            # then `advance` doesn't finish and an empty dict is returned
            self._outputs.append(outputs)

    def on_run_end(self) -> _OUTPUTS_TYPE:
        self.optimizer_loop._hiddens = None
        # this is not necessary as the manual loop runs for only 1 iteration, but just in case
        self.manual_loop._hiddens = None
        output, self._outputs = self._outputs, None  # free memory
        return output

    def teardown(self) -> None:
        # release memory
        self._remaining_splits = None

    def _tbptt_split_batch(self, batch: Any) -> List[Any]:
        """Splits a single batch into a list of sequence steps for tbptt.
        Args:
            batch: the current batch to split
        """
        tbptt_steps = self.trainer.lightning_module.truncated_bptt_steps
        if tbptt_steps == 0:
            return [batch]

        model_ref = self.trainer.lightning_module
        with self.trainer.profiler.profile("tbptt_split_batch"):
            splits = model_ref.tbptt_split_batch(batch, tbptt_steps)
        return splits

    def _update_running_loss(self, current_loss: Tensor) -> None:
        """Updates the running loss value with the current value."""
        if self.trainer.lightning_module.automatic_optimization:
            # track total loss for logging (avoid mem leaks)
            self.accumulated_loss.append(current_loss)

        accumulated_loss = self.accumulated_loss.mean()

        if accumulated_loss is not None:
            # calculate running loss for display
            self.running_loss.append(self.accumulated_loss.mean() * self.trainer.accumulate_grad_batches)

        # reset for next set of accumulated grads
        self.accumulated_loss.reset()


# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from collections import defaultdict
from typing import Any, Dict, Generator, Iterator, List, Optional, overload, Tuple, Union

import numpy as np
import torch

from pytorch_lightning import loops  # import as loops to avoid circular imports
# from ipl_batch_loop import IPL_TrainingBatchLoop
from pytorch_lightning.loops.batch.training_batch_loop import _OUTPUTS_TYPE as _BATCH_OUTPUTS_TYPE
from pytorch_lightning.loops.utilities import _get_active_optimizers, _is_max_limit_reached, _update_dataloader_iter
from pytorch_lightning.trainer.connectors.logger_connector.result import ResultCollection
from pytorch_lightning.trainer.progress import BatchProgress, SchedulerProgress
from pytorch_lightning.utilities import rank_zero_warn
from pytorch_lightning.utilities.apply_func import apply_to_collection
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.fetching import AbstractDataFetcher
from pytorch_lightning.utilities.model_helpers import is_overridden
from pytorch_lightning.utilities.signature_utils import is_param_in_hook_signature
from pytorch_lightning.utilities.warnings import rank_zero_deprecation, WarningCache
from utils import _speech_collate_fn
_OUTPUTS_TYPE = List[_BATCH_OUTPUTS_TYPE]


class IPL_TrainingEpochLoop(loops.Loop[_OUTPUTS_TYPE]):
    """Runs over all batches in a dataloader (one epoch).
    Args:
        min_steps: The minimum number of steps (batches) to process
        max_steps: The maximum number of steps (batches) to process
    """

    def __init__(self, min_steps: Optional[int] = 0, max_steps: int = -1,
                n_supervised: int = 1,
                n_unsupervised: int = 4,
                 p: float = 0.5
                ) -> None:
        super().__init__()
        if max_steps is None:
            rank_zero_deprecation(
                "Setting `max_steps = None` is deprecated in v1.5 and will no longer be supported in v1.7."
                " Use `max_steps = -1` instead."
            )
            max_steps = -1
        elif max_steps < -1:
            raise MisconfigurationException(
                f"`max_steps` must be a non-negative integer or -1 (infinite steps). You passed in {max_steps}."
            )
        self.min_steps = min_steps
        self.max_steps = max_steps
        self.n_supervised = n_supervised
        self.n_unsupervised = n_unsupervised
        self.p = p
        self.global_step: int = 0
        self.batch_progress = BatchProgress()
        self.scheduler_progress = SchedulerProgress()

        self.batch_loop: Optional[IPL_TrainingBatchLoop] = None
        self.val_loop: Optional["loops.EvaluationLoop"] = None

        self._results = ResultCollection(training=True)
        self._outputs: _OUTPUTS_TYPE = []
        self._warning_cache = WarningCache()
        self._dataloader_iter: Optional[Iterator] = None
        # caches the loaded dataloader state until dataloader objects are available
        self._dataloader_state_dict: Dict[str, Any] = {}
        
#         self.output_file = '/store1/vkreeger/IPL/cache_logs.txt'
#         self.tok_model = SentencePieceTokenizer('/store1/vkreeger/IPL/tokenizers/tokenizer_spe_unigram_v1024/tokenizer.model')
#         self.alphabet = [" ", "а", "б", "в", "г", "д", "ё", "е", "ж", "з", "и", "й", "к", "л",
#                    "м", "н", "о", "п", "р", "с", "т", "у", "ф", "х", "ц", "ч", "ш", "щ",
#                    "ь", "ы", "ъ", "э", "ю", "я"]
        self.alphabet = [" ", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
                   "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "'"]

    @property
    def total_batch_idx(self) -> int:
        """Returns the current batch index (across epochs)"""
        # use `ready` instead of `completed` in case this is accessed after `completed` has been increased
        # but before the next `ready` increase
        return self.batch_progress.total.ready - 1

    @property
    def batch_idx(self) -> int:
        """Returns the current batch index (within this epoch)"""
        # use `ready` instead of `completed` in case this is accessed after `completed` has been increased
        # but before the next `ready` increase
        return self.batch_progress.current.ready - 1

    @property
    def _is_training_done(self) -> bool:
        max_steps_reached = _is_max_limit_reached(self.global_step, self.max_steps)
        return max_steps_reached or self._num_ready_batches_reached()

    @property
    def _is_validation_done(self) -> bool:
        # when we are restarting we want to check whether the val loop has finished
        return not self.restarting or self.val_loop.done

    @property
    def done(self) -> bool:
        """Returns whether the training should be stopped.
        The criteria are that the number of steps reached the max steps, the last batch is reached or the trainer
        signals to stop (e.g. by early stopping).
        """
        return (self._is_training_done and self._is_validation_done) or self.trainer.should_stop

    def connect(
            self,
            batch_loop: IPL_TrainingBatchLoop = None,
            val_loop: Optional["loops.EvaluationLoop"] = None,
    ) -> None:
        """Optionally connect a custom batch or validation loop to this training epoch loop."""
        if batch_loop is not None:
            self.batch_loop = batch_loop
        if val_loop is not None:
            self.val_loop = val_loop

    def reset(self) -> None:
        """Resets the internal state of the loop for a new run."""
        assert self.batch_loop is not None
        assert self.batch_loop.optimizer_loop is not None
        if self.restarting:
            self.batch_progress.reset_on_restart()
            self.scheduler_progress.reset_on_restart()
            self.batch_loop.optimizer_loop.optim_progress.reset_on_restart()
        else:
            self.batch_progress.reset_on_run()
            self.scheduler_progress.reset_on_run()
            self.batch_loop.optimizer_loop.optim_progress.reset_on_run()

        self._outputs = []

    def on_run_start(self, data_fetcher: AbstractDataFetcher, **kwargs: Any) -> None:
        # hook
        self.trainer.logger_connector.on_epoch_start()
        self.trainer.call_hook("on_epoch_start")
        self.trainer.call_hook("on_train_epoch_start")
        self.trainer.fit_loop.epoch_progress.increment_started()
        
        self._reload_dataloader_state_dict(data_fetcher)
        self._dataloader_iter = _update_dataloader_iter(data_fetcher, self.batch_idx + 1)
        self._test_dataloader_iter = iter(self.trainer.model.test_dataloader())
    
    def _advance_supervised(self):
        batch_idx, (batch, self.batch_progress.is_last_batch) = next(self._dataloader_iter)
        if not self.trainer._data_connector.train_data_fetcher.store_on_device:
            with self.trainer.profiler.profile("training_batch_to_device"):
                batch = self.trainer.accelerator.batch_to_device(batch)

        self.batch_progress.increment_ready()

        self.trainer.logger_connector.on_batch_start(batch_idx, batch)

        if batch is None:
            self._warning_cache.warn("train_dataloader yielded None. If this was on purpose, ignore this warning...")
            batch_output = []
        else:
            # hook
            response = self.trainer.call_hook("on_batch_start")
            if response == -1:
                self.batch_progress.increment_processed()
                raise StopIteration

            # TODO: Update this in v1.7 (deprecation: #9816)
            model_fx = self.trainer.lightning_module.on_train_batch_start
            extra_kwargs = (
                {"dataloader_idx": 0}
                if callable(model_fx) and is_param_in_hook_signature(model_fx, "dataloader_idx", explicit=True)
                else {}
            )

            # hook
            response = self.trainer.call_hook("on_train_batch_start", batch, batch_idx, **extra_kwargs)
            if response == -1:
                self.batch_progress.increment_processed()
                raise StopIteration

            self.batch_progress.increment_started()

            with self.trainer.profiler.profile("run_training_batch"):
                batch_output = self.batch_loop.run(batch, batch_idx)

        self.batch_progress.increment_processed()

        # update non-plateau LR schedulers
        # update epoch-interval ones only when we are at the end of training epoch
        self.update_lr_schedulers("step", update_plateau_schedulers=False)
        if self._num_ready_batches_reached():
            self.update_lr_schedulers("epoch", update_plateau_schedulers=False)

        batch_end_outputs = self._prepare_outputs_training_batch_end(
            batch_output,
            automatic=self.trainer.lightning_module.trainer.lightning_module.automatic_optimization,
            num_optimizers=len(self.trainer.optimizers),
        )

        # TODO: Update this in v1.7 (deprecation: #9816)
        model_fx = self.trainer.lightning_module.on_train_batch_end
        extra_kwargs = (
            {"dataloader_idx": 0}
            if callable(model_fx) and is_param_in_hook_signature(model_fx, "dataloader_idx", explicit=True)
            else {}
        )
        self.trainer.call_hook("on_train_batch_end", batch_end_outputs, batch, batch_idx, **extra_kwargs)
        self.trainer.call_hook("on_batch_end")
        self.trainer.logger_connector.on_batch_end()

        self.batch_progress.increment_completed()

        if is_overridden("training_epoch_end", self.trainer.lightning_module):
            self._outputs.append(batch_output)

        # -----------------------------------------
        # SAVE METRICS TO LOGGERS AND PROGRESS_BAR
        # -----------------------------------------
        self.trainer.logger_connector.update_train_step_metrics()
    
    def get_cache_batch(self):
        idx = torch.randint(len(self.trainer.model.cache),(1,)).item()
        
        if torch.rand(1).item() < self.p:
            print(f"\nBatch will be replaced! Cache size: {len(self.trainer.model.cache)}")
            is_bad = True
            batch = self.trainer.model.cache.pop(idx)  

            with torch.no_grad():
                while (is_bad):
                    new_batch = next(self._test_dataloader_iter, None)
                    if new_batch is None:
                        self._test_dataloader_iter = iter(self.trainer.model.test_dataloader())
                        new_batch = next(self._test_dataloader_iter, None)
                        
                    new_batch = self.trainer.accelerator.batch_to_device(new_batch)                            
                    self.trainer.model.training = False
                    results = self.trainer.model._predict_step(new_batch, 0)                   

                    cache_batch = _speech_collate_fn([(new_batch[0][i][:new_batch[1][i].item()].cpu(), 
                                    new_batch[1][i].cpu(), 
                                    results[i].y_sequence, \
                                    len(results[i].y_sequence)) for i in range(len(new_batch[0]))], 0)
                    is_bad = False
                    
                    for i in range(len(new_batch[0])):
                        res_text = results[i].text
                        unk_sym = False
                        if len(res_text) == 0:                    
                            is_bad = True
                            print("\nThis batch is bad! Empty prediction")
                            break
                        for c in res_text:
                            if c not in self.alphabet:
                                unk_sym = True
                                print("\nThis batch is bad! Unknown symbol")
                                break
                        if unk_sym:
                            is_bad = True
                            break

#                     for i in range(len(new_batch[0])):
#                         if len(results[i].text) == 0:
#                             print(f"Sample is bad:\n {results[i]}")
#                             is_bad = True
#                             break
            self.trainer.model.cache.append(cache_batch)
            self.trainer.model.training = True
        else:
            batch = self.trainer.model.cache[idx]
            
#         print(f"\n ========================================== \nCurrent cache size: {len(self.trainer.model.cache)}\n ========================================== \n")
        return batch
        
    def _advance_unsupervised(self):
        batch_idx, (batch, self.batch_progress.is_last_batch) = next(self._dataloader_iter)
        batch = self.get_cache_batch()
        
        if not self.trainer._data_connector.train_data_fetcher.store_on_device:            
            with self.trainer.profiler.profile("training_batch_to_device"):
                batch = self.trainer.accelerator.batch_to_device(batch)

        self.batch_progress.increment_ready()

        self.trainer.logger_connector.on_batch_start(batch_idx, batch)

        if batch is None:
            self._warning_cache.warn("train_dataloader yielded None. If this was on purpose, ignore this warning...")
            batch_output = []
        else:
            # hook
            response = self.trainer.call_hook("on_batch_start")
            if response == -1:
                self.batch_progress.increment_processed()
                raise StopIteration

            # TODO: Update this in v1.7 (deprecation: #9816)
            model_fx = self.trainer.lightning_module.on_train_batch_start
            extra_kwargs = (
                {"dataloader_idx": 0}
                if callable(model_fx) and is_param_in_hook_signature(model_fx, "dataloader_idx", explicit=True)
                else {}
            )

            # hook
            response = self.trainer.call_hook("on_train_batch_start", batch, batch_idx, **extra_kwargs)
            if response == -1:
                self.batch_progress.increment_processed()
                raise StopIteration

            self.batch_progress.increment_started()

            with self.trainer.profiler.profile("run_training_batch"):
                batch_output = self.batch_loop.run(batch, batch_idx)

        self.batch_progress.increment_processed()

        # update non-plateau LR schedulers
        # update epoch-interval ones only when we are at the end of training epoch
        self.update_lr_schedulers("step", update_plateau_schedulers=False)
        if self._num_ready_batches_reached():
            self.update_lr_schedulers("epoch", update_plateau_schedulers=False)

        batch_end_outputs = self._prepare_outputs_training_batch_end(
            batch_output,
            automatic=self.trainer.lightning_module.trainer.lightning_module.automatic_optimization,
            num_optimizers=len(self.trainer.optimizers),
        )

        # TODO: Update this in v1.7 (deprecation: #9816)
        model_fx = self.trainer.lightning_module.on_train_batch_end
        extra_kwargs = (
            {"dataloader_idx": 0}
            if callable(model_fx) and is_param_in_hook_signature(model_fx, "dataloader_idx", explicit=True)
            else {}
        )
        self.trainer.call_hook("on_train_batch_end", batch_end_outputs, batch, batch_idx, **extra_kwargs)
        self.trainer.call_hook("on_batch_end")
        self.trainer.logger_connector.on_batch_end()

        self.batch_progress.increment_completed()

        if is_overridden("training_epoch_end", self.trainer.lightning_module):
            self._outputs.append(batch_output)

        # -----------------------------------------
        # SAVE METRICS TO LOGGERS AND PROGRESS_BAR
        # -----------------------------------------
        self.trainer.logger_connector.update_train_step_metrics()
    
    def advance(self, *args: Any, **kwargs: Any) -> None:
        """Runs a single training batch.
        Args:
            dataloader_iter: the iterator over the dataloader producing the new batch
        Raises:
            StopIteration: When the epoch is canceled by the user returning -1
        """
        if self.restarting and self._should_check_val_fx(self.batch_idx, self.batch_progress.is_last_batch):
            # skip training and run validation in `on_advance_end`
            return
        
        if self.global_step % (self.n_unsupervised + self.n_supervised) >= self.n_supervised:
            if self.global_step % (self.n_unsupervised + self.n_supervised) == self.n_supervised:
                    
                # logging cache samples  
#                 with open(self.output_file, 'a') as out:
#                     items = random.choices(self.trainer.model.cache, k=2)
#                     out.write(f"\n ========================================== \n CACHE ITER {self.global_step} \n ========================================== \n")
#                     for item in items:
#                         for i in item[2]:
#                             out.write(self.tok_model.ids_to_text(i.tolist())); out.write('\n')
                        
                torch.cuda.empty_cache()
                        
            self._advance_unsupervised()
        else:
            self._advance_supervised()
                
        

    def on_advance_end(self):
        """Runs validation and Checkpointing if necessary.
        Raises:
            StopIteration: if :attr:`done` evaluates to ``True`` to finish this epoch
        """
        # -----------------------------------------
        # VALIDATE IF NEEDED + CHECKPOINT CALLBACK
        # -----------------------------------------
        should_check_val = self._should_check_val_fx(self.batch_idx, self.batch_progress.is_last_batch)
        if should_check_val:
            self.trainer.validating = True
            self._run_validation()
            self.trainer.training = True

        # -----------------------------------------
        # SAVE LOGGERS (ie: Tensorboard, etc...)
        # -----------------------------------------
        self._save_loggers_on_train_batch_end()

        # update plateau LR scheduler after metrics are logged
        self.update_lr_schedulers("step", update_plateau_schedulers=True)

        if not self._should_accumulate():
            # progress global step according to grads progress
            self.global_step += 1

        # if training finished, try to exit in `on_run_end` instead as we should have enough time
        # TODO: @tchaton verify this assumption is True.
        if not self._is_training_done:
            # if fault tolerant is enabled and process has been notified, exit.
            self.trainer._exit_gracefully_on_signal()

    def on_run_end(self) -> None:
        """Calls the on_epoch_end hook.
        Returns:
            The output of each training step for each optimizer
        Raises:
            MisconfigurationException: ``train_epoch_end`` does not return ``None``
        """
        # inform logger the batch loop has finished
        self.trainer.logger_connector.epoch_end_reached()

        # get the model and call model.training_epoch_end
        model = self.trainer.lightning_module
        if is_overridden("training_epoch_end", model) and self._outputs:
            epoch_end_outputs = self._prepare_outputs_training_epoch_end(
                self._outputs,
                automatic=model.automatic_optimization,
                num_optimizers=len(self.trainer.optimizers),
            )
            # run lightning module hook training_epoch_end
            # refresh the result for custom logging at the epoch level
            model._current_fx_name = "training_epoch_end"
            epoch_end_outputs = model.training_epoch_end(epoch_end_outputs)
            if epoch_end_outputs is not None:
                raise MisconfigurationException(
                    "`training_epoch_end` expects a return of None. "
                    "HINT: remove the return statement in `training_epoch_end`."
                )
        # free memory
        self._outputs = []

        self.trainer.fit_loop.epoch_progress.increment_processed()

        # call train epoch end hooks
        self.trainer.call_hook("on_train_epoch_end")
        self.trainer.call_hook("on_epoch_end")
        self.trainer.logger_connector.on_epoch_end()

        if self._num_ready_batches_reached():
            self.update_lr_schedulers("epoch", update_plateau_schedulers=True)

        # if fault tolerant is enabled and process has been notified, exit.
        self.trainer._exit_gracefully_on_signal()

    def teardown(self) -> None:
        self._results.cpu()
        self.batch_loop.teardown()
        self.val_loop.teardown()

    def on_save_checkpoint(self) -> Dict:
        state_dict = super().on_save_checkpoint()

        if (
                self.trainer.train_dataloader is None
                or self._num_completed_batches_reached()  # did not finish
                # TODO: fault-tolerance requires a minimum number of batches so probably should be > 0
                or self.batch_progress.current.ready == 0  # did not start
        ):
            return state_dict
        state_dict["dataloader_state_dict"] = self.trainer.train_dataloader.state_dict(
            has_completed=self._has_completed()
        )
        return state_dict

    def on_load_checkpoint(self, state_dict: Dict) -> None:
        # cache the dataloader state dict until the dataloader objects are available
        self._dataloader_state_dict = state_dict.get("dataloader_state_dict")

    def _run_validation(self):
        # reload dataloaders
        self.val_loop._reload_evaluation_dataloaders()

        with torch.no_grad():
            self.val_loop.run()

    def _accumulated_batches_reached(self) -> bool:
        """Determine if accumulation will be finished by the end of the current batch."""
        return self.batch_progress.current.ready % self.trainer.accumulate_grad_batches == 0

    def _num_ready_batches_reached(self) -> bool:
        """Checks if we are in the last batch or if there are more batches to follow."""
        epoch_finished_on_ready = self.batch_progress.current.ready == self.trainer.num_training_batches
        return epoch_finished_on_ready or self.batch_progress.is_last_batch

    def _num_completed_batches_reached(self) -> bool:
        epoch_finished_on_completed = self.batch_progress.current.completed == self.trainer.num_training_batches
        dataloader_consumed_successfully = self.batch_progress.is_last_batch and self._has_completed()
        return epoch_finished_on_completed or dataloader_consumed_successfully

    def _has_completed(self) -> bool:
        return self.batch_progress.current.ready == self.batch_progress.current.completed

    def _should_accumulate(self) -> bool:
        """Checks if the optimizer step should be performed or gradients should be accumulated for the current
        step."""
        accumulation_done = self._accumulated_batches_reached()
        # Lightning steps on the final batch
        is_final_batch = self._num_ready_batches_reached()
        # but the TTP might not
        ttp_accumulates_on_final_batch = (
                self.trainer.training_type_plugin.handles_gradient_accumulation or not is_final_batch
        )
        return not accumulation_done and ttp_accumulates_on_final_batch

    @staticmethod
    def _prepare_outputs_training_batch_end(
            batch_output: _BATCH_OUTPUTS_TYPE,
            automatic: bool,
            num_optimizers: int,
    ) -> Union[List[List[Dict[str, Any]]], List[Dict[str, Any]]]:
        """Processes the outputs from the batch loop into the format passed to the ``training_batch_end`` hook.
        ``(tbptt_steps, n_opt) -> (n_opt, tbptt_steps)``. The optimizer dimension might have been squeezed.
        """
        if not batch_output:
            return []

        # convert optimizer dicts to list
        if automatic:
            batch_output = apply_to_collection(
                batch_output, dtype=dict, function=_convert_optim_dict, num_optimizers=num_optimizers
            )
        array = np.array(batch_output, dtype=object)
        if array.ndim == 1:
            array = np.expand_dims(array, 1)

        array = array.transpose((1, 0))
        array = array.squeeze()
        array = array.tolist()
        array = _recursive_unpad(array)
        return array

    @staticmethod
    def _prepare_outputs_training_epoch_end(
            batch_outputs: _OUTPUTS_TYPE,
            automatic: bool,
            num_optimizers: int,
    ) -> Union[List[List[List[Dict[str, Any]]]], List[List[Dict[str, Any]]], List[Dict[str, Any]]]:
        """Processes the outputs from the batch loop into the format passed to the ``training_epoch_end`` hook.
        ``(n_batches, tbptt_steps, n_opt) -> (n_opt, n_batches, tbptt_steps)``.
        All single-element dimensions might have been squeezed.
        This processing is necessary because the format of the inputs to the ``training_epoch_end`` hook does not
        match the loop structure and because empty dimensions are squeezed. This could break with loop customization.
        """
        # `batch_outputs` (plural) is the same as `epoch_end_output` (singular)
        if not batch_outputs:
            return []

        # convert optimizer dicts to list
        if automatic:
            batch_outputs = apply_to_collection(
                batch_outputs, dtype=dict, function=_convert_optim_dict, num_optimizers=num_optimizers
            )

        array = _recursive_pad(batch_outputs)
        if array.ndim == 2:
            array = np.expand_dims(array, 2)
        array = array.transpose((2, 0, 1))
        array = array.squeeze()
        array = array.tolist()
        array = _recursive_unpad(array)

        # in case we squeezed from 1-element array to a 0-dim array
        array = array if isinstance(array, list) else [array]
        # remove residual empty lists
        array = [item for item in array if not isinstance(item, list) or len(item)]
        return array

    def update_lr_schedulers(self, interval: str, update_plateau_schedulers: bool) -> None:
        """updates the lr schedulers based on the given interval."""
        if interval == "step" and self._should_accumulate():
            return
        active_optimizers = _get_active_optimizers(
            self.trainer.optimizers, self.trainer.optimizer_frequencies, self.total_batch_idx
        )
        self._update_learning_rates(
            interval=interval,
            update_plateau_schedulers=update_plateau_schedulers,
            opt_indices=[opt_idx for opt_idx, _ in active_optimizers],
        )

    def _update_learning_rates(
            self, interval: str, update_plateau_schedulers: bool, opt_indices: Optional[List[int]] = None
    ) -> None:
        """Update learning rates.
        Args:
            interval: either 'epoch' or 'step'.
            update_plateau_schedulers: control whether ``ReduceLROnPlateau`` or non-plateau schedulers get updated.
                This is used so non-plateau schedulers can be updated before running validation. Checkpoints are
                commonly saved during validation, however, on-plateau schedulers might monitor a validation metric
                so they have to be updated separately.
            opt_indices: indices of the optimizers to update.
        """
        if not self.trainer.lr_schedulers or not self.trainer.lightning_module.automatic_optimization:
            return

        if opt_indices is None:
            opt_indices = []

        for lr_scheduler in self.trainer.lr_schedulers:
            if isinstance(lr_scheduler["opt_idx"], int) and lr_scheduler["opt_idx"] not in opt_indices:
                continue

            if update_plateau_schedulers ^ lr_scheduler["reduce_on_plateau"]:
                continue

            current_idx = self.batch_idx if interval == "step" else self.trainer.current_epoch
            current_idx += 1  # account for both batch and epoch starts from 0
            # Take step if call to update_learning_rates matches the interval key and
            # the current step modulo the schedulers frequency is zero
            if lr_scheduler["interval"] == interval and current_idx % lr_scheduler["frequency"] == 0:
                monitor_val = None
                if lr_scheduler["reduce_on_plateau"]:
                    # If instance of ReduceLROnPlateau, we need a monitor
                    monitor_key = lr_scheduler["monitor"]
                    monitor_val = self._get_monitor_value(monitor_key)
                    if monitor_val is None:
                        if lr_scheduler.get("strict", True):
                            avail_metrics = list(self.trainer.callback_metrics)
                            raise MisconfigurationException(
                                f"ReduceLROnPlateau conditioned on metric {monitor_key}"
                                f" which is not available. Available metrics are: {avail_metrics}."
                                " Condition can be set using `monitor` key in lr scheduler dict"
                            )
                        rank_zero_warn(
                            f"ReduceLROnPlateau conditioned on metric {monitor_key}"
                            " which is not available but strict is set to `False`."
                            " Skipping learning rate update.",
                            RuntimeWarning,
                        )
                        continue

                self.scheduler_progress.increment_ready()

                # update LR
                if lr_scheduler["reduce_on_plateau"]:
                    lr_scheduler["scheduler"].step(monitor_val)
                else:
                    lr_scheduler["scheduler"].step()

                self.scheduler_progress.increment_completed()

    def _get_monitor_value(self, key: str) -> Any:
        # this is a separate method to aid in testing
        return self.trainer.callback_metrics.get(key)

    def _should_check_val_fx(self, batch_idx: int, is_last_batch: bool) -> bool:
        """Decide if we should run validation."""
        if not self.trainer.enable_validation:
            return False

        is_val_check_epoch = (self.trainer.current_epoch + 1) % self.trainer.check_val_every_n_epoch == 0
        if not is_val_check_epoch:
            return False

        # val_check_batch is inf for iterable datasets with no length defined
        is_infinite_dataset = self.trainer.val_check_batch == float("inf")
        if is_last_batch and is_infinite_dataset:
            return True

        if self.trainer.should_stop:
            return True

        # TODO(@awaelchli): let training/eval loop handle logic around limit_*_batches and val_check_batch
        is_val_check_batch = is_last_batch
        if isinstance(self.trainer.limit_train_batches, int) and is_infinite_dataset:
            is_val_check_batch = (batch_idx + 1) % self.trainer.limit_train_batches == 0
        elif self.trainer.val_check_batch != float("inf"):
            is_val_check_batch = (batch_idx + 1) % self.trainer.val_check_batch == 0
        return is_val_check_batch

    def _save_loggers_on_train_batch_end(self) -> None:
        """Flushes loggers to disk."""
        # when loggers should save to disk
        should_flush_logs = self.trainer.logger_connector.should_flush_logs
        if should_flush_logs and self.trainer.is_global_zero and self.trainer.logger is not None:
            self.trainer.logger.save()

    def _reload_dataloader_state_dict(self, data_fetcher: AbstractDataFetcher):
        if self._dataloader_state_dict:
            data_fetcher.dataloader.load_state_dict(self._dataloader_state_dict)
            self._dataloader_state_dict = None

# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
from typing import Optional

from pytorch_lightning.loops import Loop
# from ipl_epoch_loop import IPL_TrainingEpochLoop
from pytorch_lightning.loops.utilities import _is_max_limit_reached
from pytorch_lightning.trainer.connectors.logger_connector.result import ResultCollection
from pytorch_lightning.trainer.progress import Progress
from pytorch_lightning.trainer.supporters import TensorRunningAccum
from pytorch_lightning.utilities import rank_zero_deprecation
from pytorch_lightning.utilities.exceptions import MisconfigurationException
# from ipl_prediction_epoch_loop import IPL_PredictionEpochLoop
from nemo.collections.common.tokenizers.sentencepiece_tokenizer import SentencePieceTokenizer
from tqdm import tqdm
from utils import _speech_collate_fn
import torch
import random
log = logging.getLogger(__name__)


class IPL_FitLoop(Loop):
    """This Loop iterates over the epochs to run the training.
    Args:
        min_epochs: The minimum number of epochs
        max_epochs: The maximum number of epochs, can be set -1 to turn this limit off
    """

    def __init__(
            self,
            min_epochs: Optional[int] = 1,
            max_epochs: int = 1000,
            cache_size: int = 100
    ) -> None:
        super().__init__()
        if max_epochs < -1:
            # Allow max_epochs to be zero, since this will be handled by fit_loop.done
            raise MisconfigurationException(
                f"`max_epochs` must be a non-negative integer or -1. You passed in {max_epochs}."
            )

        self.max_epochs = max_epochs
        self.min_epochs = min_epochs
        self.epoch_loop: Optional[IPL_TrainingEpochLoop] = None
        self.epoch_progress = Progress()
        self._is_fresh_start_epoch: bool = True
        self.cache_size = cache_size
        
#         self.output_file = '/store1/vkreeger/IPL/cache_logs.txt'
#         self.tok_model = SentencePieceTokenizer('/store1/vkreeger/IPL/tokenizers/tokenizer_spe_unigram_v1024/tokenizer.model')

    @property
    def current_epoch(self) -> int:
        """Return the current epoch."""
        return self.epoch_progress.current.completed

    @current_epoch.setter
    def current_epoch(self, value: int) -> None:
        """Setter for the current epoch."""
        self.epoch_progress.current.completed = value

    @property
    def global_step(self) -> int:
        """Returns the global step."""
        return self.epoch_loop.global_step

    @global_step.setter
    def global_step(self, value: int) -> None:
        """Sets the global step (forwards to epoch_loop)"""
        self.epoch_loop.global_step = value

    @property
    def total_batch_idx(self) -> int:
        """Returns the current batch index (across epochs)"""
        return self.epoch_loop.total_batch_idx

    @property
    def batch_idx(self) -> int:
        """Returns the current batch index (within this epoch)"""
        return self.epoch_loop.batch_idx

    @property
    def split_idx(self) -> int:
        """Returns the index of the current batch split (within the current batch) for bptt."""
        return self.epoch_loop.batch_loop.split_idx

    @property
    def min_steps(self) -> int:
        # TODO(@justusschock): Why aren't we using the attribute in this class?
        """Returns the minimum numnber of steps to run."""
        return self.epoch_loop.min_steps

    @min_steps.setter
    def min_steps(self, value: int) -> None:
        """Sets the minimum number of steps (forwards to epoch_loop)"""
        # TODO(@awaelchli): This setter is required by debugging connector (fast dev run), should be avoided
        self.epoch_loop.min_steps = value

    @property
    def max_steps(self) -> int:
        """Returns the maximum number of steps to run."""
        return self.epoch_loop.max_steps

    @max_steps.setter
    def max_steps(self, value: int) -> None:
        """Sets the maximum number of steps (forwards to epoch_loop)"""
        # TODO(@awaelchli): This setter is required by debugging connector (fast dev run), should be avoided
        if value is None:
            rank_zero_deprecation(
                "Setting `max_steps = None` is deprecated in v1.5 and will no longer be supported in v1.7."
                " Use `max_steps = -1` instead."
            )
            value = -1
        elif value < -1:
            raise MisconfigurationException(
                f"`max_steps` must be a non-negative integer or -1 (infinite steps). You passed in {value}."
            )
        self.epoch_loop.max_steps = value

    @property
    def running_loss(self) -> TensorRunningAccum:
        """Returns the running loss."""
        return self.epoch_loop.batch_loop.running_loss

    @property
    def _skip_backward(self) -> bool:
        """Determines whether the loop will skip backward during automatic optimization."""
        assert self.epoch_loop.batch_loop is not None
        assert self.epoch_loop.batch_loop.optimizer_loop is not None
        return self.epoch_loop.batch_loop.optimizer_loop._skip_backward

    @_skip_backward.setter
    def _skip_backward(self, value: bool) -> None:
        """Determines whether the loop will skip backward during automatic optimization."""
        assert self.epoch_loop.batch_loop is not None
        assert self.epoch_loop.batch_loop.optimizer_loop is not None
        self.epoch_loop.batch_loop.optimizer_loop._skip_backward = value

    @property
    def _results(self) -> ResultCollection:
        if self.trainer.training:
            return self.epoch_loop._results
        if self.trainer.validating:
            return self.epoch_loop.val_loop._results
        raise RuntimeError("`FitLoop._results` property isn't defined. Accessed outside of scope")

    @property
    def done(self) -> bool:
        """Evaluates when to leave the loop.
        Returns True if trainer.should_stop was set (e.g. by early stopping) or if the maximum number of steps or epochs
        is reached.
        """
        # TODO(@awaelchli): Move track steps inside training loop and move part of these condition inside training loop
        stop_steps = _is_max_limit_reached(self.global_step, self.max_steps)
        stop_epochs = _is_max_limit_reached(self.current_epoch, self.max_epochs)

        should_stop = False
        
        if self.trainer.should_stop:
            # early stopping
            met_min_epochs = self.current_epoch >= self.min_epochs if self.min_epochs else True
            met_min_steps = self.global_step >= self.min_steps if self.min_steps else True
            if met_min_epochs and met_min_steps:
                should_stop = True
            else:
                log.info(
                    "Trainer was signaled to stop but required minimum epochs"
                    f" ({self.min_epochs}) or minimum steps ({self.min_steps}) has"
                    " not been met. Training will continue..."
                )
        self.trainer.should_stop = should_stop

        return stop_steps or should_stop or stop_epochs or self.trainer.num_training_batches == 0

    @property
    def skip(self) -> bool:
        """Whether we should skip the training and immediately return from the call to :meth:`run`."""
        # since `trainer.num_training_batches` depends on the `train_dataloader` but that won't be called
        # until `on_run_start`, we use `limit_train_batches` instead
        return self.done or self.trainer.limit_train_batches == 0

    def connect(self, epoch_loop: IPL_TrainingEpochLoop):
        """Connects a training epoch loop to this fit loop."""
        self.epoch_loop = epoch_loop

    def reset(self) -> None:
        """Resets the internal state of this loop."""
        if self.restarting:
            self.epoch_progress.reset_on_restart()

    def prepare_cache(self):
        print(f"\n ========================================== \n Initializing IPL Cache... \n Cache size: {self.cache_size}\n ========================================== \n")  

        """self.trainer.predict_dataloaders = []
        loop = IPL_PredictionEpochLoop()
        loop.trainer = self.trainer
        dl_predictions, dl_batch_indices = loop.run(enumerate(self.trainer.model.test_dataloader()), 
                                                                0, self.cache_size, 1, True)
        print(dl_predictions)"""
        
#         alphabet = [" ", "а", "б", "в", "г", "д", "ё", "е", "ж", "з", "и", "й", "к", "л",
#                    "м", "н", "о", "п", "р", "с", "т", "у", "ф", "х", "ц", "ч", "ш", "щ",
#                    "ь", "ы", "ъ", "э", "ю", "я"]
        
        alphabet = [" ", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
                   "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "'"]
        
        self.trainer.model.cache = []        
        self.trainer.model.training = False
        self.trainer.model.test_dataloader().dataset.return_sample_id = True
        with torch.no_grad():
            for batch_idx, batch in tqdm(enumerate(self.trainer.model.test_dataloader())):            
                batch = self.trainer.accelerator.batch_to_device(batch)    

                results = self.trainer.model._predict_step(batch, batch_idx)
                cache_batch = _speech_collate_fn([(batch[0][i][:batch[1][i].item()].cpu(), 
                                batch[1][i].cpu(), 
                                results[i].y_sequence, \
                                len(results[i].y_sequence)) for i in range(len(batch[0]))], 0)
                print(f"\nCache batch: \n{[b.text for b in results]}")
#                 print(cache_batch)
                
#                 for b in results:
#                     for c in b.text:
#                         if c not in alphabet:
#                             print(f"\nUnknown symbol: {c}")
#                             print(b)
                
                is_bad = False
                for i in range(len(batch[0])):
                        res_text = results[i].text
                        unk_sym = False
                        if len(res_text) == 0:                    
                            is_bad = True
                            print("\nThis batch is bad! Empty prediction")
                            break
                        for c in res_text:
                            if c not in alphabet:
                                unk_sym = True
                                print("\nThis batch is bad! Unknown symbol")
                                break
                        if unk_sym:
                            is_bad = True
                            break
                if is_bad:
                    continue
                self.trainer.model.cache.append(cache_batch)

                if len(self.trainer.model.cache) >= self.cache_size:
                    break
        self.trainer.model.training = True
        torch.cuda.empty_cache()
        print("\n ========================================== \n Finished Initializing IPL Cache... \n ========================================== \n")
        
#         # logging cache samples  
#         with open(self.output_file, 'w') as out:
#             items = random.choices(self.trainer.model.cache, k=2)
#             out.write("\n ========================================== \n CACHE INITIALIZED \n ========================================== \n")
#             for item in items:
#                 for i in item[2]:
#                     out.write(self.tok_model.ids_to_text(i.tolist())); out.write('\n')
        
    def on_run_start(self) -> None:
        """Calls the ``on_train_start`` hook."""
        # reset train dataloader and val dataloader
        self.trainer.reset_train_val_dataloaders(self.trainer.lightning_module)
        self._is_fresh_start_epoch = True
        self._results.to(device=self.trainer.lightning_module.device)
        self.trainer.call_hook("on_train_start")
        
        self.prepare_cache()        

    def on_advance_start(self) -> None:
        """Prepares the dataloader for training and calls the hooks ``on_epoch_start`` and
        ``on_train_epoch_start``"""
        model = self.trainer.lightning_module

        # reset train dataloader
        if not self._is_fresh_start_epoch and self.trainer._should_reload_dl_epoch:
            self.trainer.reset_train_dataloader(model)
        self._is_fresh_start_epoch = False

        if self.trainer.train_dataloader is not None and callable(
                getattr(self.trainer.train_dataloader.sampler, "set_epoch", None)
        ):
            # set seed for distributed sampler (enables shuffling for each epoch)
            self.trainer.train_dataloader.sampler.set_epoch(self.current_epoch)

        # changing gradient according accumulation_scheduler
        self.trainer.accumulation_scheduler.on_train_epoch_start(self.trainer, self.trainer.lightning_module)

        # stores accumulated grad fractions per batch
        self.epoch_loop.batch_loop.accumulated_loss = TensorRunningAccum(
            window_length=self.trainer.accumulate_grad_batches
        )

        self.epoch_progress.increment_ready()

    def advance(self) -> None:
        """Runs one whole epoch."""
        dataloader = self.trainer.training_type_plugin.process_dataloader(self.trainer.train_dataloader)
        data_fetcher = self.trainer._data_connector.get_profiled_dataloader(dataloader)

        with self.trainer.profiler.profile("run_training_epoch"):
            
            self.epoch_loop.run(data_fetcher)

            # the global step is manually decreased here due to backwards compatibility with existing loggers
            # as they expect that the same step is used when logging epoch end metrics even when the batch loop has
            # finished. this means the attribute does not exactly track the number of optimizer steps applied.
            # TODO(@carmocca): deprecate and rename so users don't get confused
            self.global_step -= 1
            # log epoch metrics
            self.trainer.logger_connector.update_train_epoch_metrics()
            self.global_step += 1

    def on_advance_end(self) -> None:
        self.epoch_progress.increment_completed()

    def on_run_end(self) -> None:
        """Calls the ``on_train_end`` hook."""
        # NOTE: the current_epoch is already incremented
        # Lightning today does not increment the current epoch at the last epoch run in Trainer.fit
        # To simulate that current behavior, we decrement here.
        # TODO: must be fixed by https://github.com/PyTorchLightning/pytorch-lightning/issues/5007
        self.current_epoch = max(self.current_epoch - 1, 0)

        # hook
        self.trainer.call_hook("on_train_end")

        # give accelerators a chance to finish
        self.trainer.training_type_plugin.on_train_end()

    def teardown(self) -> None:
        self.epoch_loop.teardown()

    def _should_accumulate(self) -> bool:
        """Whether the gradients should be accumulated."""
        return self.epoch_loop._should_accumulate()


def _convert_optim_dict(outs: Dict[int, Dict[str, Any]], num_optimizers: int) -> List[Dict[str, Any]]:
    """Converts an optimizer dict to a list in which the key of the dict determines the position of the element.
    Example::
        >>> _convert_optim_dict({0: {"loss": 0.0}, 2: {"loss": 0.2}}, num_optimizers=3)
        [{'loss': 0.0}, None, {'loss': 0.2}]
    """
    return [outs[opt_idx] if opt_idx in outs else None for opt_idx in range(num_optimizers)]


@overload
def _recursive_unpad(nested: Any, value: Optional[Any] = None) -> Any:
    ...


@overload
def _recursive_unpad(nested: List[Any], value: Optional[Any] = None) -> List[Any]:
    ...


def _recursive_unpad(nested: Union[Any, List[Any]], value: Optional[Any] = None) -> Union[Any, List[Any]]:
    """Removes the given pad value from the nested list. Not strictly the reverse operation of
    :func:`_recursive_pad` because it removes the padding element everywhere, not just from the end of a list.
    Example::
        >>> _recursive_unpad([[[0, 1, 0]], [2], [0, 0]], value=0)
        [[[1]], [2], []]
    """
    if not isinstance(nested, list):
        return nested

    return [_recursive_unpad(item, value) for item in nested if item != value]


def _recursive_pad(nested: List[Any], fill_value: Optional[Any] = None) -> np.array:
    """Pads a jagged nested list of lists with the given value such that a proper multi-dimensional array can be
    formed with rectangular shape. The padding appends to the incomplete lists.
    Example::
        >>> _recursive_pad([[], [1], [2, 3], [4]], fill_value=0)  # doctest: +NORMALIZE_WHITESPACE
        array([[0, 0], [1, 0], [2, 3], [4, 0]], dtype=object)
    """
    # code adapted from stackexchange:
    # https://codereview.stackexchange.com/questions/222623/pad-a-ragged-multidimensional-array-to-rectangular-shape
    dimensions = _get_max_shape(nested)
    result = np.full(dimensions, fill_value, dtype=object)
    for index, value in _iterate_nested_array(nested):
        result[index] = value
    return result


def _get_dimensions(array: List[Any], level: int = 0) -> Generator:
    yield level, len(array)
    if all(isinstance(row, list) for row in array):
        for row in array:
            yield from _get_dimensions(row, level + 1)


def _get_max_shape(array: List[Any]) -> List[int]:
    """Calculates the max size in each dimension of a jagged (non-rectangular) nested list of lists.
    Example::
        >>> _get_max_shape([[], [[1], [2]], []])
        [3, 2, 1]
    """
    dimensions = defaultdict(int)
    for level, length in _get_dimensions(array):
        dimensions[level] = max(dimensions[level], length)
    return [value for _, value in sorted(dimensions.items())]


def _iterate_nested_array(array: List[Any], index: Tuple = ()) -> Generator:
    if all(isinstance(item, list) for item in array):
        for idx, row in enumerate(array):
            yield from _iterate_nested_array(row, (*index, idx))
    else:  # final level
        yield (*index, slice(len(array))), array
        

