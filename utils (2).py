import random
import json
import glob
import os
from omegaconf import OmegaConf

import torch
import nemo.collections.asr as nemo_asr
import pytorch_lightning as pl

def _speech_collate_fn(batch, pad_id):
    """collate batch of audio sig, audio len, tokens, tokens len
    Args:
        batch (Optional[FloatTensor], Optional[LongTensor], LongTensor,
               LongTensor):  A tuple of tuples of signal, signal lengths,
               encoded tokens, and encoded tokens length.  This collate func
               assumes the signals are 1d torch tensors (i.e. mono audio).
    """
    packed_batch = list(zip(*batch))
    if len(packed_batch) == 5:
        _, audio_lengths, _, tokens_lengths, sample_ids = packed_batch
    elif len(packed_batch) == 4:
        sample_ids = None
        _, audio_lengths, _, tokens_lengths = packed_batch
    else:
        raise ValueError("Expects 4 or 5 tensors in the batch!")
    max_audio_len = 0
    has_audio = audio_lengths[0] is not None
    if has_audio:
        max_audio_len = max(audio_lengths)
    max_tokens_len = max(tokens_lengths)

    audio_signal, tokens = [], []
    for b in batch:
        if len(b) == 5:
            sig, sig_len, tokens_i, tokens_i_len, _ = b
        else:
            sig, sig_len, tokens_i, tokens_i_len = b
        if has_audio:
            sig_len = sig_len.item()            
            if sig_len < max_audio_len:
                pad = (0, max_audio_len - sig_len)
                sig = torch.nn.functional.pad(sig, pad)
            audio_signal.append(sig)
        tokens_i_len = tokens_i_len
        if tokens_i_len < max_tokens_len:
            pad = (0, max_tokens_len - tokens_i_len)
            tokens_i = torch.nn.functional.pad(tokens_i, pad, value=pad_id)
        tokens.append(tokens_i)

    if has_audio:
        audio_signal = torch.stack(audio_signal)
        audio_lengths = torch.stack(audio_lengths)
    else:
        audio_signal, audio_lengths = None, None
    tokens = torch.stack(tokens)
    tokens_lengths = torch.tensor(tokens_lengths)
    if sample_ids is None:
        return audio_signal, audio_lengths, tokens, tokens_lengths
    else:
        sample_ids = torch.tensor(sample_ids, dtype=torch.int32)
        return audio_signal, audio_lengths, tokens, tokens_lengths, sample_ids
    
def get_last_ckpt(c_path):
    ckpts = glob.glob(c_path + '/*')
    if not ckpts:
        return None
    latest_ckpt = max(ckpts, key=os.path.getctime)
    print(f"-------------Latest checkpoint: {latest_ckpt}-------------\n")
    return latest_ckpt


def load_trainer(config):
    return pl.Trainer(**config.trainer)


def load_ckpt(c_path, model):
    if os.path.isdir(c_path):
        ckpt_path = get_last_ckpt(c_path)
    else:
        ckpt_path = c_path
    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt['state_dict'], strict=False)
    return model


def load_model(c_path, config, trainer):
    model = nemo_asr.models.EncDecCTCModel(cfg=config.model, trainer=trainer)
    model = load_ckpt(c_path, model)
    return model


def load_config(config_path):
    config = OmegaConf.load(config_path)
    config = OmegaConf.to_container(config, resolve=True)
    config = OmegaConf.create(config)
    return config


def load_manifest(filename):
    with open(filename) as f:
        content = f.readlines()
    return [json.loads(t) for t in content]

def generate_one_step_manifest(cache_batch, one_step_manifest_path):
    if os.path.exists(one_step_manifest_path):
        os.remove(one_step_manifest_path)

    with open(one_step_manifest_path, 'a') as out:
        for sample in cache_batch:
            json.dump(sample, out)
            out.write('\n')
            
def get_random_batch(dataset, batch_size):
        return random.sample(dataset, k=batch_size)
    
    
def remove_batch(dataset, batch, manifest):
    for sample in batch:
        dataset.remove(sample)
    if os.path.exists(manifest):
        os.remove(manifest)
    with open(manifest, 'a') as out:
        for sample in dataset:
            json.dump(sample, out)
            out.write('\n')
            
def add_batch(dataset, batch, manifest):
    for sample in batch:
        dataset.append(sample)
    with open(manifest, 'a') as out:
        for sample in batch:
            json.dump(sample, out)
            out.write('\n')
    
    