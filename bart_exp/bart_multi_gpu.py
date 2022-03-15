import os 
import glob
import copy
import json
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf

import torch
from sklearn.model_selection import train_test_split
from transformers import Trainer, TrainingArguments, AutoTokenizer, BartForConditionalGeneration, DataCollatorWithPadding, BartTokenizer

def decode_output(predicts, manifest, tokenizer):
    logits = predicts.predictions[0]
    label_ids = predicts.label_ids
    
    idx = 0
    
    with open('test.txt', 'w', encoding='utf-8') as out:
        for logit, label in zip(logits, label_ids):
            softmax_preds = softmax(logit)
            ids = [np.argmax(score) for score in logit]
            print(ids)
            # for _id in ids:
            #     print(f"{_id} | {tokenizer.decode([_id])}")
            dec_pred = tokenizer.decode(ids, skip_special_tokens=True)
            inpt = manifest[idx]['pred_text']
            true_lbl = manifest[idx]['text']
            print(f"INPUT: {inpt}")
            print(f"REF  : {true_lbl}")
            print(f"PRED : {dec_pred}")
            # out.write(dec_pred + '\n')
            print("-----------------------------------------------------------")
            idx += 1
    
def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum(axis=-1).reshape([x.shape[0], 1])

class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def load_manifest(filename):
    with open(filename) as f:
        content = f.readlines()

    manifest = [json.loads(t) for t in content]
    return manifest

def main():
    # model = BartForConditionalGeneration.from_pretrained('/nas/vkriger/bart_test/finetuning_bart_ru_model_fix/checkpoint-1000') # trained checkpoint for facebook/bart-base
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')
    print("Got model")
    
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
    print(f"Got tokenizer: {tokenizer.name_or_path}")

    m_path = '/nas/vkriger/bart_test/train_dataset.json'
    m = load_manifest(m_path)
    m = [s for s in m if s['duration'] <= 25.0]
    
    train_texts, train_labels = [], []

    for sample in m:
        train_texts.append(sample['pred_text'])
        train_labels.append(sample['text'])

    train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=.01)

    print(f"Train texts: {len(train_texts)}\nVal texts: {len(val_texts)}\nTrain labels: {len(train_labels)}\nVal labels: {len(val_labels)}")

    print("Started tokenization...")

    train_texts_encodings = tokenizer(train_texts, truncation=True, padding='max_length', max_length=300)
    val_texts_encodings = tokenizer(val_texts, truncation=True, padding='max_length', max_length=300)
    train_labels_encodings = tokenizer(train_labels, truncation=True, padding='max_length', max_length=300)['input_ids']
    val_labels_encodings = tokenizer(val_labels, truncation=True, padding='max_length', max_length=300)['input_ids']
    
    # print(val_texts_encodings)
    # print(val_labels_encodings)

    print("Done tokenization!")

    # ==== end new part ====

    train_dataset = Dataset(train_texts_encodings, train_labels_encodings)
    val_dataset = Dataset(val_texts_encodings, val_labels_encodings)
    
    print("Starting training...")
    training_args = TrainingArguments(
        output_dir='./finetuning_bart_ru_gpu',          # output directory
        num_train_epochs=15,              # total number of training epochs
        per_device_train_batch_size=8,  # batch size per device during training
        per_device_eval_batch_size=8,   # batch size for evaluation
        warmup_steps=5000,                # number of warmup steps for learning rate scheduler
        weight_decay=0.001,               # strength of weight decay
        logging_dir='./logs',            # directory for storing logs
        logging_steps=100,
        evaluation_strategy='steps',
        eval_steps=15000,
        gradient_accumulation_steps=2,
        sharded_ddp='simple'
        # do_predict=True,
        # do_eval=True
        # no_cuda=True
    )
    
    trainer = Trainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=train_dataset,         # training dataset
        eval_dataset=val_dataset,          # evaluation dataset
        tokenizer=tokenizer,
    )
    
    trainer.train()
    # predicts = trainer.predict(test_dataset=pred_dataset) 
    # decode_output(predicts, m, tokenizer)

    
if __name__ == '__main__':
    main()
    
# RAYON_RS_NUM_CPUS=5 OMP_NUM_THREADS=4 NCCL_P2P_DISABLE=1 python3 -m torch.distributed.launch --nproc_per_node=4 bart_multi_gpu.py
