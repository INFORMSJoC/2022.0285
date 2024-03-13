from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer
from utils.utils import *
import os
import json
import pickle
import sys
sys.path.append("../../")
class ClassificationDataset(Dataset):
    def __init__(self, config, prefix):
        self.config = config
        self.tokenizer = BertTokenizer.from_pretrained(os.path.join(self.config.model_dir, self.config.model_name))
        self.tokenizer.pair_token = "[pair]"
        self.subreddit_dict = pickle.load(open(os.path.join('../data/reddit', "subreddit_dict.pkl"), "rb"))
        self.data = json.load(open(os.path.join(config.data_path, prefix + "_ids.json")))
        if self.config.debug:
            self.data = self.data[:100]

    def __getitem__(self, idx):
        data = self.data[idx]
        # input_ids, attention_mask, token_type_ids, clause_idx, clause_labels
        input_dict = encode(self.tokenizer, self.subreddit_dict, data, max_length=self.config.max_len, clause_max_length=self.config.clause_max_len)
        return input_dict

    def __len__(self):
        return len(self.data)

def cmed_collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    input_dict = batch
    cur_batch = len(batch)
    max_text_len = max(map(lambda x: len(x["input_ids"]), batch))
    max_clause_len = max(map(lambda x: len(x["subreddit_input_ids"]), batch))
    max_idx_len = max(map(lambda x: len(x["clause_idx"]), batch))
    max_label_len = max(map(lambda x: len(x["clause_labels"]), batch))
    assert max_idx_len == max_label_len

    batch_token_ids = torch.LongTensor(cur_batch, max_text_len).zero_()
    batch_attention_mask = torch.LongTensor(cur_batch, max_text_len).zero_()
    batch_input_mask = torch.LongTensor(cur_batch, max_text_len).zero_()
    batch_token_type_ids = torch.LongTensor(cur_batch, max_text_len).zero_()
    batch_position_ids = torch.LongTensor(cur_batch, max_text_len).zero_()
    batch_subreddit_token_ids = torch.LongTensor(cur_batch, max_clause_len).zero_()
    batch_subreddit_attention_mask = torch.LongTensor(cur_batch, max_clause_len).zero_()
    batch_subreddit_token_type_ids = torch.LongTensor(cur_batch, max_clause_len).zero_()
    batch_clause_idx = torch.LongTensor(cur_batch, max_idx_len).zero_()
    batch_sarcasm_idx = torch.LongTensor(cur_batch, 1).zero_()
    batch_subreddit_idx = torch.LongTensor(cur_batch, 1).zero_()
    batch_clause_labels = torch.LongTensor(cur_batch, max_label_len).zero_().fill_(-100)
    batch_subreddit_position_ids = torch.LongTensor(cur_batch, max_clause_len).zero_()

    for i in range(cur_batch):
        batch_token_ids[i, :len(input_dict[i]["input_ids"])].copy_(input_dict[i]["input_ids"])
        batch_attention_mask[i, :len(input_dict[i]["attention_mask"])].copy_(input_dict[i]["attention_mask"])
        batch_input_mask[i, :len(input_dict[i]["input_mask"])].copy_(input_dict[i]["input_mask"])
        batch_token_type_ids[i, :len(input_dict[i]["token_type_ids"])].copy_(input_dict[i]["token_type_ids"])
        batch_position_ids[i, :len(input_dict[i]["position_ids"])].copy_(input_dict[i]["position_ids"])
        batch_subreddit_token_ids[i, :len(input_dict[i]["subreddit_input_ids"])].copy_(input_dict[i]["subreddit_input_ids"])
        batch_subreddit_attention_mask[i, :len(input_dict[i]["subreddit_attention_mask"])].copy_(input_dict[i]["subreddit_attention_mask"])
        batch_subreddit_token_type_ids[i, :len(input_dict[i]["subreddit_token_type_ids"])].copy_(input_dict[i]["subreddit_token_type_ids"])
        batch_subreddit_position_ids[i, :len(input_dict[i]["subreddit_position_ids"])].copy_(input_dict[i]["subreddit_position_ids"])
        batch_subreddit_idx[i, :].copy_(input_dict[i]["subreddit_idx"])
        batch_clause_idx[i, :len(input_dict[i]["clause_idx"])].copy_(input_dict[i]["clause_idx"])
        batch_sarcasm_idx[i, :].copy_(input_dict[i]["sarcasm_idx"])
        batch_clause_labels[i, :len(input_dict[i]["clause_labels"])].copy_(input_dict[i]["clause_labels"])
    return {"input_ids": batch_token_ids,
            "attention_mask": batch_attention_mask,
            "input_mask": batch_input_mask,
            "token_type_ids": batch_token_type_ids,
            "position_ids": batch_position_ids,
            "subreddit_input_ids": batch_subreddit_token_ids,
            "subreddit_attention_mask": batch_subreddit_attention_mask,
            "subreddit_token_type_ids": batch_subreddit_token_type_ids,
            "subreddit_position_ids": batch_subreddit_position_ids,
            "clause_idx": batch_clause_idx,
            "sarcasm_idx": batch_sarcasm_idx,
            "subreddit_idx": batch_subreddit_idx,
            "clause_labels": batch_clause_labels
            }

def get_loader(config, prefix="train", num_workers=0, collate_fn=cmed_collate_fn):
    dataset = ClassificationDataset(config, prefix)
    if prefix == "train":
        data_loader = DataLoader(dataset=dataset,
                                 batch_size=config.batch_size,
                                 shuffle=True,
                                 pin_memory=True,
                                 num_workers=num_workers,
                                 collate_fn=collate_fn)
    else:
        data_loader = DataLoader(dataset=dataset,
                                 batch_size=config.batch_size * 4,
                                 shuffle=False,
                                 pin_memory=True,
                                 num_workers=num_workers,
                                 collate_fn=collate_fn)
    return data_loader

class DataPreFetcher(object):
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_data = next(self.loader)
        except StopIteration:
            self.next_data = None
            return
        with torch.cuda.stream(self.stream):
            for k, v in self.next_data.items():
                if isinstance(v, torch.Tensor):
                    self.next_data[k] = self.next_data[k].cuda(non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        data = self.next_data
        self.preload()
        return data

