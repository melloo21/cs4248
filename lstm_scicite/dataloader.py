import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from config import *
from transformers import BertTokenizer
import re
import pandas as pd

class TextDataset(Dataset):
    def __init__(self, data_path, tokenizer):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.data = self.load_data()

    def load_data(self):
        with open(self.data_path, 'r') as file:
            data = [json.loads(line.strip()) for line in file]
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['string']
        label = item['label']

        # text = re.sub(r'\d+(\.\d+)?', '<NUM>', text)
        # if pd.notna(item['citeStart']) and pd.notna(item['citeEnd']):
        #     text = text[:int(item['citeStart'])] + "<CITATION>" + text[int(item['citeEnd']):]

        encoded_text = self.tokenizer.encode(text, add_special_tokens=True)
        return encoded_text, label

# Initialize BPE tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
vocab_size = tokenizer.vocab_size  # 从tokenizer获取词汇表大小


def collate_fn(batch):
    inputs, labels = zip(*batch)
    inputs = [torch.tensor(seq, dtype=torch.long) for seq in inputs]
    inputs = pad_sequence(inputs, batch_first=True, padding_value=tokenizer.pad_token_id)
    # inputs = inputs.to(torch.float)
    labels = torch.tensor([0 if label == 'background' else 1 if label == 'method' else 2 for label in labels])
    return inputs, labels
