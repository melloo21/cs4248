import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from config import *


# Define Byte Pair Encoding (BPE) tokenizer
class BPETokenizer:
    def __init__(self, vocab_size=10000):
        self.vocab_size = vocab_size
        self.tokenizer = None

    def fit(self, texts):
        # Perform BPE tokenization
        vocab = Counter()
        for text in texts:
            vocab.update(text.split())
        self.tokenizer = {word: idx for idx, (word, _) in enumerate(vocab.most_common(self.vocab_size))}
        self.tokenizer['<unk>'] = self.vocab_size  # unknown token
        self.tokenizer['<pad>'] = self.vocab_size + 1  # padding token

    def encode(self, text):
        if self.tokenizer is None:
            raise ValueError("Tokenizer not fitted yet!")
        tokens = [self.tokenizer.get(token, self.tokenizer['<unk>']) for token in text.split()]
        return tokens

# Define dataset class
class TextDataset(Dataset):
    def __init__(self, data_path, tokenizer):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.data = self.load_data()
        self.fit_tokenizer([item['string'] for item in self.data])

    def load_data(self):
        with open(self.data_path, 'r') as file:
            data = [json.loads(line.strip()) for line in file]
        return data

    def fit_tokenizer(self, texts):
        self.tokenizer.fit(texts)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['string']
        label = item['label']
        encoded_text = self.tokenizer.encode(text)
        return encoded_text, label

def collate_fn(batch):
    inputs, labels = zip(*batch)
    inputs = [torch.tensor(seq, dtype=torch.float32) for seq in inputs]
    inputs = pad_sequence(inputs, batch_first=True, padding_value=tokenizer.vocab_size+1)  # Padding
    labels = torch.tensor([0 if label == 'background' else 1 if label == 'method' else 2 for label in labels])
    return inputs, labels


# Initialize BPE tokenizer
tokenizer = BPETokenizer(vocab_size=vocab_size)

