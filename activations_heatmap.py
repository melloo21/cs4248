import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import sentencepiece as spm
from sklearn.metrics import classification_report
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import numpy as np

def visualize_activations(model, sentence, sp, vocab, device):
    tokens = sp.encode_as_pieces(sentence)
    encoded = [vocab.get(token, vocab["<unk>"]) for token in tokens]
    encoded = torch.tensor([encoded], device=device)

    activations = model(encoded, return_activations=True)
    act = activations[0].squeeze(0).cpu().detach().numpy()

    num_kernels_to_display = min(5, act.shape[0])

    fig, axes = plt.subplots(num_kernels_to_display, 1, figsize=(max(10, len(tokens)), num_kernels_to_display))

    if num_kernels_to_display == 1:
        axes = [axes]

    for i in range(num_kernels_to_display):
        ax = axes[i]
        ax.imshow(act[i:i + 1], cmap='hot', aspect='auto')
        ax.set_yticks([])
        ax.set_xticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=90, ha='center', fontsize=8)

    plt.tight_layout()
    plt.show()


def load_data(file_path):
    texts, labels = [], []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            texts.append(data["string"])
            labels.append(data["label"])
    return texts, labels

def build_vocab_and_labels(data_path, sp):
    labels_set = set()

    with open(data_path, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            labels_set.add(data["label"])

    vocab = {"<pad>": 0, "<unk>": 1}
    for idx in range(sp.get_piece_size()):
        token = sp.id_to_piece(idx)
        vocab[token] = idx + 2

    label_dict = {label: idx for idx, label in enumerate(labels_set)}
    return vocab, label_dict

class Text_Numericalization(Dataset):
    def __init__(self, texts, labels, vocab, label_dict, sp, max_length=256):
        self.texts = [self.encode(text, sp, vocab, max_length) for text in texts]
        self.labels = [label_dict[label] for label in labels]

    def encode(self, text, sp, vocab, max_length):
        encoded = sp.encode_as_pieces(text)
        encoded_ids = [vocab.get(token, vocab["<unk>"]) for token in encoded][:max_length]
        encoded_ids += [vocab["<pad>"]] * (max_length - len(encoded_ids))
        return encoded_ids

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return torch.tensor(self.texts[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes, num_filters, kernel_sizes):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, kernel_size) for kernel_size in kernel_sizes
        ])
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, x, return_activations=False):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        activations = [torch.relu(conv(x)) for conv in self.convs]
        if return_activations:
            return activations
        x = [torch.max_pool1d(i, i.size(2)).squeeze(2) for i in activations]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

def evaluate_model(model, test_loader, device = None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            # print(f"Max index in data: {data.max().item()}, vocab_size: {vocab_size}")
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    report = classification_report(all_labels, all_predictions)
    print(report)

sp = spm.SentencePieceProcessor()
sp.load('sp_model.model')

vocab, label_dict = build_vocab_and_labels('./train.jsonl', sp)

texts_test, labels_test = load_data('./test.jsonl')
test_dataset = Text_Numericalization(texts_test, labels_test, vocab, label_dict, sp)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

vocab_size = len(vocab) + 2
embed_dim = 100
num_classes = 3
num_filters = 100
kernel_sizes = [3, 4, 5]

model = TextCNN(vocab_size, embed_dim, num_classes, num_filters, kernel_sizes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.load_state_dict(torch.load("./cp/last_model_sp.pth"))

sample_text = "In addition, the result of the present study supports previous studies, which did not find increased rates of first-born children among individual with OCD (20,31,34)."
visualize_activations(model, sample_text, sp, vocab, device)