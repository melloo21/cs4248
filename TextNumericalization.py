import torch
from torch.utils.data import Dataset
class Text_Numericalization(Dataset):
    def __init__(self, texts, labels, vocab, label_dict, max_length=256):
        self.texts = [self.encode(text.split(), vocab, max_length) for text in texts]
        self.labels = [label_dict[label] for label in labels]

    def encode(self, tokens, vocab, max_length):
        return [vocab.get(token, vocab["<unk>"]) for token in tokens][:max_length] + [vocab["<pad>"]] * \
                    (max_length - len(tokens))

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return torch.tensor(self.texts[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)