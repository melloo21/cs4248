from collections import Counter
import json

def load_data(file_path, vocab, label_dict):
    texts, labels = [], []
    with open(file_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            texts.append(data["string"])
            labels.append(data["label"])
    return texts, labels

def build_vocab_and_labels(data_path):
    word_counter = Counter()
    labels_set = set()
    with open(data_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            words = data["string"].split()
            word_counter.update(words)
            labels_set.add(data["label"])
    vocab = {"<pad>": 0, "<unk>": 1}
    for word, _ in word_counter.items():
        vocab[word] = len(vocab)
    label_dict = {label: idx for idx, label in enumerate(labels_set)}
    return vocab, label_dict
