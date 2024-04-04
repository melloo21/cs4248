import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from TextNumericalization import *
from TextCNN import *
from utils import build_vocab_and_labels, load_data
from sklearn.metrics import classification_report, f1_score, accuracy_score

def evaluate_model(model, test_loader):
    model.eval()
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for texts, labels in test_loader:
            outputs = model(texts)
            _, predicted = torch.max(outputs.data, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    report = classification_report(all_labels, all_predictions)
    print(report)

vocab, label_dict = build_vocab_and_labels('./train.jsonl')

texts_test, labels_test = load_data('./test.jsonl', vocab, label_dict)

test_dataset = Text_Numericalization(texts_test, labels_test, vocab, label_dict)

test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vocab_size = len(vocab)
embed_dim = 100
num_classes = len(label_dict)
num_filters = 100
kernel_sizes = [3, 4, 5]

model = TextCNN(vocab_size, embed_dim, num_classes, num_filters, kernel_sizes)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

model.load_state_dict(torch.load("./cp/last_model.pth"))
model.to(device)

evaluate_model(model, test_loader)
