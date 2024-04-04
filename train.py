import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from TextNumericalization import *
from TextCNN import *
from utils import build_vocab_and_labels, load_data
from sklearn.metrics import classification_report, f1_score, accuracy_score
import os

vocab, label_dict = build_vocab_and_labels('./train.jsonl')

texts_train, labels_train = load_data('./train.jsonl', vocab, label_dict)
texts_dev, labels_dev = load_data('./dev.jsonl', vocab, label_dict)
texts_test, labels_test = load_data('./test.jsonl', vocab, label_dict)

train_dataset = Text_Numericalization(texts_train, labels_train, vocab, label_dict)
dev_dataset = Text_Numericalization(texts_dev, labels_dev, vocab, label_dict)
test_dataset = Text_Numericalization(texts_test, labels_test, vocab, label_dict)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

vocab_size = len(vocab)
embed_dim = 100
num_classes = len(label_dict)
num_filters = 100
kernel_sizes = [3, 4, 5]
model = TextCNN(vocab_size, embed_dim, num_classes, num_filters, kernel_sizes)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

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
def train_model(model, train_loader, dev_loader, criterion, optimizer, num_epochs=10, checkpoint_dir=".", device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    best_accuracy = 0.0
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for data, labels in train_loader:
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * data.size(0)

        avg_loss = total_loss / len(train_loader.dataset)

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, labels in dev_loader:
                data, labels = data.to(device), labels.to(device)
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}, Validation Accuracy: {accuracy:.4f}")

        # 如果当前epoch的准确率是最佳的，保存模型
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_path = os.path.join(checkpoint_dir, "best_model.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved to {best_model_path} with Accuracy: {best_accuracy:.4f}")
    last_model_path = os.path.join(checkpoint_dir, "last_model.pth")
    torch.save(model.state_dict(), last_model_path)

train_model(model, train_loader, dev_loader, criterion, optimizer, num_epochs=8, checkpoint_dir="./cp")

evaluate_model(model, test_loader)