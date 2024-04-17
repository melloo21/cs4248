from dataloader import *
from lstm import *
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

# Load test dataset
test_data = TextDataset('dataset/test.jsonl', tokenizer)
test_loader = DataLoader(dataset=test_data,
                         batch_size=1,
                         shuffle=False,
                         collate_fn=collate_fn)

# Initialize LSTM model
model = LSTMWithSelfAttention(vocab_size=vocab_size, embedding_dim=embedding_dim, hidden_size=hidden_size, num_layers=num_layers, num_classes=num_classes).to(device)
model.load_state_dict(torch.load("checkpoint/best.pth"))

# Test the model
model.eval()
predictions = []
labels = []
with torch.no_grad():
    for batch in test_loader:
        inputs, label = batch
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        predictions.extend(predicted.cpu().numpy())
        labels.extend(label.cpu().numpy())

f1_macro = f1_score(labels, predictions, average='macro')
print(f"Macro F1: {f1_macro:.4f}")

report = classification_report(labels, predictions, target_names=['background', 'mathod', 'result'])
print(report)

