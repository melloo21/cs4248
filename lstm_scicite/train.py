from dataloader import *
from lstm import *
import os
from torch.optim import lr_scheduler
from sklearn.metrics import f1_score
from transformers import BertModel



# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LSTMWithSelfAttention(vocab_size=vocab_size, embedding_dim=embedding_dim, hidden_size=hidden_size, num_layers=num_layers, num_classes=num_classes).to(device)


# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# Load train dataset
train_data = TextDataset('dataset/train.jsonl', tokenizer)

# Data loader
train_loader = DataLoader(dataset=train_data,
                          batch_size=batch_size,
                          shuffle=True,
                          collate_fn=collate_fn)

# Load dev dataset
dev_data = TextDataset('dataset/dev.jsonl', tokenizer)
dev_loader = DataLoader(dataset=dev_data,
                        batch_size=batch_size,
                        shuffle=False,
                        collate_fn=collate_fn)


checkpoint_dir = "checkpoint"
# model.load_state_dict(torch.load("checkpoint/last.pth"))

best_accuracy = 0.0
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
        inputs, labels = batch
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(inputs)

        # Calculate loss
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    scheduler.step()

    # Evaluate on dev set
    model.eval()
    correct = 0
    total = 0

    dev_labels = []
    dev_predictions = []
    with torch.no_grad():
        for batch in dev_loader:
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            dev_labels.extend(labels.cpu().numpy())
            dev_predictions.extend(predicted.cpu().numpy())

    accuracy = correct / total
    f1 = f1_score(dev_labels, dev_predictions, average='macro')

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}, Validation Accuracy: {accuracy:.4f}, Validation macro-f1: {f1:.4f}')

    # Save best model
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best.pth"))

# Save last model
torch.save(model.state_dict(), os.path.join(checkpoint_dir, "last.pth"))
