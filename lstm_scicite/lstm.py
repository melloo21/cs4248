import torch
import torch.nn as nn
import torch.nn.functional as F

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define LSTM model
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, num_classes):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)  # 添加嵌入层
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.embedding(x)  # x:(batch_size, sequence_length) -> (batch_size, sequence_length, embedding_dim)

        # h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        # c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        # out, _ = self.lstm(x, (h0, c0))
        
        out, _ = self.lstm(x)  # out:(batch_size, sequence_length, hidden_size)

        out = self.fc(out[:, -1, :])  # out:(batch_size, num_classes)
        return out


class BiLSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, num_classes):
        super(BiLSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        x = self.embedding(x)  # x: (batch_size, sequence_length) -> (batch_size, sequence_length, embedding_dim)
        out, _ = self.lstm(x)  # out: (batch_size, sequence_length, hidden_size * 2)
        out = self.fc(out[:, -1, :])  # out: (batch_size, num_classes)
        return out



class LSTMWithSelfAttention(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, num_classes):
        super(LSTMWithSelfAttention, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)  # 添加嵌入层
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.W_s1 = nn.Linear(hidden_size, 10)
        self.W_s2 = nn.Linear(10, 1)

    def forward(self, x):
        x = x.to(torch.long)
        x = self.embedding(x)  

        # LSTM layer
        h_lstm, _ = self.lstm(x)

        # Self-attention mechanism
        u = torch.tanh(self.W_s1(h_lstm))
        a = self.W_s2(u).squeeze(2)
        a = F.softmax(a, dim=1)
        a = a.unsqueeze(2)
        attended_h = torch.sum(a * h_lstm, dim=1)

        # Fully connected layer
        out = self.fc(attended_h)
        return out

class BiLSTMWithSelfAttention(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, num_classes):
        super(BiLSTMWithSelfAttention, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)  
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        self.W_s1 = nn.Linear(hidden_size * 2, 10)  
        self.W_s2 = nn.Linear(10, 1)

    def forward(self, x):
        x = self.embedding(x)
        h_lstm, _ = self.lstm(x)

        u = torch.tanh(self.W_s1(h_lstm))
        a = self.W_s2(u).squeeze(2)
        a = F.softmax(a, dim=1)
        a = a.unsqueeze(2)
        attended_h = torch.sum(a * h_lstm, dim=1)

        out = self.fc(attended_h)
        return out
