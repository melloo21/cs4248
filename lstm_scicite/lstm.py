import torch
import torch.nn as nn

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define LSTM model

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x:(batch_size, sequence_length, input_size)
        # h:(num_layers, batch_size, hidden_size)
        x = x.unsqueeze(2)
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


class BiLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BiLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes) 
    def forward(self, x):
        # x:(batch_size, sequence_length, input_size)
        # h:(num_layers * num_directions, batch_size, hidden_size)
        x = x.unsqueeze(2)
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(device)  
        c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMWithSelfAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMWithSelfAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

        self.W_s1 = nn.Linear(hidden_size, 10) # attention size = 10
        self.W_s2 = nn.Linear(10, 1)

    def forward(self, x):
        # x:(batch_size, sequence_length, input_size)
        # h:(num_layers, batch_size, hidden_size)

        # LSTM layer
        # h_lstm, _ = self.lstm(x)
        x = x.unsqueeze(2)
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        h_lstm, _ = self.lstm(x, (h0, c0))

        # Self-attention mechanism
        u = torch.tanh(self.W_s1(h_lstm))
        a = F.softmax(self.W_s2(u), dim=1)
        attended_h = torch.sum(a * h_lstm, dim=1)

        # Fully connected layer
        out = self.fc(attended_h)
        return out
