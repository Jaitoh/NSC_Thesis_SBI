import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LSTM_Embedding_Small(nn.Module):
    def __init__(self, 
                 dms, 
                 l, 
                 hidden_size=64,
                 output_size=20,
            ):
        super(LSTM_Embedding_Small, self).__init__()
        self.rnn1 = nn.LSTM(input_size=dms, hidden_size=hidden_size*8, num_layers=1, batch_first=True)
        # self.dropout1 = nn.Dropout(p=0.05)
        self.rnn2 = nn.LSTM(input_size=hidden_size*8, hidden_size=hidden_size*2, num_layers=1, batch_first=True)
        # self.dropout2 = nn.Dropout(p=0.05)
        self.rnn3 = nn.LSTM(input_size=hidden_size*2, hidden_size=hidden_size, num_layers=1, batch_first=True)
        # self.dropout3 = nn.Dropout(p=0.05)
        # self.rnn4 = nn.LSTM(input_size=hidden_size*2, hidden_size=hidden_size, num_layers=1, batch_first=True)
        # self.dropout4 = nn.Dropout(p=0.05)
        self.fc1 = nn.Linear(l*hidden_size, 4*hidden_size)
        # self.bn1 = nn.BatchNorm1d(2*hidden_size)
        self.bn1 = nn.LayerNorm(4*hidden_size)
        self.fc2 = nn.Linear(4*hidden_size, output_size)

    def forward(self, x):
        # x has shape (B, DMS, L)
        x = x.permute(0, 2, 1) # (B, L, DMS) - (batch_size, seq_len, input_size)
        # Pass the first part (x1) through the RNN
        out, _ = self.rnn1(x)  
        # out    = self.dropout1(out)
        out, _ = self.rnn2(out)
        # out    = self.dropout2(ou1t)
        out, _ = self.rnn3(out)
        # out    = self.dropout3(out)
        # out, _ = self.rnn4(out)
        # Flatten the RNN output and pass it through the first FC layer
        out = out.reshape(out.shape[0], -1)
        # out = F.relu(self.fc1(out))
        out = F.leaky_relu(self.fc1(out), negative_slope=0.01)
        out = self.bn1(out)
        # out = self.dropout4(out)
        # out = F.relu(self.fc2(out))
        out = F.leaky_relu(self.fc2(out), negative_slope=0.01)
        return out

class RNN_Embedding_Small(nn.Module):
    def __init__(self, 
                 dms, 
                 l, 
                 hidden_size=64,
                 output_size=20,
            ):
        super(RNN_Embedding_Small, self).__init__()
        self.rnn1 = nn.RNN(input_size=dms, hidden_size=hidden_size*8, batch_first=True)
        # self.dropout1 = nn.Dropout(p=0.05)
        self.rnn2 = nn.RNN(input_size=hidden_size*8, hidden_size=hidden_size*2, batch_first=True)
        # self.dropout2 = nn.Dropout(p=0.05)
        self.rnn3 = nn.RNN(input_size=hidden_size*2, hidden_size=hidden_size, batch_first=True)
        # self.dropout3 = nn.Dropout(p=0.05)
        # self.rnn4 = nn.LSTM(input_size=hidden_size*2, hidden_size=hidden_size, num_layers=1, batch_first=True)
        # self.dropout4 = nn.Dropout(p=0.05)
        self.fc1 = nn.Linear(l*hidden_size, 4*hidden_size)
        # self.bn1 = nn.BatchNorm1d(2*hidden_size)
        self.bn1 = nn.LayerNorm(4*hidden_size)
        self.fc2 = nn.Linear(4*hidden_size, output_size)

    def forward(self, x):
        # x has shape (B, DMS, L)
        x = x.permute(0, 2, 1) # (B, L, DMS) - (batch_size, seq_len, input_size)
        # Pass the first part (x1) through the RNN
        out, _ = self.rnn1(x)  
        # out    = self.dropout1(out)
        out, _ = self.rnn2(out)
        # out    = self.dropout2(ou1t)
        out, _ = self.rnn3(out)
        # out    = self.dropout3(out)
        # out, _ = self.rnn4(out)
        # Flatten the RNN output and pass it through the first FC layer
        out = out.reshape(out.shape[0], -1)
        # out = F.relu(self.fc1(out))
        out = F.leaky_relu(self.fc1(out), negative_slope=0.01)
        out = self.bn1(out)
        # out = self.dropout4(out)
        # out = F.relu(self.fc2(out))
        out = F.leaky_relu(self.fc2(out), negative_slope=0.01)
        return out
    
class LSTM_Embedding(nn.Module):
    def __init__(self, 
                 dms, 
                 l, 
                 hidden_size=64,
                 output_size=20,
            ):
        super(LSTM_Embedding, self).__init__()
        self.rnn1 = nn.LSTM(input_size=dms, hidden_size=hidden_size*8, num_layers=1, batch_first=True)
        # self.dropout1 = nn.Dropout(p=0.05)
        self.rnn2 = nn.LSTM(input_size=hidden_size*8, hidden_size=hidden_size*4, num_layers=1, batch_first=True)
        # self.dropout2 = nn.Dropout(p=0.05)
        self.rnn3 = nn.LSTM(input_size=hidden_size*4, hidden_size=hidden_size*2, num_layers=1, batch_first=True)
        # self.dropout3 = nn.Dropout(p=0.05)
        self.rnn4 = nn.LSTM(input_size=hidden_size*2, hidden_size=hidden_size, num_layers=1, batch_first=True)
        # self.dropout4 = nn.Dropout(p=0.05)
        self.fc1 = nn.Linear(l*hidden_size, 4*hidden_size)
        # self.bn1 = nn.BatchNorm1d(2*hidden_size)
        self.bn1 = nn.LayerNorm(4*hidden_size)
        self.fc2 = nn.Linear(4*hidden_size, output_size)

    def forward(self, x):
        # x has shape (B, DMS, L)
        x = x.permute(0, 2, 1) # (B, L, DMS) - (batch_size, seq_len, input_size)
        # Pass the first part (x1) through the RNN
        out, _ = self.rnn1(x)  
        # out    = self.dropout1(out)
        out, _ = self.rnn2(out)
        # out    = self.dropout2(out)
        out, _ = self.rnn3(out)
        # out    = self.dropout3(out)
        out, _ = self.rnn4(out)
        # Flatten the RNN output and pass it through the first FC layer
        out = out.reshape(out.shape[0], -1)
        # out = F.relu(self.fc1(out))
        out = F.leaky_relu(self.fc1(out), negative_slope=0.01)
        out = self.bn1(out)
        # out = self.dropout4(out)
        # out = F.relu(self.fc2(out))
        out = F.leaky_relu(self.fc2(out), negative_slope=0.01)
        return out
    
class Mixed_LSTM_Embedding(nn.Module):
    def __init__(self, 
                 dms, 
                 l, 
                 hidden_size=64,
                 output_size=20,
            ):
        super(LSTM_Embedding, self).__init__()
        self.rnn1 = nn.LSTM(input_size=dms, hidden_size=dms, num_layers=1, batch_first=True)
        self.rnn2 = nn.LSTM(input_size=dms, hidden_size=hidden_size, num_layers=2, batch_first=True)
        self.bn1 = nn.BatchNorm1d(dms * (l - 1))
        self.fc1 = nn.Linear(dms * (l - 1), 2*dms)
        self.fc2 = nn.Linear(2*dms, hidden_size)
        # self.bn2 = nn.BatchNorm1d(hidden_size)
        
        self.fc3 = nn.Linear(hidden_size + dms, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)
        

    def forward(self, x):
        # x has shape (B, DMS, L)
        x1, x2 = torch.split(x, split_size_or_sections=[x.shape[-1] - 1, 1], dim=-1)
        # x2 has shape (B, DMS, 1)
        x1 = x1.permute(0, 2, 1) # (B, L-1, DMS) - (batch_size, seq_len, input_size)
        
        # Pass the first part (x1) through the RNN
        out, _ = self.rnn1(x1)  
        out, _ = self.rnn2(out)
        
        # Flatten the RNN output and pass it through the first FC layer
        out = out.reshape(out.size(0), -1)
        out = self.bn1(out)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        # out = self.bn2(out)
        
        # Concatenate the FC output with the second part of x (x2)
        out = torch.cat([out, x2.squeeze(dim=-1)], dim=-1)
        
        # Pass the concatenated tensor through the second FC layer
        out = F.relu(self.fc3(out))
        out = F.relu(self.fc4(out))
        
        return out
