import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchsummary import summary

def kaiming_weight_initialization(named_parameters):
    for name, param in named_parameters:
        if 'weight_ih' in name:
            # nn.init.xavier_uniform_(param.data)  # Xavier initialization
            nn.init.kaiming_uniform_(param.data)  # Kaiming initialization
        elif 'weight_hh' in name:
            nn.init.orthogonal_(param.data)
        elif 'bias' in name:
            param.data.fill_(0)

class GRUUnit(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GRUUnit, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)

    def forward(self, x):
        output, _ = self.gru(x)
        return output
    
class ResidualBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(ResidualBlock, self).__init__()
        self.layers = nn.ModuleList([GRUUnit(input_dim if i==0 else hidden_dim, hidden_dim) for i in range(num_layers)])
        self.linear = nn.Linear(input_dim, hidden_dim)

    def forward(self, x):
        residual = self.linear(x)
        for layer in self.layers:
            out = layer(x)
            x = out + residual
            residual = x
        return x

class RNN_Multi_Head(nn.Module):
    def __init__(self, DM, S, L):
        super(RNN_Multi_Head, self).__init__()
        num_layers = 3
        self.blocks1 = nn.ModuleList([ResidualBlock(S, 64, num_layers) for _ in range(DM)])
        self.blocks2 = nn.ModuleList([ResidualBlock(64, 8, num_layers) for _ in range(DM)])
        self.blocks3 = nn.ModuleList([ResidualBlock(8, 1, num_layers) for _ in range(DM)])
        
        self.block4 = ResidualBlock(DM, 16, num_layers)
        self.block5 = ResidualBlock(16, 8, num_layers)
        
        self.fc1 = nn.Linear(in_features=L*8, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=64)
        self.fc4 = nn.Linear(in_features=64, out_features=20)
        self.dropout = nn.Dropout(p=0.2)
        
        kaiming_weight_initialization(self.named_parameters())

    def forward(self, x):
        B, DM, S, L = x.shape
        outputs = []
        for i in range(DM):
            block1 = self.blocks1[i]
            block2 = self.blocks2[i]
            block3 = self.blocks3[i]
            
            sliced = x[:, i, :, :].squeeze(1)       # Shape: (B, S, L) - (B, 700, 16)
            sliced = sliced.permute(0, 2, 1)        # Shape: (B, L, S)
            
            output = block1(sliced)                 # Shape: (B, L, 64)
            output = block2(output)                 # Shape: (B, L, 8)
            output = block3(output)                 # Shape: (B, L, 1)
            
            outputs.append(output)
        
        x = torch.cat(outputs, dim=-1)              # Shape: (B, L, DM)
        x = self.block4(x)                          # Shape: (B, L, 16)
        x = self.block5(x)                          # Shape: (B, L, 8)
        
        x = x.reshape(x.shape[0], -1)               # Shape: (B, L*8)
        x = self.dropout(F.relu(self.fc1(x)))       # Shape: (B, 256)
        x = self.dropout(F.relu(self.fc2(x)))       # Shape: (B, 128)
        x = self.dropout(F.relu(self.fc3(x)))       # Shape: (B, 64)
        x = F.relu(self.fc4(x))                     # Shape: (B, 20)
        
        return x
    
# test network with summary
# net = RNN_avg(700, 16, 16)
# summary(net, (700, 16, 16))

class Conv1D_RNN(nn.Module):
    def __init__(self, DM, S, L):
        super(Conv1D_RNN, self).__init__()
        # self.conv1 = nn.Conv1d(in_channels=700, out_channels=350, kernel_size=3, stride=1, padding=1)
        self.convs1 = nn.ModuleList([nn.Conv1d(in_channels=S, out_channels=256, kernel_size=3, stride=1, padding=1) for _ in range(DM)])
        # self.bn1    = nn.BatchNorm1d(num_features=256)
        self.bn1    = nn.ModuleList([nn.BatchNorm1d(num_features=256) for _ in range(DM)])
        self.pool   = nn.AvgPool1d(kernel_size=8, stride=8, padding=0)
        self.convs2 = nn.ModuleList([nn.Conv1d(in_channels=32, out_channels=8, kernel_size=3, stride=1, padding=1) for _ in range(DM)])
        self.bn2    = nn.ModuleList([nn.BatchNorm1d(num_features=8) for _ in range(DM)])
        self.convs3 = nn.ModuleList([nn.Conv1d(in_channels=S, out_channels=1, kernel_size=3, stride=1, padding=1) for _ in range(DM)])
        self.bn3    = nn.ModuleList([nn.BatchNorm1d(num_features=1) for _ in range(DM)])
        
        self.batch_norm = nn.BatchNorm1d(num_features=DM)
        self.gru = nn.GRU(input_size=DM, hidden_size=8, num_layers=1, batch_first=True)
        
        self.fc1 = nn.Linear(in_features=L*8, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=64)
        self.fc4 = nn.Linear(in_features=64, out_features=20)
        self.dropout = nn.Dropout(p=0.2)
        
        kaiming_weight_initialization(self.named_parameters())

    def _avg_pool(self, output):                # Shape: (B, 256, L)
        output = output.permute(0, 2, 1)        # Shape: (B, L, 256)
        output = self.pool(output)              # Shape: (B, L, 32)
        return output.permute(0, 2, 1)          # Shape: (B, 32, L)
        
    def forward(self, x):
        #                                           x shape: (B, DM, S, L) - (B, 3*7, 700, 16)
        B, DM, S, L = x.shape
        outputs = []
        for i in range(DM):
            conv1 = self.convs1[i]
            conv2 = self.convs2[i]
            conv3 = self.convs3[i]
            bn1   = self.bn1[i]
            bn2   = self.bn2[i]
            bn3   = self.bn3[i]
            # print(x.shape)
            sliced = x[:, i, :, :].squeeze(1)       # Shape: (B, S, L) - (B, 700, 16)
            # print(sliced.shape, x.shape)
            
            # conv1
            output = conv1(sliced)                  # Shape: (B, 256, L) - (B, 256, 16)
            # print(output.shape)
            output = bn1(output)                    # Shape: (B, 256, L) - (B, 256, 16)
            output = F.relu(output)
            output = self._avg_pool(output)         # Shape: (B, 32, L)
            # print(output.shape)
            
            # conv2
            output = conv2(output)                  # Shape: (B, 8, L)
            # print(output.shape)
            output = bn2(output)                    # Shape: (B, 8, L)
            output = F.relu(output)
            output = self._avg_pool(output)         # Shape: (B, 1, L)
            # print('conv2', output.shape)
            
            # Res conv3
            output_r = conv3(sliced)                # Shape: (B, 1, L)
            # print(output.shape)
            output += output_r                      # Shape: (B, 1, L)
            output = bn3(output)                    # Shape: (B, 1, L)
            output = F.relu(output)
            # print('conv3', output.shape)
            
            # append
            outputs.append(output)
            # print('outputs', output.shape)
        
        # Concatenate along the DM dimension
        x = torch.cat(outputs, dim=1)               # Shape: (B, DM, L) - (B, 3*7, 16)
        # print(x.shape)
        x = self.batch_norm(x)                      # Shape: (B, DM, L)
        x = x.permute(0, 2, 1)                      # Shape: (B, L, DM)
        # print(x.shape)
        
        x, _ = self.gru(x)                          # Shape: (B, L, 8)
        # print('gru', x.shape)
        x = x.reshape(x.shape[0], -1)               # Shape: (B, L*8)
        # print('flatten', x.shape)
        x = self.dropout(F.relu(self.fc1(x)))       # Shape: (B, 256)
        x = self.dropout(F.relu(self.fc2(x)))       # Shape: (B, 128)
        x = self.dropout(F.relu(self.fc3(x)))       # Shape: (B, 64)
        x = self.fc4(x)                             # Shape: (B, 20)
        # print(x.shape)
        return x


class LSTM_Embedding(nn.Module):
    def __init__(self, 
                 dms, 
                 l, 
                 hidden_size=64,
                 output_size=20,
            ):
        super(LSTM_Embedding, self).__init__()
        self.rnn1 = nn.LSTM(input_size=dms, hidden_size=hidden_size*8, num_layers=2, batch_first=True)
        # self.norm1 = nn.LayerNorm(hidden_size*8)
        # self.dropout1 = nn.Dropout(p=0.05)
        self.rnn2 = nn.LSTM(input_size=hidden_size*8, hidden_size=hidden_size*4, num_layers=2, batch_first=True)
        # self.norm2 = nn.LayerNorm(hidden_size*4)
        # self.dropout2 = nn.Dropout(p=0.05)
        self.rnn3 = nn.LSTM(input_size=hidden_size*4, hidden_size=hidden_size*2, num_layers=1, batch_first=True)
        # self.norm3 = nn.LayerNorm(hidden_size)
        self.rnn4 = nn.LSTM(input_size=hidden_size*2, hidden_size=hidden_size  , num_layers=1, batch_first=True)
        
        self.fc1 = nn.Linear(l*hidden_size, 4*hidden_size)
        # self.dropout3 = nn.Dropout(p=0.05)
        # self.bn1 = nn.BatchNorm1d(4*hidden_size)
        self.bn1 = nn.LayerNorm(4*hidden_size)
        self.fc2 = nn.Linear(4*hidden_size, output_size)
        
        # weight initialization
        # for name, param in self.named_parameters():
        #     if 'weight_ih' in name:
        #         # nn.init.xavier_uniform_(param.data)  # Xavier initialization
        #         nn.init.kaiming_uniform_(param.data)  # Kaiming initialization
        #     elif 'weight_hh' in name:
        #         nn.init.orthogonal_(param.data)
        #     elif 'bias' in name:
        #         param.data.fill_(0)
        kaiming_weight_initialization(self.named_parameters())
        
    def forward(self, x):
        # x has shape (B, DMS, L)
        x = x.permute(0, 2, 1) # (B, L, DMS) - (batch_size, seq_len, input_size)
        # Pass the first part (x1) through the RNN
        out, _ = self.rnn1(x)  
        # out = self.norm1(out)
        # out    = self.dropout1(out)
        out, _ = self.rnn2(out)
        # out = self.norm2(out)
        # out    = self.dropout2(out)
        out, _ = self.rnn3(out)
        # out = self.norm3(out)
        # out    = self.dropout3(out)
        out, _ = self.rnn4(out)
        # Flatten the RNN output and pass it through the first FC layer
        out = out.reshape(out.shape[0], -1)
        # out = F.relu(self.fc1(out))
        out = F.leaky_relu(self.fc1(out), negative_slope=0.01)
        # out = self.dropout3(out)
        out = self.bn1(out)
        # out = self.dropout4(out)
        # out = F.relu(self.fc2(out))
        out = F.leaky_relu(self.fc2(out), negative_slope=0.01)
        return out


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
        # self.rnn3 = nn.LSTM(input_size=hidden_size*2, hidden_size=hidden_size, num_layers=1, batch_first=True)
        # self.dropout3 = nn.Dropout(p=0.05)
        # self.dropout4 = nn.Dropout(p=0.05)
        self.fc1 = nn.Linear(l*hidden_size*2, 4*hidden_size)
        # self.bn1 = nn.BatchNorm1d(2*hidden_size)
        self.bn1 = nn.LayerNorm(4*hidden_size)
        self.fc2 = nn.Linear(4*hidden_size, output_size)
        
        # weight initialization
        # for name, param in self.named_parameters():
        #     if 'weight_ih' in name:
        #         # nn.init.xavier_uniform_(param.data)  # Xavier initialization
        #         nn.init.kaiming_uniform_(param.data)  # Kaiming initialization
        #     elif 'weight_hh' in name:
        #         nn.init.orthogonal_(param.data)
        #     elif 'bias' in name:
        #         param.data.fill_(0)
        kaiming_weight_initialization(self.named_parameters())
        
    def forward(self, x):
        # x has shape (B, DMS, L)
        x = x.permute(0, 2, 1) # (B, L, DMS) - (batch_size, seq_len, input_size)
        # Pass the first part (x1) through the RNN
        out, _ = self.rnn1(x)  
        # out    = self.dropout1(out)
        out, _ = self.rnn2(out)
        # out    = self.dropout2(ou1t)
        # out, _ = self.rnn3(out)
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
