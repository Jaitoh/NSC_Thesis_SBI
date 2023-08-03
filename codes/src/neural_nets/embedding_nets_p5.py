import torch
import torch.nn as nn
import torch.nn.functional as F

# from torchsummary import summary
import sys

sys.path.append("./src")
from utils.train import kaiming_weight_initialization


class Conv_Transformer(nn.Module):
    def __init__(self, DMS, nhead=8, num_encoder_layers=6):
        super(Conv_Transformer, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=DMS, out_channels=1024, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=1024, out_channels=512, kernel_size=3, padding=1)

        encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=nhead, dropout=0.1, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 256)

        self.chR_fc_1 = nn.Linear(DMS, 1024)
        self.chR_fc_2 = nn.Linear(1024, 512)
        self.chR_fc_3 = nn.Linear(512, 256)

        self.fc_out1 = nn.Linear(512, 512)
        self.fc_out2 = nn.Linear(512, 256)

        kaiming_weight_initialization(self.named_parameters())

    def forward(self, x):
        # (B, DMS, L + 1)
        seqC = x[..., 1:-1]  # (B, DMS, L)
        chR = x[..., -1]  # (B, DMS)

        seqC = self.conv1(seqC)  # (B, 1024, L)
        seqC = self.conv2(seqC)  # (B, 512, L)

        seqC = seqC.permute(0, 2, 1)  # (B, L, 512)

        seqC = self.transformer_encoder(seqC)  # (B, L, 512)

        seqC = seqC[:, -1, :]  # (B, 512)

        seqC = self.fc1(seqC)  # (B, 512)
        seqC = self.fc2(seqC)  # (B, 256)

        chR = self.chR_fc_1(chR)  # (B, 1024)
        chR = self.chR_fc_2(chR)  # (B, 512)
        chR = self.chR_fc_3(chR)  # (B, 256)

        out = torch.cat((seqC, chR), dim=1)  # (B, 512)

        out = self.fc_out1(out)  # (B, 512)
        out = self.fc_out2(out)  # (B, 256)

        return out


class Conv_LSTM(nn.Module):
    def __init__(self, DMS):
        super(Conv_LSTM, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=DMS, out_channels=1024, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=1024, out_channels=512, kernel_size=3, padding=1)
        self.lstm1 = nn.LSTM(input_size=512, hidden_size=1024, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=1024, hidden_size=512, batch_first=True)

        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 256)

        self.chR_conv1 = nn.Conv1d(in_channels=DMS, out_channels=1024, kernel_size=3, padding=1)
        self.chR_conv2 = nn.Conv1d(in_channels=1024, out_channels=512, kernel_size=3, padding=1)
        self.chR_fc_1 = nn.Linear(512, 512)
        self.chR_fc_2 = nn.Linear(512, 256)

        self.fc_out1 = nn.Linear(512, 512)
        self.fc_out2 = nn.Linear(512, 256)

        kaiming_weight_initialization(self.named_parameters())

    def forward(self, x):
        # (B, DMS, L + 1)
        seqC = x[..., 1:-1]  # (B, DMS, L) !removed the first signal
        chR = x[..., -1].unsqueeze(-1)  # (B, DMS, 1)

        seqC = self.conv1(seqC)  # (B, 1024, L)
        seqC = self.conv2(seqC)  # (B, 512, L)

        seqC = seqC.permute(0, 2, 1)  # (B, L, 512)
        seqC, _ = self.lstm1(seqC)  # (B, L, 1024)
        seqC, _ = self.lstm2(seqC)  # (B, L, 512)

        seqC = seqC[:, -1, :]  # (B, 512)

        seqC = self.fc1(seqC)  # (B, 512)
        seqC = self.fc2(seqC)  # (B, 256)

        chR = self.chR_conv1(chR)  # (B, 1024, 1)
        chR = self.chR_conv2(chR)  # (B, 512, 1)

        chR = chR.squeeze(-1)  # (B, 512)
        chR = self.chR_fc_1(chR)  # (B, 512)
        chR = self.chR_fc_2(chR)  # (B, 256)

        out = torch.cat((seqC, chR), dim=1)  # (B, 512)

        out = self.fc_out1(out)  # (B, 512)
        out = self.fc_out2(out)  # (B, 256)

        return out


class Conv_NET(nn.Module):
    def __init__(self, DMS):
        super(Conv_NET, self).__init__()

        self.pool = nn.MaxPool1d(2)
        self.activate = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.05)
        self.dropout2 = nn.Dropout(p=0.1)

        # seqC network
        self.conv1 = nn.Conv1d(in_channels=DMS, out_channels=1024, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=1024, out_channels=2048, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=2048, out_channels=1024, kernel_size=3, padding=1)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 256)

        # chR network
        self.chR_conv1 = nn.Conv1d(in_channels=DMS, out_channels=1024, kernel_size=3, padding=1)
        self.chR_conv2 = nn.Conv1d(in_channels=1024, out_channels=512, kernel_size=3, padding=1)
        # self.chR_fc_0 = nn.Linear(DMS, 1024)
        # self.chR_fc_1 = nn.Linear(1024, 512)
        self.chR_fc_2 = nn.Linear(512, 512)
        self.chR_fc_3 = nn.Linear(512, 256)
        self.chR_fc_4 = nn.Linear(256, 256)

        # output network
        self.fc_out1 = nn.Linear(512, 512)
        self.fc_out2 = nn.Linear(512, 256)
        self.fc_out3 = nn.Linear(256, 256)

        kaiming_weight_initialization(self.named_parameters())

    def forward(self, x):
        # (B, DMS, L + 1)
        seqC = x[..., 1:-1]  # (B, DMS, 14) !removed the first signal
        chR = x[..., -1].unsqueeze(-1)  # (B, DMS, 1)

        seqC = self.activate(self.conv1(seqC))  # (B, 1024, 14)
        seqC = self.pool(seqC)  # (B, 1024, 7)
        # seqC = self.dropout1(seqC)
        seqC = self.activate(self.conv2(seqC))  # (B, 2048, 7)
        seqC = self.pool(seqC)  # (B, 2048, 3)
        # seqC = self.dropout1(seqC)
        seqC = self.activate(self.conv3(seqC))  # (B, 1024, 3)
        seqC = self.pool(seqC)  # (B, 1024, 1)
        seqC = seqC.squeeze(-1)  # (B, 1024)

        seqC = self.activate(self.fc1(seqC))  # (B, 512)
        seqC = self.dropout1(seqC)
        seqC = self.activate(self.fc2(seqC))  # (B, 256)
        seqC = self.activate(self.fc3(seqC))  # (B, 256)

        chR = self.activate(self.chR_conv1(chR))  # (B, 1024, 1)
        # chR = self.dropout2(chR)
        chR = self.activate(self.chR_conv2(chR))  # (B, 512, 1)
        # chR = self.dropout2(chR)
        chR = chR.squeeze(-1)  # (B, 512)
        chR = self.activate(self.chR_fc_2(chR))  # (B, 512)
        chR = self.dropout1(chR)
        chR = self.activate(self.chR_fc_3(chR))  # (B, 256)
        chR = self.activate(self.chR_fc_4(chR))  # (B, 256)

        out = torch.cat((seqC, chR), dim=1)  # (B, 512)

        out = self.activate(self.fc_out1(out))  # (B, 512)
        out = self.dropout1(out)
        out = self.activate(self.fc_out2(out))  # (B, 256)
        out = self.activate(self.fc_out3(out))  # (B, 256)

        return out


class GRU3_FC(nn.Module):
    def __init__(self, DMS):
        super(GRU3_FC, self).__init__()
        self.seqC_gru_1 = nn.GRU(DMS, 1024, batch_first=True)
        self.seqC_gru_2 = nn.GRU(1024, 512, batch_first=True)
        self.seqC_gru_3 = nn.GRU(512, 256, batch_first=True)

        self.seqC_fc_1 = nn.Linear(256, 512)
        self.seqC_fc_2 = nn.Linear(512, 256)

        self.chR_fc_1 = nn.Linear(DMS, 1024)
        self.chR_fc_2 = nn.Linear(1024, 512)
        self.chR_fc_3 = nn.Linear(512, 256)

        self.final_fc = nn.Linear(512, 256)

        self.relu = nn.ReLU()

        kaiming_weight_initialization(self.named_parameters())

    def forward(self, x):
        # x: [B, DMS, 16]
        seqC = x[..., 1:15]  # [B, DMS, 15]
        chR = x[..., -1]  # [B, DMS]

        # seqC [B, DMS, 15]
        seqC = seqC.permute(0, 2, 1)  # -> [B, 15, DMS]

        seqC, _ = self.seqC_gru_1(seqC)  # [B, 15, 1024]
        seqC, _ = self.seqC_gru_2(seqC)  # [B, 15, 512]
        seqC, _ = self.seqC_gru_3(seqC)  # [B, 15, 256]

        seqC = seqC[:, -1, :]  # [B, 256]
        # seqC = seqC.view(seqC.size(0), -1)  # Flatten to [B, 256]

        seqC = self.relu(self.seqC_fc_1(seqC))  # [B, 512]
        seqC = self.relu(self.seqC_fc_2(seqC))  # [B, 256]

        # chR [B, DMS]
        chR = self.relu(self.chR_fc_1(chR))  # [B, 1024]
        chR = self.relu(self.chR_fc_2(chR))  # [B, 512]
        chR = self.relu(self.chR_fc_3(chR))  # [B, 256]

        # Concatenate along the channel dimension
        out = torch.cat((seqC, chR), dim=1)  # [B, 512]
        out = self.relu(self.final_fc(out))  # [B, 256]

        return out
