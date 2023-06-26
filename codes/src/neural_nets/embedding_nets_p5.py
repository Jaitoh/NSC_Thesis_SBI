import torch
import torch.nn as nn
import torch.nn.functional as F

# from torchsummary import summary
import sys

sys.path.append("./src")
from utils.train import kaiming_weight_initialization


class GRU3_FC(nn.Module):
    def __init__(self, input_size=3, num_layers=1):
        super().__init__()

        self.gru1 = nn.GRU(input_size, 64, num_layers, batch_first=True)
        self.gru2 = nn.GRU(64, 256, num_layers, batch_first=True)
        self.gru3 = nn.GRU(256, 1024, num_layers, batch_first=True)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)

        kaiming_weight_initialization(self.named_parameters())

    def forward(self, x):
        # x: [B, L, 3] or [B, L, 1]
        x, _ = self.gru1(x)  # -> [B, L, 64]
        x, _ = self.gru2(x)  # -> [B, L, 256]
        x, _ = self.gru3(x)  # -> [B, L, 1024]
        x = x[:, -1, :].unsqueeze(1)  # -> [B, 1, 1024]
        x = x.squeeze(1)  # -> [B, 1024]
        x = F.relu(self.fc1(x))  # -> [B, 512]
        x = F.relu(self.fc2(x))  # -> [B, 256]

        return x
