import torch
import torch.nn as nn
import torch.nn.functional as F

# from torchsummary import summary
import sys

sys.path.append("./src")
from utils.train import kaiming_weight_initialization


class GRU_FC(nn.Module):
    def __init__(self, input_size=3, hidden_size=512, num_layers=1):
        super().__init__()

        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 256)
        self.fc2 = nn.Linear(256, 128)

        kaiming_weight_initialization(self.named_parameters())

    def forward(self, x):
        # x: [B, L, 3] or [B, L, 1]
        x, _ = self.gru(x)  # -> [B, L, 512]
        x = x[:, -1, :].unsqueeze(1)  # -> [B, 1, 512]
        x = x.squeeze(1)  # -> [B, 512]
        x = F.relu(self.fc1(x))  # -> [B, 256]
        x = F.relu(self.fc2(x))  # -> [B, 128]


class Multi_Head_GRU_FC(nn.Module):
    def __init__(self, feature_lengths, input_size=3, hidden_size=512, num_layers=1):
        super().__init__()

        # Initialize feature lengths
        self.feature_lengths = feature_lengths

        # Initialize GRUs for each feature
        self.grus = nn.ModuleList(
            [nn.GRU(input_size, hidden_size, num_layers) for _ in feature_lengths]
        )

        # Initialize FC layers
        self.fc1 = nn.Linear(hidden_size * len(feature_lengths), 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)

        # Initialize activation function
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: [B, Ls, 3]
        # Split the input into features -> [B, Ls, 3] -> [B, L1, 1], [B, L2, 1], ...
        xs = torch.split(x, self.feature_lengths, dim=1)

        # Pass each feature through its corresponding GRU and get the last timestamp output
        # -> [B, 512], [B, 512], ...
        xs = [self.grus[i](xs[i])[0][:, -1, :] for i in range(len(self.feature_lengths))]

        # Concatenate the features
        # -> [B, 512 * len(feature_lengths)]
        x = torch.cat(xs, dim=1)

        # Pass the output through the FC layers with ReLU activation
        x = self.relu(self.fc1(x))  # -> [B, 512]
        x = self.relu(self.fc2(x))  # -> [B, 256]
        x = self.relu(self.fc3(x))  # -> [B, 128]

        return x
