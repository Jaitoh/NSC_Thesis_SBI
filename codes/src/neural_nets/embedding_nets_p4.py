import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# from torchsummary import summary
import sys

sys.path.append("./src")
from utils.train import kaiming_weight_initialization


class CNN_FC2(nn.Module):
    def __init__(self, input_feature_length):
        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels=1, out_channels=64, kernel_size=5, stride=1, padding=2
        )
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv1d(
            in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2
        )
        self.pool2 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv1d(
            in_channels=128, out_channels=256, kernel_size=5, stride=1, padding=2
        )
        self.pool3 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv1d(
            in_channels=256, out_channels=512, kernel_size=5, stride=1, padding=2
        )
        self.pool4 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.conv5 = nn.Conv1d(
            in_channels=512, out_channels=1024, kernel_size=5, stride=1, padding=2
        )
        self.pool5 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.conv6 = nn.Conv1d(
            in_channels=1024, out_channels=512, kernel_size=5, stride=1, padding=2
        )
        self.pool6 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.conv7 = nn.Conv1d(
            in_channels=512, out_channels=256, kernel_size=5, stride=1, padding=2
        )
        self.pool7 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.fc1 = nn.Linear(256 * math.ceil(input_feature_length / 2**7), 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 256)

        kaiming_weight_initialization(self.named_parameters())

    def forward(self, x):
        # x: [B, M*221, 1]
        x = x.permute(0, 2, 1)  # [B, 1, M*221]
        x = F.relu(self.conv1(x))  # [B, 64, M*221=663]
        x = self.pool1(x)  # [B, 64, 332]
        x = F.relu(self.conv2(x))  # [B, 128, 332]
        x = self.pool2(x)  # [B, 128, 166]
        x = F.relu(self.conv3(x))  # [B, 256, 166]
        x = self.pool3(x)  # [B, 256, 83]
        x = F.relu(self.conv4(x))  # [B, 512, 83]
        x = self.pool4(x)  # [B, 512, 42]
        x = F.relu(self.conv5(x))  # [B, 1024, 42]
        x = self.pool5(x)  # [B, 1024, 21]
        x = F.relu(self.conv6(x))  # [B, 512, 21]
        x = self.pool6(x)  # [B, 512, 11]
        x = F.relu(self.conv7(x))  # [B, 256, 11]
        x = self.pool7(x)  # [B, 256, 6]

        x = x.view(x.size(0), -1)  # [B, 256*6=1536]
        x = F.relu(self.fc1(x))  # [B, 1024]
        x = F.relu(self.fc2(x))  # [B, 512]
        x = F.relu(self.fc3(x))  # [B, 256]
        x = F.relu(self.fc4(x))  # [B, 256]

        return x


class Multi_Channel_CNN_FC(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1
        )
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv1d(
            in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1
        )
        self.pool2 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv1d(
            in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1
        )
        self.pool3 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.fc1 = nn.Linear(7168, 1024)
        self.fc2 = nn.Linear(1024, 256)

        kaiming_weight_initialization(self.named_parameters())

    def forward(self, x):
        # x: [B, 221, 3]
        x = x.permute(0, 2, 1)  # [B, 3, 221]
        x = F.relu(self.conv1(x))  # [B, 64, 221]
        x = self.pool1(x)  # [B, 64, 111]
        x = F.relu(self.conv2(x))  # [B, 128, 111]
        x = self.pool2(x)  # [B, 128, 55]
        x = F.relu(self.conv3(x))  # [B, 256, 56]
        x = self.pool3(x)  # [B, 256, 28]

        x = x.view(x.size(0), -1)  # [B, 256*56]
        x = F.relu(self.fc1(x))  # [B, 1024]
        x = F.relu(self.fc2(x))  # [B, 256]

        return x


class MLP_FC(nn.Module):
    def __init__(self, input_feature_length):
        super().__init__()

        self.fc1 = nn.Linear(input_feature_length, 1024)
        self.fc2 = nn.Linear(1024, 2048)
        self.fc3 = nn.Linear(2048, 1024)
        self.fc4 = nn.Linear(1024, 512)
        self.fc5 = nn.Linear(512, 512)
        self.fc6 = nn.Linear(512, 256)
        self.fc7 = nn.Linear(256, 256)

        kaiming_weight_initialization(self.named_parameters())

    def forward(self, x):
        # x: [B, 3*221, 1]
        x = x.view(x.size(0), -1)  # [B, 3*221]
        x = F.relu(self.fc1(x))  # [B, 1024]
        x = F.relu(self.fc2(x))  # [B, 2048]
        x = F.relu(self.fc3(x))  # [B, 1024]
        x = F.relu(self.fc4(x))  # [B, 512]
        x = F.relu(self.fc5(x))  # [B, 512]
        x = F.relu(self.fc6(x))  # [B, 256]
        x = F.relu(self.fc7(x))  # [B, 256]

        return x


class CNN_FC(nn.Module):
    def __init__(self, input_feature_length):
        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1
        )
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv1d(
            in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1
        )
        self.pool2 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv1d(
            in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1
        )
        self.pool3 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.fc1 = nn.Linear(256 * math.ceil(input_feature_length / 8), 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)

        kaiming_weight_initialization(self.named_parameters())

    def forward(self, x):
        # x: [B, M*221, 1]
        x = x.permute(0, 2, 1)  # [B, 1, M*221]
        x = F.relu(self.conv1(x))  # [B, 64, M*221=663]
        x = self.pool1(x)  # [B, 64, 332]
        x = F.relu(self.conv2(x))  # [B, 128, 332]
        x = self.pool2(x)  # [B, 128, M*56]
        x = F.relu(self.conv3(x))  # [B, 256, 166]
        x = self.pool3(x)  # [B, 256, 83]

        x = x.view(x.size(0), -1)  # [B, 256*M*28]
        x = F.relu(self.fc1(x))  # [B, 1024]
        x = F.relu(self.fc2(x))  # [B, 512]
        x = F.relu(self.fc3(x))  # [B, 256]

        return x


class Multi_Channel_CNN_FC(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1
        )
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv1d(
            in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1
        )
        self.pool2 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv1d(
            in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1
        )
        self.pool3 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.fc1 = nn.Linear(7168, 1024)
        self.fc2 = nn.Linear(1024, 256)

        kaiming_weight_initialization(self.named_parameters())

    def forward(self, x):
        # x: [B, 221, 3]
        x = x.permute(0, 2, 1)  # [B, 3, 221]
        x = F.relu(self.conv1(x))  # [B, 64, 221]
        x = self.pool1(x)  # [B, 64, 111]
        x = F.relu(self.conv2(x))  # [B, 128, 111]
        x = self.pool2(x)  # [B, 128, 55]
        x = F.relu(self.conv3(x))  # [B, 256, 56]
        x = self.pool3(x)  # [B, 256, 28]

        x = x.view(x.size(0), -1)  # [B, 256*56]
        x = F.relu(self.fc1(x))  # [B, 1024]
        x = F.relu(self.fc2(x))  # [B, 256]

        return x


class MLP_FC(nn.Module):
    def __init__(self, input_feature_length):
        super().__init__()

        self.fc1 = nn.Linear(input_feature_length, 1024)
        self.fc2 = nn.Linear(1024, 2048)
        self.fc3 = nn.Linear(2048, 1024)
        self.fc4 = nn.Linear(1024, 512)
        self.fc5 = nn.Linear(512, 512)
        self.fc6 = nn.Linear(512, 256)
        self.fc7 = nn.Linear(256, 256)

        kaiming_weight_initialization(self.named_parameters())

    def forward(self, x):
        # x: [B, 3*221, 1]
        x = x.view(x.size(0), -1)  # [B, 3*221]
        x = F.relu(self.fc1(x))  # [B, 1024]
        x = F.relu(self.fc2(x))  # [B, 2048]
        x = F.relu(self.fc3(x))  # [B, 1024]
        x = F.relu(self.fc4(x))  # [B, 512]
        x = F.relu(self.fc5(x))  # [B, 512]
        x = F.relu(self.fc6(x))  # [B, 256]
        x = F.relu(self.fc7(x))  # [B, 256]

        return x


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


class LSTM3_FC(nn.Module):
    def __init__(self, input_size=3, num_layers=1):
        super().__init__()

        self.rnn1 = nn.LSTM(input_size, 64, num_layers, batch_first=True)
        self.rnn2 = nn.LSTM(64, 256, num_layers, batch_first=True)
        self.rnn3 = nn.LSTM(256, 1024, num_layers, batch_first=True)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)

        kaiming_weight_initialization(self.named_parameters())

    def forward(self, x):
        # x: [B, L, 3] or [B, L, 1]
        x, _ = self.rnn1(x)  # -> [B, L, 64]
        x, _ = self.rnn2(x)  # -> [B, L, 256]
        x, _ = self.rnn3(x)  # -> [B, L, 1024]
        x = x[:, -1, :].unsqueeze(1)  # -> [B, 1, 1024]
        x = x.squeeze(1)  # -> [B, 1024]
        x = F.relu(self.fc1(x))  # -> [B, 512]
        x = F.relu(self.fc2(x))  # -> [B, 256]

        return x


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

        return x


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
        xs = [
            self.grus[i](xs[i])[0][:, -1, :] for i in range(len(self.feature_lengths))
        ]

        # Concatenate the features
        # -> [B, 512 * len(feature_lengths)]
        x = torch.cat(xs, dim=1)

        # Pass the output through the FC layers with ReLU activation
        x = self.relu(self.fc1(x))  # -> [B, 512]
        x = self.relu(self.fc2(x))  # -> [B, 256]
        x = self.relu(self.fc3(x))  # -> [B, 128]

        return x
