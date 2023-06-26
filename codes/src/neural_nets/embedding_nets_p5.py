import torch
import torch.nn as nn
import torch.nn.functional as F

# from torchsummary import summary
import sys

sys.path.append("./src")
from utils.train import kaiming_weight_initialization


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
        seqC = x[..., :15]  # [B, DMS, 15]
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
