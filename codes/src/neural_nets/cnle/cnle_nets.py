import torch
from torch import nn

from typing import Callable, Optional
import warnings
from typing import Optional, Tuple

from torch import Tensor, nn, unique
from torch.distributions import Categorical
from torch.nn import Sigmoid, Softmax
import torch.nn.functional as F

# from sbi.neural_nets.flow import build_nsf
from sbi.utils.sbiutils import match_theta_and_x_batch_shapes, standardizing_net
from sbi.utils.torchutils import atleast_2d
from sbi.utils.user_input_checks import check_data_device

import sys
from pathlib import Path

NSC_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent.as_posix()
sys.path.append(f"{NSC_DIR}/codes/src")
from utils.dataset.dataset import separate_x
from utils.setup import clean_cache


# Define the LSTM part of the model
class LSTMNet(nn.Module):
    def __init__(
        self,
        input_dim=1,
        hidden_dim=64,
        n_layers=3,
        dropout=0.1,
    ):
        super(LSTMNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch_size, seq_length)
        x.unsqueeze_(2)  # add a channel dimension
        # x: (batch_size, seq_length, input_dim)
        _, (hidden, _) = self.lstm(x)  # hidden: (n_layers, batch_size, hidden_dim)
        # take the last layer's hidden state and apply dropout
        hidden = self.dropout(hidden[-1, :, :])  # hidden: (batch_size, hidden_dim)
        return hidden


class ConvNet(nn.Module):
    def __init__(
        self,
        input_dim=1,
        num_filters=64,
        filter_size=3,
        seqC_length=14,
    ):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels=input_dim,
            out_channels=num_filters,
            kernel_size=filter_size,
            padding=(filter_size - 1) // 2,
        )
        self.conv2 = nn.Conv1d(
            in_channels=num_filters,
            out_channels=num_filters * 2,
            kernel_size=filter_size,
            padding=(filter_size - 1) // 2,
        )
        self.pool = nn.MaxPool1d(filter_size)

        num_features = (
            seqC_length // filter_size // filter_size * 2 * num_filters
        )  # 512

        self.fc1 = nn.Linear(num_features, num_filters * 16)  # 1024
        self.fc2 = nn.Linear(num_filters * 16, num_filters * 4)  # 256
        self.fc3 = nn.Linear(num_filters * 4, num_filters)  # 64
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        # x.shape: (batch_size, seq_length)
        x.unsqueeze_(1)  # add a channel dimension
        # Conv1D expects (batch_size x num_channels x sequence_length)
        # x.shape: (batch_size, num_features, seq_length)
        x = F.relu(self.conv1(x))
        # x.shape: (batch_size, num_filters, seq_length)
        x = self.pool(x)
        # x.shape: (batch_size, num_filters, seq_length//filter_size)
        x = F.relu(self.conv2(x))
        # x.shape: (batch_size, num_filters*2, seq_length//filter_size)
        x = self.pool(x)
        # x.shape: (batch_size, num_filters*2, (seq_length//filter_size)//filter_size)
        x = x.view(x.size(0), -1)  # flatten the output for the classifier
        # x.shape: (batch_size, num_filters*2 * ((seq_length//filter_size)//filter_size))
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        # x.shape: (batch_size, num_features * 16)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        # x.shape: (batch_size, num_features * 4)
        x = F.relu(self.fc3(x))
        # x.shape: (batch_size, num_features)
        x = self.dropout(x)
        return x


class MLP(nn.Module):
    def __init__(
        self,
        num_input=4,
        num_hidden=64,
        num_layers=4,
    ):
        super(MLP, self).__init__()
        self.num_hidden = num_hidden
        self.num_input = num_input
        # self.activation = Sigmoid()
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)

        self.input_layer = nn.Linear(num_input, num_hidden)
        # Repeat hidden units hidden layers times.
        self.hidden_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.hidden_layers.append(nn.Linear(num_hidden, num_hidden))

    def forward(self, x):
        # x shape (batch_size, num_input)
        x = self.activation(self.input_layer(x))
        x = self.dropout(x)
        # x shape (batch_size, num_hidden)
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
            x = self.dropout(x)
        # x shape (batch_size, num_hidden)
        return x


class CategoricalNet(nn.Module):
    """Class to perform conditional density (mass) estimation for a categorical RV.

    Takes input parameters theta, conditioned on seqC,
    and learns the parameters p of a Categorical.

    Defines log prob and sample functions.
    """

    def __init__(
        self,
        # seqC net
        seqC_net_type="cnn",
        input_dim_seqC=1,
        hidden_dim_seqC=64,
        lstm_layers_seqC=3,
        conv_filter_size_seqC=3,
        # theta net
        num_input_theta=4,
        num_hidden_theta=64,
        num_layers_theta=4,
        # cat net
        num_categories=2,
        num_hidden_category=256,
    ):
        """Initialize the neural net.

        Args:
            --- seqC net ---
            input_dim_seqC: dimension of the input seqC.
            hidden_dim_seqC: number of hidden units in the LSTM.
            lstm_layers_seqC: number of layers in the LSTM.

            --- theta net ---
            num_input_theta: dimension of the input theta.
            num_hidden_theta: number of hidden units in the MLP.
            num_layers_theta: number of layers in the MLP.

            --- categorical net ---
            num_categories: number of categories to predict.
            num_hidden_category: number of hidden units in the MLP.
        """
        super(CategoricalNet, self).__init__()

        # self.activation = Sigmoid()
        self.activation = nn.ReLU()
        self.softmax = Softmax(dim=1)
        self.num_input_theta = num_input_theta

        # --- seqC net ---
        if seqC_net_type == "cnn" or seqC_net_type == "conv":
            self.seqC_net = ConvNet(
                input_dim=input_dim_seqC,
                num_filters=hidden_dim_seqC,
                filter_size=conv_filter_size_seqC,
                seqC_length=14,
            )
        elif seqC_net_type == "lstm":
            self.seqC_net = LSTMNet(
                input_dim=input_dim_seqC,
                hidden_dim=hidden_dim_seqC,
                n_layers=lstm_layers_seqC,
                dropout=0.1,
            )
        else:
            raise ValueError(
                f"seqC_net_type must be one of 'cnn', 'conv', or 'lstm', but is {seqC_net_type}."
            )

        # --- theta net ---
        self.theta_net = MLP(
            num_input=num_input_theta,
            num_hidden=num_hidden_theta,
            num_layers=num_layers_theta,
        )

        # --- categorical net ---
        self.input_layer = nn.Linear(
            num_hidden_theta + hidden_dim_seqC, num_hidden_category
        )

        # Repeat hidden units hidden layers times.
        self.hidden_layers = nn.ModuleList()
        self.hidden_layers.append(
            nn.Linear(num_hidden_category, num_hidden_category // 2)
        )  # 256 -> 128

        self.hidden_layers.append(
            nn.Linear(num_hidden_category // 2, num_hidden_category // 4)
        )  # 128 -> 64

        self.hidden_layers.append(
            nn.Linear(num_hidden_category // 4, num_hidden_category // 8)
        )  # 64 -> 32
        self.hidden_layers.append(
            nn.Linear(num_hidden_category // 8, num_hidden_category // 16)
        )  # 32 -> 16
        self.ps_layer = nn.Linear(num_hidden_category // 16, num_hidden_category // 32)
        # 16 -> 8
        self.output_layer = nn.Linear(num_hidden_category // 32, num_categories)
        # 8 -> 2

    def forward(
        self,
        seqC: Tensor,
        theta: Tensor,
    ) -> Tensor:
        """Return categorical probability predicted from a batch of parameters conditioned on the input seqC.

        Args:
            theta: batch of input parameters for the net.
            seqC: batch of input seqC for the net.

        Returns:
            Tensor: batch of predicted categorical probabilities.
        """
        assert seqC.dim() == 2, "input needs to have a batch dimension."
        assert theta.dim() == 2, "input needs to have a batch dimension."
        assert (
            theta.shape[1] == self.num_input_theta
        ), f"input dimensions must match num_input {self.num_input_theta}"

        theta = self.theta_net(theta)  # (batch_size, num_hidden)
        seqC = self.seqC_net(seqC)  # (batch_size, num_hidden)
        x = torch.cat((theta, seqC), dim=1)  # (batch_size, 2*num_hidden)

        # iterate n hidden layers, input x and calculate tanh activation
        x = self.activation(self.input_layer(x))
        for layer in self.hidden_layers:
            x = self.activation(layer(x))

        x = self.activation(self.ps_layer(x))

        return self.softmax(self.output_layer(x))

    def log_prob(
        self,
        seqC: Tensor,
        theta: Tensor,
        chR: Tensor,
    ) -> Tensor:
        """Return categorical log probability of categories x, given parameters theta.

        Args:
            theta: parameters.
            x: categories to evaluate.

        Returns:
            Tensor: log probs with shape (x.shape[0],)
        """
        # Predict categorical ps and evaluate.
        ps = self.forward(seqC=seqC, theta=theta)
        return Categorical(probs=ps).log_prob(chR.squeeze())

    def log_prob_iid(
        self,
        x: Tensor,
        theta: Tensor,
    ):
        """Return categorical log probability of chR, given parameters theta conditioned on seqC.

        x: [seqC, chR]
            seqC:  (num_trials, seq_length)
            chR:   (num_trials, 1)
        theta: (batch_theta, theta_dim)

        input would be repeated before computing the probability.

        Returns:
            ps: (num_trials * batch_theta, num_categories [2])
            log_probs: (num_trials * batch_theta, )

        """
        # x iid trials: XaXbXc -> XaXaXa XbXbXb XcXcXc
        # theta:        TaTbTc -> TaTbTc TaTbTc TaTbTc
        # prob:                   papbpc papbpc papbpc

        theta_repeated, x_repeated = match_theta_and_x_batch_shapes(theta, x)
        del x, theta
        x_repeated, chR_repeated = x_repeated.split(
            [x_repeated.shape[-1] - 1, 1], dim=1
        )
        # x_repeated here is equivalent to seqC_repeated

        ps = self.forward(seqC=x_repeated, theta=theta_repeated)
        del theta_repeated, x_repeated

        log_probs = Categorical(probs=ps).log_prob(chR_repeated.squeeze())
        del ps, chR_repeated
        clean_cache()

        return log_probs

    def sample(
        self,
        num_samples: int,
        seqC: Tensor,
        theta: Tensor,
    ) -> Tensor:
        """Returns samples from categorical random variable with probs predicted from
        the neural net.

        Args:
            theta: batch of parameters for prediction.
            num_samples: number of samples to obtain.

        Returns:
            Tensor: Samples with shape (num_samples, 1)
        """

        # Predict Categorical ps and sample.
        ps = self.forward(seqC, theta)
        return (
            Categorical(probs=ps)
            .sample(torch.Size((num_samples,)))
            .reshape(num_samples, -1)
        )
