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


def build_cnle(
    batch_x: Tensor,
    batch_y: Tensor,
    z_score_x: Optional[str] = None,
    z_score_y: Optional[str] = None,
    hidden_features: int = 50,
    hidden_layers: int = 2,
    **kwargs,
):
    """Returns a 'conditional' density estimator.

    Uses a categorical net to model the discrete choice
    (without a neural spline flow (NSF) to model the continuous part)
    adding a network to analysis the condition/context.

    Args:
        batch_x: batch of data (seqC, chR)
        batch_y: batch of parameters
        z_score_x: whether to z-score x.
        z_score_y: whether to z-score y.
        hidden_features: number of hidden features used in the nets.
        hidden_layers: number of hidden layers in the categorical net.
        log_transform_x: whether to apply a log-transform to x to move it to unbounded
            space, e.g., in case x consists of reaction time data (bounded by zero).

    Returns:
        MixedDensityEstimator: nn.Module for performing MNLE.
    """
    config_net = kwargs["config"].train.network
    check_data_device(batch_x, batch_y)
    # if z_score_y == "independent":
    #     embedding = standardizing_net(batch_y)
    # else:
    #     embedding = None

    # Separate continuous and discrete data.
    # seqC, chR = _separate_x(batch_x)

    # Infer input and output dims.
    dim_parameters = batch_y[0].numel()
    # num_categories = unique(chR).numel()

    # Set up a categorical RV neural net for modelling the discrete data.
    disc_nle = CategoricalNet(
        # seqC net
        seqC_net_type=config_net.seqC_net.type,
        input_dim_seqC=1,
        hidden_dim_seqC=config_net.seqC_net.hidden_dim,
        lstm_layers_seqC=config_net.seqC_net.lstm_layers,
        conv_filter_size_seqC=config_net.seqC_net.conv_filter_size,
        # theta net
        num_input_theta=dim_parameters,
        num_hidden_theta=config_net.theta_net.hidden_dim,
        num_layers_theta=config_net.theta_net.num_layers - 1,
        # cat net
        num_categories=2,
        num_hidden_category=config_net.cat_net.hidden_dim,
    )

    return ConditionedDensityEstimator(conditioned_net=disc_nle)


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


class ConditionedDensityEstimator(nn.Module):
    """Class performing Conditioned Neural Likelihood Estimation.

    CNLE model the likelihood for choices conditioned on input sequences,
    e.g., as they occur in decision-making models.
    """

    def __init__(
        self,
        conditioned_net: CategoricalNet,
    ):
        """Initialize class for combining density estimators for MNLE.

        Args:
            discrete_net: neural net to model discrete part of the data.
        """
        super(ConditionedDensityEstimator, self).__init__()

        self.conditioned_net = conditioned_net

    def forward(
        self,
        seqC: Tensor,
        theta: Tensor,
    ):
        raise NotImplementedError(
            """The forward method is not implemented for CNLE, use '.sample(...)' to
            generate samples though a forward pass."""
        )

    def sample(
        self,
        num_samples: int = 1,
        seqC: Tensor = None,
        theta: Tensor = None,
        track_gradients: bool = False,
    ) -> Tensor:
        """Return sample from mixed data distribution.

        Args:
            theta: parameters for which to generate samples.
            seqC: seqC under which condition to generate samples.
            num_samples: number of samples to generate.

        Returns:
            Tensor: samples with shape (num_samples, num_data_dimensions)
        """
        assert theta.shape[0] == 1, "Samples can be generated for a single theta only."
        assert seqC.shape[0] == 1, "Samples can be generated for a single seqC only."

        with torch.set_grad_enabled(track_gradients):
            # Sample discrete data given parameters.
            conditioned_chR = self.conditioned_net.sample(
                num_samples=num_samples,
                seqC=seqC,
                theta=theta,
            ).reshape(num_samples, 1)

        return conditioned_chR

    def log_prob(
        self,
        x: Tensor,
        theta: Tensor,
    ) -> Tensor:
        """Return log-probability of samples under the learned CNLE.

        For a fixed data point x this returns the value of the likelihood function
        evaluated at theta, L(theta | seqC, chR).

        Alternatively, it can be interpreted as the log-prob of the density
        p(chR | theta, seqC).

        Args:
            x: data (containing seqC and chR data).
            context: parameters for which to evaluate the likelihod function, or for
                which to condition p(chR | theta, seqC).

        Returns:
            Tensor: log_prob of p(chR | theta, seqC).
        """
        assert x.shape[0] == theta.shape[0], "x and theta must have same batch size."

        seqC, chR = _separate_x(x)
        num_parameters = theta.shape[0]

        conditioned_log_prob = self.conditioned_net.log_prob(
            seqC=seqC,
            theta=theta,
            chR=chR,
        ).reshape(num_parameters)

        return conditioned_log_prob

    def log_prob_iid(self, x: Tensor, theta: Tensor) -> Tensor:
        """Return log prob given a batch of iid x and a different batch of theta.

        This is different from `.log_prob()` to enable speed ups in evaluation during
        inference. The speed up is achieved by exploiting the fact that there are only
        finite number of possible categories in the discrete part of the dat: one can
        just calculate the log probs for each possible category (given the current batch
        of theta) and then copy those log probs into the entire batch of iid categories.
        For example, for the drift-diffusion model, there are only two choices, but
        often 100s or 1000 trials. With this method a evaluation over trials then passes
        a batch of `2 (one per choice) * num_thetas` into the NN, whereas the normal
        `.log_prob()` would pass `1000 * num_thetas`.

        Args:
            x: batch of iid data, data observed given the same underlying parameters or
                experimental conditions.
            theta: batch of parameters to be evaluated, i.e., each batch entry will be
                evaluated for the entire batch of iid x.

        Returns:
            Tensor: log probs with shape (num_trials, num_parameters), i.e., the log
                prob for each theta for each trial.
        """
        theta = atleast_2d(theta)
        x = atleast_2d(x)
        batch_size = theta.shape[0]
        num_trials = x.shape[0]

        # x iid trials: XaXbXc -> XaXaXa XbXbXb XcXcXc
        # theta:        TaTbTc -> TaTbTc TaTbTc TaTbTc
        theta_repeated, x_repeated = match_theta_and_x_batch_shapes(theta, x)

        net_device = next(self.conditioned_net.parameters()).device
        assert (
            net_device == x.device and x.device == theta.device
        ), f"device mismatch: net, x, theta: {net_device}, {x.device}, {theta.device}."

        seqC_repeated, chR_repeated = _separate_x(x_repeated)

        # compute the log probs for each oberseved data [seqC, chR] given each theta
        log_probs_conditioned = self.conditioned_net.log_prob(
            seqC=seqC_repeated,
            theta=theta_repeated,
            chR=chR_repeated,
        ).reshape(num_trials, batch_size)

        # Return batch over trials as required by SBI potentials.
        return log_probs_conditioned


def _separate_x(x: Tensor) -> Tuple[Tensor, Tensor]:
    """Returns the seqC and chR part of the given x.

    Assumes the chR live in the last columns of x.
    returns [seqC, chR]
    """

    assert x.ndim == 2, f"x must have two dimensions but has {x.ndim}."

    return x[:, :-1], x[:, -1]
