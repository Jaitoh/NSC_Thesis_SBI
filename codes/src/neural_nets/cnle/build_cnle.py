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

from neural_nets.cnle.cnle_nets import CategoricalNet


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
