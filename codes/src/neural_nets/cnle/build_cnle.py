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

from neural_nets.cnle.cnle_nets import *
from utils.dataset.dataset import separate_x
from utils.setup import clean_cache, report_memory, torch_var_size


def build_cnle(
    batch_x: Tensor,
    batch_y: Tensor,
    iid_batch_size_x=2,
    iid_batch_size_theta=-1,
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
    # seqC, chR = separate_x(batch_x)

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

    return ConditionedDensityEstimator(
        conditioned_net=disc_nle,
        iid_batch_size_x=iid_batch_size_x,
        iid_batch_size_theta=iid_batch_size_theta,
    )


class ConditionedDensityEstimator(nn.Module):
    """Class performing Conditioned Neural Likelihood Estimation.

    CNLE model the likelihood for choices conditioned on input sequences,
    e.g., as they occur in decision-making models.
    """

    def __init__(
        self,
        conditioned_net,
        iid_batch_size_x=2,
        iid_batch_size_theta=-1,
    ):
        """Initialize class for combining density estimators for MNLE.

        Args:
            discrete_net: neural net to model discrete part of the data.
        """
        super(ConditionedDensityEstimator, self).__init__()

        self.conditioned_net = conditioned_net
        self.iid_batch_size_x = iid_batch_size_x
        self.iid_batch_size_theta = iid_batch_size_theta

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

        seqC, chR = separate_x(x)
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

        net_device = next(self.conditioned_net.parameters()).device
        assert (
            net_device == x.device and x.device == theta.device
        ), f"device mismatch: net, x, theta: {net_device}, {x.device}, {theta.device}."

        num_trials = x.shape[0]
        batch_size_theta = theta.shape[0]

        # seperate x, theta into chunks of size self.iid_batch_size_x/theta
        if self.iid_batch_size_x == -1 or x.shape[0] <= self.iid_batch_size_x:
            x_chunks = [x]
        else:
            x_chunks = torch.split(x, self.iid_batch_size_x)

        if (
            self.iid_batch_size_theta == -1
            or theta.shape[0] <= self.iid_batch_size_theta
        ):
            theta_chunks = [theta]
        else:
            theta_chunks = torch.split(theta, self.iid_batch_size_theta)

        log_probs_conditioned = torch.empty(num_trials, batch_size_theta)

        counter_x = 0
        self.conditioned_net.eval()
        for i in range(len(x_chunks)):
            x_ = x_chunks[i].to(net_device)

            counter_theta = 0
            for j in range(len(theta_chunks)):
                theta_ = theta_chunks[j].to(net_device)
                with torch.no_grad():
                    # compute the log probs for each oberseved data [seqC, chR] given each theta
                    log_probs_ = self.conditioned_net.log_prob_iid(
                        x=x_,
                        theta=theta_,
                    ).reshape(x_.shape[0], theta_.shape[0])

                # log the log_probs
                log_probs_conditioned[
                    counter_x : counter_x + x_.shape[0],
                    counter_theta : counter_theta + theta_.shape[0],
                ] = log_probs_

                counter_theta += theta_.shape[0]

            counter_x += x_.shape[0]

            del log_probs_, x_
            clean_cache()

        # Return batch over trials as required by SBI potentials.
        return log_probs_conditioned


# def separate_x(x: Tensor) -> Tuple[Tensor, Tensor]:
#     """Returns the seqC and chR part of the given x.

#     Assumes the chR live in the last columns of x.
#     returns [seqC, chR]
#     """

#     assert x.ndim == 2, f"x must have two dimensions but has {x.ndim}."

#     return x[:, :-1], x[:, -1]
