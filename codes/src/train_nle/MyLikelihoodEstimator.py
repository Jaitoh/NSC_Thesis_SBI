import time
import os
from abc import ABC
from copy import deepcopy
from typing import Any, Callable, Dict, Optional, Union, Tuple
import h5py

import torch
from torch.optim.lr_scheduler import (
    CosineAnnealingWarmRestarts,
    ReduceLROnPlateau,
    ConstantLR,
)

import numpy as np
import matplotlib.pyplot as plt
from pyknos.nflows import flows
from torch import Tensor, nn, optim
from torch.distributions import Distribution
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils import data
from torch.utils.tensorboard.writer import SummaryWriter

from sbi import utils as utils
from sbi.inference import NeuralInference
from sbi.inference.posteriors import RejectionPosterior, VIPosterior
from sbi.types import TorchTransform
from sbi.utils import (
    check_estimator_arg,
    check_prior,
    handle_invalid_x,
    mask_sims_from_prior,
    nle_nre_apt_msg_on_invalid_x,
    validate_theta_and_x,
    warn_if_zscoring_changes_data,
    x_shape_from_simulation,
)

from sbi.inference.posteriors import MCMCPosterior, RejectionPosterior, VIPosterior
from sbi.inference.potentials.likelihood_based_potential import (
    likelihood_estimator_based_potential,
    LikelihoodBasedPotential,
    MixedLikelihoodBasedPotential,
)
from sbi.utils import mcmc_transform
from sbi.inference.snle.snle_base import LikelihoodEstimator
from sbi.neural_nets.mnle import MixedDensityEstimator
from sbi.types import TensorboardSummaryWriter, TorchModule
from sbi.utils import check_prior, del_entries

import sys
from pathlib import Path

NSC_DIR = Path(__file__).resolve().parent.parent.parent.parent.as_posix()  # NSC dir
sys.path.append(f"{NSC_DIR}/codes/src")

from utils.train import WarmupScheduler, plot_posterior_with_label, load_net
from utils.set_seed import setup_seed, seed_worker
from utils.setup import clean_cache
from utils.dataset.dataset import update_prior_min_max
from train_nle.Dataset import chR_Comb_Dataset, probR_Comb_Dataset
from posterior.My_MCMCPost import MyMCMCPosterior


class MyLikelihoodEstimator(NeuralInference, ABC):
    def __init__(
        self,
        prior: Optional[Distribution] = None,
        density_estimator: Union[str, Callable] = "maf",
        device: str = "cpu",
        logging_level: Union[int, str] = "WARNING",
        summary_writer: Optional[SummaryWriter] = None,
        show_progress_bars: bool = True,
    ):
        r"""Base class for Sequential Neural Likelihood Estimation methods.

        Args:
            prior: A probability distribution that expresses prior knowledge about the
                parameters, e.g. which ranges are meaningful for them. Any
                object with `.log_prob()`and `.sample()` (for example, a PyTorch
                distribution) can be used.
            density_estimator: If it is a string, use a pre-configured network of the
                provided type (one of nsf, maf, mdn, made). Alternatively, a function
                that builds a custom neural network can be provided. The function will
                be called with the first batch of simulations (theta, x), which can
                thus be used for shape inference and potentially for z-scoring. It
                needs to return a PyTorch `nn.Module` implementing the density
                estimator. The density estimator needs to provide the methods
                `.log_prob` and `.sample()`.

        See docstring of `NeuralInference` class for all other arguments.
        """

        super().__init__(
            prior=prior,
            device=device,
            logging_level=logging_level,
            summary_writer=summary_writer,
            show_progress_bars=show_progress_bars,
        )

        # As detailed in the docstring, `density_estimator` is either a string or
        # a callable. The function creating the neural network is attached to
        # `_build_neural_net`. It will be called in the first round and receive
        # thetas and xs as inputs, so that they can be used for shape inference and
        # potentially for z-scoring.
        check_estimator_arg(density_estimator)
        if isinstance(density_estimator, str):
            self._build_neural_net = utils.likelihood_nn(model=density_estimator)
        else:
            self._build_neural_net = density_estimator

    def append_simulations(
        self,
        theta: Tensor,
        x: Tensor,
        exclude_invalid_x: bool = False,
        from_round: int = 0,
        data_device: Optional[str] = None,
    ) -> "LikelihoodEstimator":
        r"""Store parameters and simulation outputs to use them for later training.

        Data are stored as entries in lists for each type of variable (parameter/data).

        Stores $\theta$, $x$, prior_masks (indicating if simulations are coming from the
        prior or not) and an index indicating which round the batch of simulations came
        from.

        Args:
            theta: Parameter sets.
            x: Simulation outputs.
            exclude_invalid_x: Whether invalid simulations are discarded during
                training. If `False`, SNLE raises an error when invalid simulations are
                found. If `True`, invalid simulations are discarded and training
                can proceed, but this gives systematically wrong results.
            from_round: Which round the data stemmed from. Round 0 means from the prior.
                With default settings, this is not used at all for `SNLE`. Only when
                the user later on requests `.train(discard_prior_samples=True)`, we
                use these indices to find which training data stemmed from the prior.
            data_device: Where to store the data, default is on the same device where
                the training is happening. If training a large dataset on a GPU with not
                much VRAM can set to 'cpu' to store data on system memory instead.
        Returns:
            NeuralInference object (returned so that this function is chainable).
        """

        is_valid_x, num_nans, num_infs = handle_invalid_x(x, exclude_invalid_x)

        x = x[is_valid_x]
        theta = theta[is_valid_x]

        # Check for problematic z-scoring
        warn_if_zscoring_changes_data(x)
        nle_nre_apt_msg_on_invalid_x(num_nans, num_infs, exclude_invalid_x, "SNLE")

        if data_device is None:
            data_device = self._device
        theta, x = validate_theta_and_x(
            theta, x, data_device=data_device, training_device=self._device
        )

        prior_masks = mask_sims_from_prior(int(from_round), theta.size(0))

        self._theta_roundwise.append(theta)
        self._x_roundwise.append(x)
        self._prior_masks.append(prior_masks)

        self._data_round_index.append(int(from_round))

        return self

    def prepare_dataset_network(
        self, config, continue_from_checkpoint=None, device="cuda"
    ):
        # prepare train, val dataset and dataloader
        print("".center(50, "="))
        print("prepare train, val dataset and dataloader")

        # get the original prior min and max for normalization
        # _, _, unnormed_prior_min, unnormed_prior_max = update_prior_min_max(
        #     prior_min=self.config.prior.prior_min,
        #     prior_max=self.config.prior.prior_max,
        #     ignore_ss=self.config.prior.ignore_ss,
        #     normalize=self.config.prior.normalize,
        # )

        # --- train / valid set ---
        data_dir = config.data_path
        DS_config = config.dataset

        print("".center(50, "="))
        print("[training] sets")
        # print("".center(50, "-"))
        train_dataset = chR_Comb_Dataset(
            data_dir=data_dir,
            num_max_theta=config.dataset.num_max_theta,
            num_chosen_theta=config.dataset.num_chosen_theta,
            chosen_dur_list=config.dataset.chosen_dur_list,
            part_each_dur=[0.9] * len(config.dataset.chosen_dur_list),
            last_part=False,
            theta_chosen_mode="random",
            num_probR_sample=config.dataset.num_probR_sample,
            probR_sample_mode=config.dataset.probR_sample_mode,
            print_info=True,
            config_theta=config.prior,
        )

        print("".center(50, "="))
        print("[validation] sets")
        # print("".center(50, "-"))
        valid_dataset = chR_Comb_Dataset(
            data_dir=data_dir,
            num_max_theta=config.dataset.num_max_theta,
            num_chosen_theta=config.dataset.num_chosen_theta,
            chosen_dur_list=config.dataset.chosen_dur_list,
            part_each_dur=[0.1] * len(config.dataset.chosen_dur_list),
            last_part=True,
            theta_chosen_mode="random",
            num_probR_sample=config.dataset.num_probR_sample,
            probR_sample_mode=config.dataset.probR_sample_mode,
            print_info=True,
            config_theta=config.prior,
        )

        # prepare train, val, test dataloader
        config_dataset = config.dataset
        prefetch_factor = (
            config_dataset.prefetch_factor if config_dataset.num_workers > 0 else None
        )
        loader_kwargs = {
            "batch_size": min(
                config_dataset.batch_size,
                len(train_dataset),
                len(valid_dataset),
            ),
            "drop_last": False,
            "shuffle": True,
            "pin_memory": config_dataset.pin_memory,
            "num_workers": config_dataset.num_workers,
            "prefetch_factor": prefetch_factor,
            "worker_init_fn": seed_worker,
        }
        print("".center(50, "-"))
        print(f"{loader_kwargs=}")

        g = torch.Generator()
        g.manual_seed(config.seed)

        train_dataloader = data.DataLoader(train_dataset, generator=g, **loader_kwargs)
        loader_kwargs["drop_last"] = True
        valid_dataloader = data.DataLoader(valid_dataset, generator=g, **loader_kwargs)

        x_train_batch, theta_train_batch = next(iter(train_dataloader))
        # x_valid_batch, theta_valid_batch = next(iter(valid_dataloader))
        print("".center(50, "-"))
        print("")

        # initialize the network
        if self._neural_net is None:
            # Use only training data for building the neural net (z-scoring transforms)
            self._neural_net = self._build_neural_net(
                theta_train_batch[:3].to("cpu"),
                x_train_batch[:3].to("cpu"),
            )
            self._x_shape = x_shape_from_simulation(x_train_batch.to("cpu"))

            print("\nfinished build network")
            print(self._neural_net)

            # load network from state dict if specified
            if continue_from_checkpoint != None and continue_from_checkpoint != "":
                self._neural_net = load_net(
                    continue_from_checkpoint,
                    self._neural_net,
                    device=device,
                )

        return (
            train_dataloader,
            valid_dataloader,
            self._neural_net.to(self._device),
            train_dataset,
            valid_dataset,
        )

    def train(
        self,
        # training_batch_size: int = 50,
        # learning_rate: float = 5e-4,
        # validation_fraction: float = 0.1,
        # stop_after_epochs: int = 20,
        # max_num_epochs: int = 2**31 - 1,
        # clip_max_norm: Optional[float] = 5.0,
        # resume_training: bool = False,
        # discard_prior_samples: bool = False,
        # retrain_from_scratch: bool = False,
        # show_train_summary: bool = False,
        # dataloader_kwargs: Optional[Dict] = None,
        config,
        prior_limits,
        continue_from_checkpoint=None,
        debug=False,
    ) -> flows.Flow:
        r"""Train the density estimator to learn the distribution $p(x|\theta)$.

        Args:
            resume_training: Can be used in case training time is limited, e.g. on a
                cluster. If `True`, the split between train and validation set, the
                optimizer, the number of epochs, and the best validation log-prob will
                be restored from the last time `.train()` was called.
            discard_prior_samples: Whether to discard samples simulated in round 1, i.e.
                from the prior. Training may be sped up by ignoring such less targeted
                samples.
            retrain_from_scratch: Whether to retrain the conditional density
                estimator for the posterior from scratch each round.
            show_train_summary: Whether to print the number of epochs and validation
                loss after the training.
            dataloader_kwargs: Additional or updated kwargs to be passed to the training
                and validation dataloaders (like, e.g., a collate_fn)

        Returns:
            Density estimator that has learned the distribution $p(x|\theta)$.
        """
        # Load data from most recent round.
        self._round = max(self._data_round_index)
        self.config = config
        self.log_dir = config.log_dir
        self.prior_limits = prior_limits
        self.dataset_kwargs = self.config.dataset
        self.training_kwargs = self.config.train.training
        setup_seed(config.seed)

        # ========== 1. Prepare dataset and network ==========
        train_dataloader, valid_dataloader, _, _, _ = self.prepare_dataset_network(
            self.config,
            continue_from_checkpoint=continue_from_checkpoint,
            device=self._device,
        )

        # ========== 2. initialize before training ==========
        config_training = self.config.train.training
        warmup_epochs = config_training.warmup_epochs
        initial_lr = config_training.initial_lr
        # optimizer
        self.optimizer = optim.Adam(
            list(self._neural_net.parameters()),
            lr=config_training.learning_rate,
            weight_decay=eval(config_training.weight_decay)  # ! weight decay ignored?
            if isinstance(config_training.weight_decay, str)
            else config_training.weight_decay,
        )
        # scheduler
        self.scheduler_warmup = WarmupScheduler(
            self.optimizer,
            warmup_epochs=warmup_epochs,
            init_lr=initial_lr,
            target_lr=config_training.learning_rate,
        )

        if config_training["scheduler"] == "ReduceLROnPlateau":
            self.scheduler = ReduceLROnPlateau(
                self.optimizer, **config_training["scheduler_params"]
            )
        if config_training["scheduler"] == "CosineAnnealingWarmRestarts":
            self.scheduler = CosineAnnealingWarmRestarts(
                self.optimizer, **config_training["scheduler_params"]
            )
        if config_training["scheduler"] == "None":  # constant lr
            self.scheduler = ConstantLR(self.optimizer, factor=1.0)

        # initialize values
        epoch = 0
        batch_counter = 0
        self._valid_log_prob = float("-Inf")
        self._best_valid_log_prob = float("-Inf")
        # self._best_model_from_epoch = -1
        self._summary["training_log_probs"] = []
        self._summary["learning_rates"] = []
        self._summary["validation_log_probs"] = []
        self._summary["epoch_durations_sec"] = []

        # train until no validation improvement for 'patience' epochs
        train_start_time = time.time()
        print(
            f"\n{len(train_dataloader)} train batches, {len(valid_dataloader)} valid batches"
        )

        # ========== 2. Train network ==========
        # train until no validation improvement for 'patience' epochs
        train_start_time = time.time()
        while (
            epoch <= config_training.max_num_epochs
            and not self._converged(epoch, debug)
            # and (not debug or epoch <= 2)
        ):
            # Train for a single epoch.
            self._neural_net.train()

            epoch_start_time = time.time()
            batch_timer = time.time()

            train_data_size = 0
            train_log_probs_sum = 0
            train_batch_num = 0

            for x, theta in train_dataloader:
                self.optimizer.zero_grad()
                # theta_batch, x_batch = (
                #     batch[0].to(self._device),
                #     batch[1].to(self._device),
                # )
                # train on batch
                with torch.no_grad():
                    x = x.to(self._device)
                    theta = theta.to(self._device)

                # Evaluate on x with theta as context. TODO: the sign of loss +?-?
                train_data_size += len(x)
                train_losses = self._loss(theta=theta, x=x)
                train_loss = torch.mean(train_losses)
                train_log_probs_sum -= train_losses.sum().item()

                train_loss.backward()

                # clip gradients
                clean_cache()
                clip_max_norm = config_training.clip_max_norm
                if clip_max_norm is not None:
                    clip_grad_norm_(
                        self._neural_net.parameters(),
                        max_norm=clip_max_norm,
                    )

                del x, theta, train_losses
                clean_cache()

                self.optimizer.step()

                # log one batch
                print_freq = config_training.print_freq
                batch_info = f"epoch {epoch:4}: batch {train_batch_num:4}  train_loss {-1*train_loss:.2f}, time {(time.time() - batch_timer)/60:.2f}min"
                if print_freq == 0:  # do nothing
                    pass
                elif len(train_dataloader) <= print_freq:
                    print(batch_info)
                elif train_batch_num % (len(train_dataloader) // print_freq) == 0:
                    print(batch_info)

                self._summary_writer.add_scalar(
                    "train_loss_batch", train_loss, batch_counter
                )

                train_batch_num += 1
                batch_counter += 1

                if self.config.debug and train_batch_num >= 3:
                    break

            # self.epoch += 1
            train_log_prob_average = train_log_probs_sum / train_data_size
            self._train_log_prob = train_log_prob_average
            self._summary["training_log_probs"].append(train_log_prob_average)
            self._summary_writer.add_scalars(
                "log_probs", {"training": train_log_prob_average}, epoch
            )

            # epoch log - learning rate
            if epoch < config_training.warmup_epochs:
                current_lr = self.scheduler_warmup.optimizer.param_groups[0]["lr"]
            else:
                current_lr = self.scheduler.optimizer.param_groups[0]["lr"]
            self._summary["learning_rates"].append(current_lr)
            self._summary_writer.add_scalar("learning_rates", current_lr, epoch)

            # Calculate validation performance.
            self._neural_net.eval()
            with torch.no_grad():
                valid_start_time = time.time()

                # initialize values
                valid_data_size = 0
                valid_log_prob_sum = 0

                # do validate and log
                for x_valid, theta_valid in valid_dataloader:
                    x_valid = x_valid.to(self._device)
                    theta_valid = theta_valid.to(self._device)

                    # Evaluate on x with theta as context.
                    valid_losses = self._loss(theta=theta_valid, x=x_valid)
                    valid_log_prob_sum -= valid_losses.sum().item()
                    valid_data_size += len(x_valid)

                    del x_valid, theta_valid, valid_losses
                    clean_cache()

                    if self.config.debug:
                        break

                # Take mean over all validation samples.
                self._valid_log_prob = valid_log_prob_sum / valid_data_size
                self._summary_writer.add_scalars(
                    "log_probs", {"validation": self._valid_log_prob}, epoch
                )

                toc = time.time()
                self._summary["validation_log_probs"].append(self._valid_log_prob)
                self._summary["epoch_durations_sec"].append(toc - train_start_time)

                valid_info = f"valid_log_prob: {self._valid_log_prob:.2f} in {(time.time() - valid_start_time)/60:.2f} min"
                print(valid_info)

            # update epoch info and counter
            epoch_info = f"Epochs trained: {epoch:4} | log_prob train: {self._train_log_prob:.2f} | log_prob val: {self._valid_log_prob:.2f} | . Time elapsed {(time.time()-epoch_start_time)/ 60:6.2f}min, trained in total {(time.time() - train_start_time)/60:6.2f}min"
            print("".center(50, "-"))
            print(epoch_info)
            print("".center(50, "-"))
            print("")

            # update scheduler
            if epoch < config_training["warmup_epochs"]:
                self.scheduler_warmup.step()
            elif config_training["scheduler"] == "ReduceLROnPlateau":
                self.scheduler.step(self._valid_log_prob)
            else:
                self.scheduler.step()

            # if debug and epoch > 3:
            #     break
            epoch += 1

        # Update summary.
        self._summary["epochs_trained"].append(epoch)
        self._summary["best_validation_log_prob"].append(self._best_valid_log_prob)

        # Update TensorBoard and summary dict.
        self._summarize(round_=self._round)

        del train_dataloader  # , train_dataset
        clean_cache()

        info = f"""
        -------------------------
        ||||| STATS |||||:
        -------------------------
        Total epochs trained: {epoch-1}
        Best validation performance: {self._best_valid_log_prob:.4f}, from epoch {self._best_model_from_epoch:5}
        Model from best epoch {self._best_model_from_epoch} is loaded for further training
        -------------------------
        """
        print(info)

        # finish training
        train_dataloader = None
        valid_dataloader = None
        train_dataset = None
        valid_dataset = None
        x = None
        theta = None
        del train_dataloader, x, theta, valid_dataset, train_dataset
        clean_cache()
        # avoid keeping gradients in resulting network
        self._neural_net.zero_grad(set_to_none=True)

        self._plot_training_curve()

        return self, deepcopy(self._neural_net)

    def _loss(self, theta: Tensor, x: Tensor) -> Tensor:
        r"""Return loss for SNLE, which is the likelihood of $-\log q(x_i | \theta_i)$.

        Returns:
            Negative log prob.
        """
        return -self._neural_net.log_prob(x=x, theta=theta)

    def _converged(self, epoch, debug):
        converged = False
        epoch = epoch - 1
        assert self._neural_net is not None

        if epoch != -1:
            self._plot_training_curve()

        if epoch == -1 or (self._valid_log_prob > self._best_valid_log_prob):
            self._epochs_since_last_improvement = 0
            self._best_valid_log_prob = self._valid_log_prob
            self._best_model_state_dict = deepcopy(self._neural_net.state_dict())
            self._best_model_from_epoch = epoch

            # posterior_step = self.config.train.posterior.step
            # plot posterior behavior when best model is updated
            # if epoch != -1 and posterior_step != 0 and epoch % posterior_step == 0:
            #     self.posterior_behavior_log(self.prior_limits, epoch)

            # save the model
            torch.save(
                self._neural_net,
                os.path.join(self.config.log_dir, f"model/model_check_point.pt"),
            )
        else:
            self._epochs_since_last_improvement += 1

        # If no validation improvement over many epochs, stop training.
        stop_after_epochs = self.config.train.training.stop_after_epochs
        min_num_epochs = self.config.train.training.min_num_epochs
        if (
            self._epochs_since_last_improvement > stop_after_epochs - 1
            and epoch > min_num_epochs
            or (debug and epoch > 3)
        ):
            converged = True
            self._neural_net.load_state_dict(self._best_model_state_dict)
            self._val_log_prob = self._best_valid_log_prob
            self._epochs_since_last_improvement = 0

        return converged

    def _plot_training_curve(self):
        log_dir = self.config.log_dir
        duration = np.array(self._summary["epoch_durations_sec"])
        train_log_probs = self._summary["training_log_probs"]
        valid_log_probs = self._summary["validation_log_probs"]
        learning_rates = self._summary["learning_rates"]
        best_valid_log_prob = self._best_valid_log_prob
        best_valid_log_prob_epoch = self._best_model_from_epoch

        plt.tight_layout()

        fig, axes = plt.subplots(3, 1, figsize=(25, 18))
        fig.subplots_adjust(hspace=0.6)

        # plot learning rate
        ax0 = axes[0]
        ax0.plot(learning_rates, "-", label="lr", lw=2)
        ax0.plot(best_valid_log_prob_epoch, learning_rates[best_valid_log_prob_epoch], "v", color="tab:red", lw=2)  # type: ignore

        ax0.set_xlabel("epochs")
        ax0.set_ylabel("learning rate")
        ax0.grid(alpha=0.2)
        ax0.set_title("training curve")

        ax1 = axes[1]
        ax1.plot(
            train_log_probs,
            ".-",
            label="training",
            alpha=0.8,
            lw=2,
            color="tab:blue",
            ms=0.1,
        )
        ax1.plot(
            valid_log_probs,
            ".-",
            label="validation",
            alpha=0.8,
            lw=2,
            color="tab:orange",
            ms=0.1,
        )
        ax1.plot(best_valid_log_prob_epoch, best_valid_log_prob, "v", color="red", lw=2)
        ax1.text(best_valid_log_prob_epoch, best_valid_log_prob, f"{best_valid_log_prob:.2f}", color="red", fontsize=10, ha="center", va="bottom")  # type: ignore
        # ax1.set_ylim(log_probs_lower_bound, max(valid_log_probs)+0.2)

        ax1.legend(bbox_to_anchor=(1, 1), loc="upper left", borderaxespad=0.0)
        ax1.set_xlabel("epochs")
        ax1.set_ylabel("log_prob")
        ax1.grid(alpha=0.2)

        ax2 = ax1.twiny()
        ax2.plot(
            (duration - duration[0]) / 60 / 60,
            max(valid_log_probs) * np.ones_like(valid_log_probs),
            "-",
            alpha=0,
        )
        ax2.set_xlabel("time (hours)")

        ax3 = axes[2]

        ax3.plot(
            train_log_probs,
            "o-",
            label="training",
            alpha=0.8,
            lw=2,
            color="tab:blue",
            ms=0.1,
        )
        ax3.plot(
            valid_log_probs,
            "o-",
            label="validation",
            alpha=0.8,
            lw=2,
            color="tab:orange",
            ms=0.1,
        )

        all_probs = np.concatenate([train_log_probs, valid_log_probs])
        upper = np.max(all_probs)
        lower = np.percentile(all_probs, 10)
        ax3.legend(bbox_to_anchor=(1, 1), loc="upper left", borderaxespad=0.0)
        ax3.set_xlabel("epochs")
        ax3.set_ylabel("log_prob")
        ax3.set_ylim(lower, upper)
        ax3.grid(alpha=0.2)

        # save the figure
        plt.savefig(f"{log_dir}/training_curve.png")
        plt.close()

    def build_posterior(
        self,
        density_estimator: Optional[TorchModule] = None,
        prior: Optional[Distribution] = None,
        sample_with: str = "mcmc",
        mcmc_method: str = "slice_np",
        vi_method: str = "rKL",
        mcmc_parameters: Dict[str, Any] = {},
        vi_parameters: Dict[str, Any] = {},
        rejection_sampling_parameters: Dict[str, Any] = {},
    ) -> Union[MCMCPosterior, RejectionPosterior, VIPosterior]:
        r"""Build posterior from the neural density estimator.

        CNLE trains a neural network to approximate the likelihood $p(chR|\theta, seqC)$. The
        posterior wraps the trained network such that one can directly evaluate the
        unnormalized posterior log probability
        $p(\theta|seqC, chR) \propto p(chR|\theta, seqC) \cdot p(\theta)$
        and draw samples from the posterior with MCMC or rejection sampling.

        Args:
            density_estimator: The density estimator that the posterior is based on.
                If `None`, use the latest neural density estimator that was trained.
            prior: Prior distribution.
            sample_with: Method to use for sampling from the posterior. Must be one of
                [`mcmc` | `rejection` | `vi`].
            mcmc_method: Method used for MCMC sampling, one of `slice_np`, `slice`,
                `hmc`, `nuts`. Currently defaults to `slice_np` for a custom numpy
                implementation of slice sampling; select `hmc`, `nuts` or `slice` for
                Pyro-based sampling.
            vi_method: Method used for VI, one of [`rKL`, `fKL`, `IW`, `alpha`]. Note
                some of the methods admit a `mode seeking` property (e.g. rKL) whereas
                some admit a `mass covering` one (e.g fKL).
            mcmc_parameters: Additional kwargs passed to `MCMCPosterior`.
            vi_parameters: Additional kwargs passed to `VIPosterior`.
            rejection_sampling_parameters: Additional kwargs passed to
                `RejectionPosterior`.

        Returns:
            Posterior $p(\theta|x)$  with `.sample()` and `.log_prob()` methods
            (the returned log-probability is unnormalized).
        """
        if prior is None:
            assert (
                self._prior is not None
            ), """You did not pass a prior. You have to pass the prior either at
            initialization `inference = SNLE(prior)` or to `.build_posterior
            (prior=prior)`."""
            prior = self._prior
        else:
            check_prior(prior)

        if density_estimator is None:
            likelihood_estimator = self._neural_net
            # If internal net is used device is defined.
            device = self._device
        else:
            likelihood_estimator = density_estimator
            # Otherwise, infer it from the device of the net parameters.
            device = next(density_estimator.parameters()).device.type

        (
            potential_fn,
            theta_transform,
        ) = conditioned_likelihood_estimator_based_potential(
            likelihood_estimator=likelihood_estimator, prior=prior, x_o=None
        )

        if sample_with == "mcmc":
            self._posterior = MyMCMCPosterior(
                potential_fn=potential_fn,
                theta_transform=theta_transform,
                proposal=prior,
                method=mcmc_method,
                device=device,
                x_shape=self._x_shape,
                **mcmc_parameters,
            )
        elif sample_with == "rejection":
            self._posterior = RejectionPosterior(
                potential_fn=potential_fn,
                proposal=prior,
                device=device,
                x_shape=self._x_shape,
                **rejection_sampling_parameters,
            )
        elif sample_with == "vi":
            self._posterior = VIPosterior(
                potential_fn=potential_fn,
                theta_transform=theta_transform,
                prior=prior,  # type: ignore
                vi_method=vi_method,
                device=device,
                x_shape=self._x_shape,
                **vi_parameters,
            )
        else:
            raise NotImplementedError

        return deepcopy(self._posterior)

    def posterior_behavior_log(self, limits, epoch):
        config = self.config
        if config.prior.ignore_ss:
            prior_labels = config.prior.prior_labels[:1] + config.prior.prior_labels[3:]
        else:
            prior_labels = config.prior.prior_labels

        with torch.no_grad():
            tic = time.time()
            print("--> Building posterior...", end=" ")

            mcmc_parameters = dict(
                warmup_steps=100,
                thin=2,
                num_chains=10,
                num_workers=10,
                init_strategy="sir",
            )

            posterior = self.build_posterior(
                density_estimator=self._neural_net,
                prior=self._prior,
                sample_with="mcmc",
                mcmc_method="slice",
                mcmc_parameters=mcmc_parameters,
            )

            self._model_bank = []  # !clear model bank to avoid memory leak

            print(f"in {(time.time()-tic)/60:.2f} min, Plotting ... ", end=" ")
            num_data = len(self.seen_data_for_posterior["x"])

            for fig_idx in range(num_data):
                print(f"{fig_idx}/{num_data-1}", end=" ")
                # plot posterior - seen data
                fig_x, _ = plot_posterior_with_label(
                    posterior=posterior,
                    sample_num=config.train.posterior.sampling_num,
                    x=self.seen_data_for_posterior["x"][fig_idx].to(self._device),
                    true_params=self.seen_data_for_posterior["theta"][fig_idx],
                    limits=limits,
                    prior_labels=prior_labels,
                )
                fig_path = f"{self.log_dir}/posterior/figures/posterior_seen_{fig_idx}_epoch_{epoch}.png"
                plt.savefig(fig_path)
                fig_path = (
                    f"{self.log_dir}/posterior/posterior_seen_{fig_idx}_up_to_date.png"
                )
                plt.savefig(fig_path)
                plt.close(fig_x)
                del fig_x, _
                clean_cache()

                # plot posterior - unseen data
                fig_x_val, _ = plot_posterior_with_label(
                    posterior=posterior,
                    sample_num=config.train.posterior.sampling_num,
                    x=self.unseen_data_for_posterior["x"][fig_idx].to(self._device),
                    true_params=self.unseen_data_for_posterior["theta"][fig_idx],
                    limits=limits,
                    prior_labels=prior_labels,
                )
                fig_path = f"{self.log_dir}/posterior/figures/posterior_unseen_{fig_idx}_epoch_{epoch}.png"
                plt.savefig(fig_path)
                fig_path = f"{self.log_dir}/posterior/posterior_unseen_{fig_idx}_up_to_date.png"
                plt.savefig(fig_path)
                plt.close(fig_x_val)
                del fig_x_val, _
                clean_cache()

            del posterior, current_net
            clean_cache()
            print(f"finished in {(time.time()-tic)/60:.2f}min")


class CNLE(MyLikelihoodEstimator):
    def __init__(
        self,
        prior: Optional[Distribution] = None,
        density_estimator: Union[str, Callable] = "mnle",
        device: str = "cpu",
        logging_level: Union[int, str] = "WARNING",
        summary_writer: Optional[TensorboardSummaryWriter] = None,
        show_progress_bars: bool = True,
    ):
        r"""COnditioned Neural Likelihood Estimation (MNLE) [1].

        Like SNLE, but not sequential and designed to be applied to data with conditioned
        types, e.g., continuous data and discrete data like they occur in
        decision-making experiments

        [1] Flexible and efficient simulation-based inference for models of
        decision-making, Boelts et al. 2021,
        https://www.biorxiv.org/content/10.1101/2021.12.22.473472v2

        Args:
            prior: A probability distribution that expresses prior knowledge about the
                parameters, e.g. which ranges are meaningful for them. If `None`, the
                prior must be passed to `.build_posterior()`.
            density_estimator: If it is a string, it must be "mnle" to use the
                preconfiugred neural nets for MNLE. Alternatively, a function
                that builds a custom neural network can be provided. The function will
                be called with the first batch of simulations (theta, x), which can
                thus be used for shape inference and potentially for z-scoring. It
                needs to return a PyTorch `nn.Module` implementing the density
                estimator. The density estimator needs to provide the methods
                `.log_prob`, `.log_prob_iid()` and `.sample()`.
            device: Training device, e.g., "cpu", "cuda" or "cuda:{0, 1, ...}".
            logging_level: Minimum severity of messages to log. One of the strings
                INFO, WARNING, DEBUG, ERROR and CRITICAL.
            summary_writer: A tensorboard `SummaryWriter` to control, among others, log
                file location (default is `<current working directory>/logs`.)
            show_progress_bars: Whether to show a progressbar during simulation and
                sampling.
        """

        if isinstance(density_estimator, str):
            assert (
                density_estimator == "mnle"
            ), f"""MNLE can be used with preconfigured 'mnle' density estimator only,
                not with {density_estimator}."""
        kwargs = del_entries(locals(), entries=("self", "__class__"))
        super().__init__(**kwargs)


# ========== posterior related potential functions ==========
def conditioned_likelihood_estimator_based_potential(
    likelihood_estimator,
    prior,
    x_o,
) -> Tuple[Callable, TorchTransform]:
    device = str(next(likelihood_estimator.conditioned_net.parameters()).device)

    potential_fn = ConditionedLikelihoodBasedPotential(
        likelihood_estimator, prior, x_o, device=device
    )
    theta_transform = mcmc_transform(prior, device=device)

    return potential_fn, theta_transform


class ConditionedLikelihoodBasedPotential(LikelihoodBasedPotential):
    def __init__(
        self,
        likelihood_estimator,
        prior,
        x_o,
        device="cpu",
    ):
        super().__init__(likelihood_estimator, prior, x_o, device)

    def __call__(self, theta: Tensor, track_gradients: bool = True) -> Tensor:
        # Calculate likelihood in one batch.
        with torch.set_grad_enabled(track_gradients):
            log_likelihood_trial_batch = self.likelihood_estimator.log_prob_iid(
                x=self.x_o,
                theta=theta.to(self.device),
            )
            # Reshape to (x-trials x parameters), sum over trial-log likelihoods.
            log_likelihood_trial_sum = (
                log_likelihood_trial_batch.reshape(self.x_o.shape[0], -1)
                .sum(0)
                .to(self.device)
            )

        return log_likelihood_trial_sum + self.prior.log_prob(theta)
