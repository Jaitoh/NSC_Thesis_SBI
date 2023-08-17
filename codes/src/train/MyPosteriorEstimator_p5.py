import os
import sys
import h5py
import time
import torch
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from torch.distributions import Distribution, MultivariateNormal, Uniform
from torch import nn, ones, optim
from torch.optim.lr_scheduler import (
    CosineAnnealingWarmRestarts,
    ReduceLROnPlateau,
    ConstantLR,
)
from torch.nn.utils.clip_grad import clip_grad_norm_

from sbi.inference.posteriors.direct_posterior import DirectPosterior
from pyknos.mdn.mdn import MultivariateGaussianMDN as mdn
from sbi.inference import SNPE_C
from sbi.inference.snpe.snpe_base import PosteriorEstimator
from sbi.utils import (
    del_entries,
    check_dist_class,
    x_shape_from_simulation,
    test_posterior_net_for_multi_d_x,
)
from pathlib import Path

NSC_DIR = Path(__file__).resolve().parent.parent.parent.parent.as_posix()  # NSC dir
sys.path.append(f"{NSC_DIR}/codes/src")

from utils.set_seed import setup_seed, seed_worker
from utils.setup import adapt_path

# from train.Dataset_features import Feature_Dataset
from train.Dataset_Classes import chR_2D_Dataset
from utils.train import WarmupScheduler, plot_posterior_with_label, load_net
from utils.setup import clean_cache
from utils.dataset.dataset import update_prior_min_max
from utils.dataset.dataloader import get_dataloaders

# set matplotlib, font of size 16, bold
plt.rcParams.update({"font.size": 22})
plt.rcParams["font.size"] = 22
# plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.linewidth"] = 3


class MyPosteriorEstimator_P5(PosteriorEstimator):
    def __init__(
        self,
        prior=None,
        density_estimator="maf",
        device="gpu",
        logging_level="INFO",
        summary_writer=None,
        show_progress_bars=True,
    ):
        kwargs = del_entries(locals(), entries=("self", "__class__"))
        super().__init__(**kwargs)

    def prepare_dataset_network(
        self,
        config,
        continue_from_checkpoint,
        device="gpu",
        low_batch=0,
    ):
        # prepare train, val dataset and dataloader
        print("\n=== train, val dataset and dataloader ===")
        data_path = adapt_path(config.data_path)
        with h5py.File(data_path, "r") as f:
            sets = list(f.keys())[: config.dataset.num_max_sets]

        train_set_names = sets[: int(len(sets) * 0.9)]
        valid_set_names = sets[int(len(sets) * 0.9) :]
        print(f"{train_set_names=}")
        print(f"{valid_set_names=}")

        DS_config = config.dataset
        num_train_set_T = DS_config.num_max_theta_each_set
        num_valid_set_T = DS_config.num_max_theta_each_set

        # get the original prior min and max for normalization
        _, _, unnormed_prior_min, unnormed_prior_max = update_prior_min_max(
            prior_min=config.prior.prior_min,
            prior_max=config.prior.prior_max,
            ignore_ss=config.prior.ignore_ss,
            normalize=config.prior.normalize,
        )

        print("[training] sets", end=" ")
        train_dataset = chR_2D_Dataset(
            data_path=data_path,
            chosen_set_names=train_set_names,
            num_chosen_theta_each_set=num_train_set_T,
            chosen_dur=DS_config.chosen_dur_list,
            crop_dur=DS_config.crop_dur,
            max_theta_in_a_set=num_train_set_T,
            theta_chosen_mode="random",
            seqC_process=DS_config.seqC_process,
            summary_type=DS_config.summary_type,
            permutation_mode=DS_config.permutation_mode,
            num_probR_sample=DS_config.num_probR_sample,
            ignore_ss=config.prior.ignore_ss,
            normalize_theta=config.prior.normalize,
            unnormed_prior_min=unnormed_prior_min,
            unnormed_prior_max=unnormed_prior_max,
        )

        print("\n[validation] sets", end=" ")
        valid_dataset = chR_2D_Dataset(
            data_path=data_path,
            chosen_set_names=valid_set_names,
            num_chosen_theta_each_set=num_valid_set_T,
            chosen_dur=DS_config.chosen_dur_list,
            crop_dur=DS_config.crop_dur,
            max_theta_in_a_set=num_valid_set_T,
            theta_chosen_mode="random",
            seqC_process=DS_config.seqC_process,
            summary_type=DS_config.summary_type,
            permutation_mode=DS_config.permutation_mode,
            num_probR_sample=DS_config.num_probR_sample,
            ignore_ss=config.prior.ignore_ss,
            normalize_theta=config.prior.normalize,
            unnormed_prior_min=unnormed_prior_min,
            unnormed_prior_max=unnormed_prior_max,
        )

        # prepare train, val, test dataloader
        train_dataloader, valid_dataloader = get_dataloaders(
            config=config,
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            low_batch=low_batch,
        )

        # collect posterior sets
        print(f"\ncollect posterior sets...", end=" ")
        tic = time.time()
        self.seen_data_for_posterior = {"x": [], "theta": []}
        self.unseen_data_for_posterior = {"x": [], "theta": []}

        x_train_batch, theta_train_batch = next(iter(train_dataloader))
        x_valid_batch, theta_valid_batch = next(iter(valid_dataloader))

        for i in range(config.train.posterior.num_posterior_check):
            self.seen_data_for_posterior["x"].append(x_train_batch[i, :])
            self.seen_data_for_posterior["theta"].append(theta_train_batch[i, :])

            self.unseen_data_for_posterior["x"].append(x_valid_batch[i, :])
            self.unseen_data_for_posterior["theta"].append(theta_valid_batch[i, :])
        print(f"takes {time.time() - tic:.2f} seconds = {(time.time() - tic) / 60:.2f} minutes")

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

            test_posterior_net_for_multi_d_x(
                self._neural_net,
                theta_train_batch.to("cpu"),
                x_train_batch.to("cpu"),
            )

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

    def train_base_p5(
        self,
        config,
        prior_limits,
        continue_from_checkpoint=None,
        debug=False,
    ):
        self.config = config
        self.log_dir = config.log_dir
        self.prior_limits = prior_limits
        self.dataset_kwargs = self.config.dataset
        self.training_kwargs = self.config.train.training
        setup_seed(config.seed)

        train_dataloader, valid_dataloader, _ = self.prepare_dataset_network(
            self.config,
            continue_from_checkpoint=continue_from_checkpoint,
            device=self._device,
        )

        # initialize optimizer / sheduler
        config_training = self.config.train.training
        warmup_epochs = config_training.warmup_epochs
        initial_lr = config_training.initial_lr
        # optimizer
        self.optimizer = optim.Adam(
            list(self._neural_net.parameters()),
            lr=config_training.learning_rate,
            weight_decay=eval(config_training.weight_decay)
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
            self.scheduler = ReduceLROnPlateau(self.optimizer, **config_training["scheduler_params"])
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
        print(f"\n{len(train_dataloader)} train batches, {len(valid_dataloader)} valid batches")
        while (
            epoch <= config_training.max_num_epochs
            and not self._converged(epoch, debug)
            # and (not debug or epoch <= 2)
        ):
            # train and log one epoch
            self._neural_net.train()

            epoch_start_time = time.time()
            batch_timer = time.time()

            train_data_size = 0
            train_log_probs_sum = 0
            train_batch_num = 0

            # train one epoch
            for x, theta in train_dataloader:
                self.optimizer.zero_grad()
                # train on batch
                with torch.no_grad():
                    x = x.to(self._device)
                    theta = theta.to(self._device)
                    masks_batch = torch.ones_like(theta[:, 0]).to(self._device)

                train_data_size += len(x)
                train_losses = self._loss(
                    theta,
                    x,
                    masks_batch,
                    proposal=self._proposal_roundwise[-1],
                    calibration_kernel=lambda x: ones([len(x)], device=self._device),
                    force_first_round_loss=True,
                )
                train_loss = torch.mean(train_losses)
                train_loss.backward()
                train_log_probs_sum -= train_losses.sum().item()

                # clip gradients
                clean_cache()
                clip_max_norm = config_training.clip_max_norm
                if clip_max_norm is not None:
                    clip_grad_norm_(self._neural_net.parameters(), max_norm=clip_max_norm)

                del x, theta, masks_batch, train_losses
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

                self._summary_writer.add_scalar("train_loss_batch", train_loss, batch_counter)

                train_batch_num += 1
                batch_counter += 1

                if self.config.debug and train_batch_num >= 3:
                    break

            # epoch log - training log prob
            train_log_prob_average = train_log_probs_sum / train_data_size
            self._train_log_prob = train_log_prob_average
            self._summary["training_log_probs"].append(train_log_prob_average)
            self._summary_writer.add_scalars("log_probs", {"training": train_log_prob_average}, epoch)

            # epoch log - learning rate
            if epoch < config_training.warmup_epochs:
                current_lr = self.scheduler_warmup.optimizer.param_groups[0]["lr"]
            else:
                current_lr = self.scheduler.optimizer.param_groups[0]["lr"]
            self._summary["learning_rates"].append(current_lr)
            self._summary_writer.add_scalar("learning_rates", current_lr, epoch)

            # # log the graident after each epoch
            # for name, param in self._neural_net.named_parameters():
            #     if param.requires_grad:
            #         self._writer_hist.add_histogram(
            #             f"Gradients/{name}", param.grad, epoch
            #         )

            # # log the bias, activations, layer, weights after each epoch
            # for name, param in self._neural_net.named_parameters():
            #     if param.requires_grad:
            #         self._writer_hist.add_histogram(
            #             f"Weights/{name}", param, epoch
            #         )

            # validate one epoch
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
                    masks_batch = torch.ones_like(theta_valid[:, 0])

                    # update validation loss
                    valid_losses = self._loss(
                        theta_valid,
                        x_valid,
                        masks_batch,
                        proposal=self._proposal_roundwise[-1],  # TODO: check proposal
                        calibration_kernel=lambda x: ones([len(x)], device=self._device),
                        force_first_round_loss=True,
                    )

                    valid_log_prob_sum -= valid_losses.sum().item()
                    valid_data_size += len(x_valid)

                    del x_valid, theta_valid, masks_batch, valid_losses
                    clean_cache()

                    if self.config.debug:
                        break

                # epoch log - validation log prob
                self._valid_log_prob = valid_log_prob_sum / valid_data_size
                self._summary_writer.add_scalars("log_probs", {"validation": self._valid_log_prob}, epoch)

                toc = time.time()
                self._summary["validation_log_probs"].append(self._valid_log_prob)
                self._summary["epoch_durations_sec"].append(toc - train_start_time)

                valid_info = f"\nvalid_log_prob: {self._valid_log_prob:.2f} in {(time.time() - valid_start_time)/60:.2f} min"
                print(valid_info)

            # update epoch info and counter
            epoch_info = f"| Epochs trained: {epoch:4} | log_prob train: {self._train_log_prob:.2f} | log_prob val: {self._valid_log_prob:.2f} | . Time elapsed {(time.time()-epoch_start_time)/ 60:6.2f}min, trained in total {(time.time() - train_start_time)/60:6.2f}min"
            print(epoch_info)

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
        # save best model
        # torch.save(
        #     deepcopy(self._neural_net.state_dict()),
        #     os.path.join(self.config.log_dir, f"model/best_model.pt"),
        # )

        self._plot_training_curve()

        return self, deepcopy(self._neural_net)

    def _converged(self, epoch, debug):
        converged = False
        epoch = epoch - 1
        assert self._neural_net is not None

        if epoch == -1 or (self._valid_log_prob > self._best_valid_log_prob):
            self._epochs_since_last_improvement = 0
            self._best_valid_log_prob = self._valid_log_prob
            self._best_model_state_dict = deepcopy(self._neural_net.state_dict())
            self._best_model_from_epoch = epoch

            posterior_step = self.config.train.posterior.step
            # plot posterior behavior when best model is updated
            if epoch != -1 and posterior_step != 0 and epoch % posterior_step == 0:
                self._posterior_behavior_log(self.prior_limits, epoch)

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

        if epoch != -1:
            self._plot_training_curve()

        return converged

    def _posterior_behavior_log(self, limits, epoch):
        config = self.config
        if config.prior.ignore_ss:
            prior_labels = config.prior.prior_labels[:1] + config.prior.prior_labels[3:]
        else:
            prior_labels = config.prior.prior_labels

        with torch.no_grad():
            current_net = deepcopy(self._neural_net)

            tic = time.time()
            print("--> Building posterior...", end=" ")

            posterior = self.build_posterior(current_net)
            self._model_bank = []  # !clear model bank to avoid memory leak

            print(f"in {(time.time()-tic)/60:.2f} min, Plotting ... ", end=" ")
            num_data = len(self.seen_data_for_posterior["x"])
            for fig_idx in range(num_data):
                print(f"{fig_idx}/{num_data-1}", end=" ")
                # plot posterior - seen data
                fig_x, _, _ = plot_posterior_with_label(
                    posterior=posterior,
                    sample_num=config.train.posterior.sampling_num,
                    x=self.seen_data_for_posterior["x"][fig_idx].to(self._device),
                    true_params=self.seen_data_for_posterior["theta"][fig_idx],
                    limits=limits,
                    prior_labels=prior_labels,
                    show_progress_bars=False,
                )
                fig_path = f"{self.log_dir}/posterior/figures/posterior_seen_{fig_idx}_epoch_{epoch}.png"
                plt.savefig(fig_path)
                fig_path = f"{self.log_dir}/posterior/posterior_seen_{fig_idx}_up_to_date.png"
                plt.savefig(fig_path)
                plt.close(fig_x)
                del fig_x, _
                clean_cache()

                # plot posterior - unseen data
                fig_x_val, _, _ = plot_posterior_with_label(
                    posterior=posterior,
                    sample_num=config.train.posterior.sampling_num,
                    x=self.unseen_data_for_posterior["x"][fig_idx].to(self._device),
                    true_params=self.unseen_data_for_posterior["theta"][fig_idx],
                    limits=limits,
                    prior_labels=prior_labels,
                    show_progress_bars=False,
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
        ax1.plot(train_log_probs, ".-", label="training", alpha=0.8, lw=2, color="tab:blue", ms=0.1)
        ax1.plot(valid_log_probs, ".-", label="validation", alpha=0.8, lw=2, color="tab:orange", ms=0.1)
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

        ax3.plot(train_log_probs, "o-", label="training", alpha=0.8, lw=2, color="tab:blue", ms=0.1)
        ax3.plot(valid_log_probs, "o-", label="validation", alpha=0.8, lw=2, color="tab:orange", ms=0.1)

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


class MySNPE_C_P5(SNPE_C, MyPosteriorEstimator_P5):
    def __init__(
        self,
        prior=None,
        density_estimator="maf",
        device="gpu",
        logging_level="INFO",
        summary_writer=None,
        show_progress_bars=True,
    ):
        kwargs = del_entries(locals(), entries=("self", "__class__"))
        super().__init__(**kwargs)

    def train(
        self,
        config,
        prior_limits,
        continue_from_checkpoint=None,
        debug=False,
    ):
        self.config = config
        self._num_atoms = self.config.train.training.num_atoms
        self._use_combined_loss = self.config.train.training.use_combined_loss
        kwargs = del_entries(
            locals(),
            entries=(
                "self",
                "__class__",
                "num_atoms",
                "use_combined_loss",
                "__pydevd_ret_val_dict",
            ),
        )

        self._round = max(self._data_round_index)

        if self._round > 0:
            # Set the proposal to the last proposal that was passed by the user. For
            # atomic SNPE, it does not matter what the proposal is. For non-atomic
            # SNPE, we only use the latest data that was passed, i.e. the one from the
            # last proposal.
            proposal = self._proposal_roundwise[-1]
            self.use_non_atomic_loss = (
                isinstance(proposal, DirectPosterior)
                and isinstance(proposal.posterior_estimator._distribution, mdn)
                and self._neural_net is not None
                and isinstance(self._neural_net._distribution, mdn)
                and check_dist_class(self._prior, class_to_check=(Uniform, MultivariateNormal))[0]
            )

            algorithm = "non-atomic" if self.use_non_atomic_loss else "atomic"
            print(f"Using SNPE-C with {algorithm} loss")

            if self.use_non_atomic_loss:
                # Take care of z-scoring, pre-compute and store prior terms.
                self._set_state_for_mog_proposal()

        return super().train_base_p5(**kwargs)  # type: ignore
