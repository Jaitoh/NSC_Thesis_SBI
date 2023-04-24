import os
import time
from copy import deepcopy

import torch
from torch import nn
from torch import Tensor, nn, ones, optim
from torch.distributions import Distribution, MultivariateNormal, Uniform
from torch.utils import data
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard.writer import SummaryWriter
from torch.nn.utils.clip_grad import clip_grad_norm_
from pyknos.mdn.mdn import MultivariateGaussianMDN as mdn
from typing import Any, Callable, Dict, Optional, Tuple, Union
import matplotlib.pyplot as plt
from tqdm import tqdm

from sbi.inference import SNPE_C
from sbi.inference.posteriors.direct_posterior import DirectPosterior
from sbi.inference.snpe.snpe_base import PosteriorEstimator
from sbi.utils import (
    check_dist_class,
    del_entries,
    test_posterior_net_for_multi_d_x,
    x_shape_from_simulation,
    del_entries,
    validate_theta_and_x,
    handle_invalid_x,
    warn_if_zscoring_changes_data,
    nle_nre_apt_msg_on_invalid_x,
    npe_msg_on_invalid_x,
)   
from sbi.utils.sbiutils import mask_sims_from_prior

import sys
sys.path.append('./src')
from train.MyData import MyDataset
from utils.train import (
    plot_posterior_seen,
)

class MyPosteriorEstimator(PosteriorEstimator):
    def __init__(
        self,
        prior: Optional[Distribution] = None,
        density_estimator: Union[str, Callable] = "maf",
        device: str = "gpu",
        logging_level: Union[int, str] = "INFO",
        summary_writer: Optional[SummaryWriter] = None,
        show_progress_bars: bool = True,    
    ):
        kwargs = del_entries(locals(), entries=("self", "__class__"))
        super().__init__(**kwargs)
    
    @staticmethod
    def _maybe_show_progress(show, epoch, starting_time, train_log_prob, val_log_prob):
        if show:
            print(f"\n | Epochs trained: {epoch:5}. Time elapsed {(time.time()-starting_time)/ 60:6.2f}min ||  log_prob train: {train_log_prob:.2f} val: {val_log_prob:.2f} \n")
    
    def _converged(self, epoch: int, stop_after_epochs: int) -> bool:
        """Return whether the training converged yet and save best model state so far.

        Checks for improvement in validation performance over previous epochs.

        Args:
            epoch: Current epoch in training.
            stop_after_epochs: How many fruitless epochs to let pass before stopping.

        Returns:
            Whether the training has stopped improving, i.e. has converged.
        """
        converged = False

        assert self._neural_net is not None
        neural_net = self._neural_net

        # (Re)-start the epoch count with the first epoch or any improvement.
        if epoch == 0 or self._val_log_prob > self._best_val_log_prob:
            self._best_val_log_prob = self._val_log_prob
            self._epochs_since_last_improvement = 0
            self._best_model_state_dict = deepcopy(neural_net.state_dict())
            self._best_model_from_epoch = epoch
            torch.save(deepcopy(neural_net.state_dict()), f"{self.log_dir}/model/best_model_state_dict.pt")
        else:
            self._epochs_since_last_improvement += 1

        # If no validation improvement over many epochs, stop training.
        if self._epochs_since_last_improvement > stop_after_epochs - 1:
            # neural_net.load_state_dict(self._best_model_state_dict)
            converged = True
            self._neural_net.load_state_dict(self._best_model_state_dict)
            self._val_log_prob = self._best_val_log_prob
            self._epochs_since_last_improvement = 0
            torch.save(deepcopy(neural_net.state_dict()), f"{self.log_dir}/model/best_model_state_dict.pt")
            
        return converged
    
    def check_posterior_behavior(self, config, limits, log_dir):
        with torch.no_grad():
            
            epoch = self.epoch
            current_net = deepcopy(self._neural_net)

            if epoch%config['train']['posterior']['step'] == 0: 
                
                print("\nPlotting posterior...")

                posterior = self.build_posterior(current_net)

                for fig_idx in range(len(self.posterior_train_set['x'])):
                    
                    figure = plt.figure()
                    plt.imshow(self.posterior_train_set['x'][fig_idx][:100, :].cpu())
                    plt.savefig(f'{log_dir}/posterior/figures/x_train_{fig_idx}.png')
                    plt.close(figure)
                    
                    figure = plt.figure()
                    plt.imshow(self.posterior_val_set['x'][fig_idx][:100, :].cpu())
                    plt.savefig(f'{log_dir}/posterior/figures/x_val_{fig_idx}.png')
                    plt.close(figure)
                    
                    fig_x, _ = plot_posterior_seen(
                        posterior       = posterior, 
                        sample_num      = config['train']['posterior']['sampling_num'],
                        x               = self.posterior_train_set['x'][fig_idx].to(self._device),
                        true_params     = self.posterior_train_set['theta'][fig_idx],
                        limits          = limits,
                        prior_labels    = config['prior']['prior_labels'],
                    )
                    plt.savefig(f"{log_dir}/posterior/figures/posterior_x_train_{fig_idx}_epoch_{epoch}.png")
                    plt.close(fig_x)

                    fig_x_val, _ = plot_posterior_seen(
                        posterior       = posterior, 
                        sample_num      = config['train']['posterior']['sampling_num'],
                        x               = self.posterior_val_set['x'][fig_idx].to(self._device),
                        true_params     = self.posterior_val_set['theta'][fig_idx],
                        limits          = limits,
                        prior_labels    = config['prior']['prior_labels'],
                    )
                    plt.savefig(f"{log_dir}/posterior/figures/posterior_x_val_{fig_idx}_epoch_{epoch}.png")
                    plt.close(fig_x_val)
                    
                print(f"posteriors plots saved to {log_dir}/posterior/figures/")
        
    def get_dataloaders(
        self,
        starting_round: int = 0,
        training_batch_size: int = 50,
        validation_fraction: float = 0.1,
        resume_training: bool = False,
        seed: Optional[int] = 100,
        dataset_kwargs: Optional[dict] = None,
        dataloader_kwargs: Optional[dict] = None,
    ) -> Tuple[data.DataLoader, data.DataLoader]:
        """Return dataloaders for training and validation.

        Args:
            dataset: holding all theta and x, optionally masks.
            training_batch_size: training arg of inference methods.
            resume_training: Whether the current call is resuming training so that no
                new training and validation indices into the dataset have to be created.
            dataloader_kwargs: Additional or updated kwargs to be passed to the training
                and validation dataloaders (like, e.g., a collate_fn).

        Returns:
            Tuple of dataloaders for training and validation.

        """

        # theta, x, prior_masks = self.get_simulations(starting_round)
        # dataset = data.TensorDataset(theta, x, prior_masks)

        dataset = MyDataset(
            data_path           = dataset_kwargs['data_path'],
            num_theta_each_set  = dataset_kwargs['num_theta_each_set'],
            seqC_process        = dataset_kwargs['seqC_process'],
            nan2num             = dataset_kwargs['nan2num'],
            summary_type        = dataset_kwargs['summary_type'],
        )

        
        # Get total number of training examples.
        num_examples = len(dataset)
        # Select random train and validation splits from (theta, x) pairs.
        num_training_examples = int((1 - validation_fraction) * num_examples)
        num_validation_examples = num_examples - num_training_examples

        # if not resume_training:
        # Seperate indicies for training and validation
        permuted_indices = torch.randperm(num_examples)
        self.train_indices, self.val_indices = (
            permuted_indices[:num_training_examples],
            permuted_indices[num_training_examples:],
        )

        # Create training and validation loaders using a subset sampler.
        # Intentionally use dicts to define the default dataloader args
        # Then, use dataloader_kwargs to override (or add to) any of these defaults
        # https://stackoverflow.com/questions/44784577/in-method-call-args-how-to-override-keyword-argument-of-unpacked-dict
        train_loader_kwargs = {
            "batch_size": min(training_batch_size, num_training_examples),
            "drop_last" : True,
            "sampler"   : SubsetRandomSampler(self.train_indices.tolist()),
            # "pin_memory": True,
        }
        val_loader_kwargs = {
            "batch_size": min(training_batch_size, num_validation_examples),
            "shuffle"   : False,
            "drop_last" : True,
            "sampler"   : SubsetRandomSampler(self.val_indices.tolist()),
            # "pin_memory": True,
        }
        if dataloader_kwargs is not None:
            train_loader_kwargs = dict(train_loader_kwargs, **dataloader_kwargs)
            val_loader_kwargs = dict(val_loader_kwargs, **dataloader_kwargs)

        print(f'\n--- data loader ---\nfinal train_loader_kwargs: \n{train_loader_kwargs}')
        print(f'final val_loader_kwargs: \n{val_loader_kwargs}')
        
        g = torch.Generator()
        g.manual_seed(seed)
        
        # train_loader = MyDataLoader(dataset, generator=g, **train_loader_kwargs)
        # val_loader   = MyDataLoader(dataset, generator=g, **val_loader_kwargs)
        train_loader = data.DataLoader(dataset, generator=g, **train_loader_kwargs)
        val_loader   = data.DataLoader(dataset, generator=g, **val_loader_kwargs)
        
        # load and show one example of the dataset
        print('\nloading one batch of the training dataset...')
        start_time = time.time()
        x, theta = next(iter(train_loader))
        print(f'loading one batch of the training dataset takes {time.time() - start_time:.2f} seconds')
        
        print(f'\n--- dataset ---\nwhole original dataset size [{len(dataset)}]',
              f'\none batch of the training dataset:',
              f'\n| x info:', f'shape {x.shape}, dtype: {x.dtype}, device: {x.device}',
              f'\n| theta info:', f'shape {theta.shape}, dtype: {theta.dtype}, device: {theta.device}',
             )
        
        print(f'\n--- collect posterior set ---\n')
        start_time = time.time()
        self.posterior_train_set = {
            'x': [],
            'theta': [],
        }
        
        self.posterior_val_set = {
            'x': [],
            'theta': [],
        }
        
        self.posterior_train_set['x'].append(x[0, ...])
        self.posterior_train_set['x'].append(x[1, ...])
        self.posterior_train_set['theta'].append(theta[0, ...])
        self.posterior_train_set['theta'].append(theta[1, ...])
        
        x, theta = next(iter(train_loader))
        self.posterior_train_set['x'].append(x[0, ...])
        self.posterior_train_set['x'].append(x[1, ...])
        self.posterior_train_set['theta'].append(theta[0, ...])
        self.posterior_train_set['theta'].append(theta[1, ...])
        
        x, theta = next(iter(val_loader))
        self.posterior_val_set['x'].append(x[0, ...])
        self.posterior_val_set['x'].append(x[1, ...])
        self.posterior_val_set['theta'].append(theta[0, ...])
        self.posterior_val_set['theta'].append(theta[1, ...])
        
        x, theta = next(iter(val_loader))
        self.posterior_val_set['x'].append(x[0, ...])
        self.posterior_val_set['x'].append(x[1, ...])
        self.posterior_val_set['theta'].append(theta[0, ...])
        self.posterior_val_set['theta'].append(theta[1, ...])
        
        print(f'collect posterior set takes {time.time() - start_time:.2f} seconds')
        return train_loader, val_loader
    
    def train_base(
        self,
        training_batch_size: int = 50,
        learning_rate: float = 5e-4,
        validation_fraction: float = 0.1,
        stop_after_epochs: int = 20,
        max_num_epochs: int = 2**31 - 1,
        clip_max_norm: Optional[float] = 5.0,
        
        calibration_kernel: Optional[Callable] = None,
        resume_training: bool = False, # False in the first epoch
        force_first_round_loss: bool = False, # True for the first round
        discard_prior_samples: bool = False, # False for the first round
        retrain_from_scratch: bool = False,
        show_train_summary: bool = True,
        
        seed: Optional[int] = 100,
        dataset_kwargs: Optional[dict] = None,
        dataloader_kwargs: Optional[dict] = None,
        
        config = None,
        limits = None,
        log_dir= None,
    ) -> nn.Module:
        r"""Return density estimator that approximates the distribution $p(\theta|x)$.

        Args:
            training_batch_size: Training batch size.
            learning_rate: Learning rate for Adam optimizer.
            validation_fraction: The fraction of data to use for validation.
            stop_after_epochs: The number of epochs to wait for improvement on the
                validation set before terminating training.
            max_num_epochs: Maximum number of epochs to run. If reached, we stop
                training even when the validation loss is still decreasing. Otherwise,
                we train until validation loss increases (see also `stop_after_epochs`).
            clip_max_norm: Value at which to clip the total gradient norm in order to
                prevent exploding gradients. Use None for no clipping.
            calibration_kernel: A function to calibrate the loss with respect to the
                simulations `x`. See Lueckmann, Gonçalves et al., NeurIPS 2017.
            resume_training: Can be used in case training time is limited, e.g. on a
                cluster. If `True`, the split between train and validation set, the
                optimizer, the number of epochs, and the best validation log-prob will
                be restored from the last time `.train()` was called.
            force_first_round_loss: If `True`, train with maximum likelihood,
                i.e., potentially ignoring the correction for using a proposal
                distribution different from the prior.
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
            Density estimator that approximates the distribution $p(\theta|x)$.
        """
        self.log_dir = log_dir
        
        # Load data from most recent round.
        self._round = max(self._data_round_index)

        if self._round == 0 and self._neural_net is not None:
            assert force_first_round_loss, (
                "You have already trained this neural network. After you had trained "
                "the network, you again appended simulations with `append_simulations"
                "(theta, x)`, but you did not provide a proposal. If the new "
                "simulations are sampled from the prior, you can set "
                "`.train(..., force_first_round_loss=True`). However, if the new "
                "simulations were not sampled from the prior, you should pass the "
                "proposal, i.e. `append_simulations(theta, x, proposal)`. If "
                "your samples are not sampled from the prior and you do not pass a "
                "proposal and you set `force_first_round_loss=True`, the result of "
                "SNPE will not be the true posterior. Instead, it will be the proposal "
                "posterior, which (usually) is more narrow than the true posterior."
            )

        # Calibration kernels proposed in Lueckmann, Gonçalves et al., 2017.
        if calibration_kernel is None:
            calibration_kernel = lambda x: ones([len(x)], device=self._device)

        # Starting index for the training set (1 = discard round-0 samples).
        start_idx = int(discard_prior_samples and self._round > 0)

        # For non-atomic loss, we can not reuse samples from previous rounds as of now.
        # SNPE-A can, by construction of the algorithm, only use samples from the last
        # round. SNPE-A is the only algorithm that has an attribute `_ran_final_round`,
        # so this is how we check for whether or not we are using SNPE-A.
        if self.use_non_atomic_loss or hasattr(self, "_ran_final_round"):
            start_idx = self._round

        # Set the proposal to the last proposal that was passed by the user. For
        # atomic SNPE, it does not matter what the proposal is. For non-atomic
        # SNPE, we only use the latest data that was passed, i.e. the one from the
        # last proposal.
        proposal = self._proposal_roundwise[-1]

        train_loader, val_loader = self.get_dataloaders(
            start_idx,
            training_batch_size,
            validation_fraction,
            resume_training,
            seed=seed,
            dataset_kwargs=dataset_kwargs,
            dataloader_kwargs=dataloader_kwargs,
        )
        # First round or if retraining from scratch:
        # Call the `self._build_neural_net` with the rounds' thetas and xs as
        # arguments, which will build the neural network.
        # This is passed into NeuralPosterior, to create a neural posterior which
        # can `sample()` and `log_prob()`. The network is accessible via `.net`.
        if self._neural_net is None or retrain_from_scratch:

            # Get theta,x to initialize NN
            # theta, x, _ = self.get_simulations(starting_round=start_idx)
            x, theta = next(iter(train_loader))
            
            # Use only training data for building the neural net (z-scoring transforms)
            self._neural_net = self._build_neural_net(
                theta[0:2].to("cpu"),
                x[0:2].to("cpu"),
            )
            self._x_shape = x_shape_from_simulation(x.to("cpu"))
            
            print('\nfinished build network')
            print(f'\n{self._neural_net}')
            test_posterior_net_for_multi_d_x(
                self._neural_net,
                theta.to("cpu"),
                x.to("cpu"),
            )

            del theta, x

        # Move entire net to device for training.
        self._neural_net.to(self._device)

        if not resume_training:
            self.optimizer = optim.Adam(
                list(self._neural_net.parameters()), lr=learning_rate
            )
            self.epoch, self._val_log_prob = 0, float("-Inf")

        while self.epoch <= max_num_epochs and not self._converged(
            self.epoch, stop_after_epochs
        ):
            starting_time = time.time()
            # Train for a single epoch.
            self._neural_net.train()
            train_log_probs_sum = 0
            epoch_start_time = time.time()
            train_batch_num = 0
            batch_timer = time.time()
            for batch in train_loader:
                self.optimizer.zero_grad()
                # Get batches on current device.
                # theta_batch, x_batch, masks_batch = (
                #     batch[0].to(self._device),
                #     batch[1].to(self._device),
                #     batch[2].to(self._device),
                # )
                x_batch, theta_batch = (
                    batch[0].to(self._device),
                    batch[1].to(self._device)
                )
                masks_batch = torch.ones_like(theta_batch[:, 0])

                train_losses = self._loss(
                    theta_batch,
                    x_batch,
                    masks_batch,
                    proposal,
                    calibration_kernel,
                    force_first_round_loss=force_first_round_loss,
                )
                train_loss = torch.mean(train_losses)
                train_log_probs_sum -= train_losses.sum().item()

                train_loss.backward()
                if clip_max_norm is not None:
                    clip_grad_norm_(
                        self._neural_net.parameters(), max_norm=clip_max_norm
                    )
                self.optimizer.step()
                if train_batch_num % 5 == 0:
                    print(f'epoch {self.epoch}: batch {train_batch_num} train_loss {-1*train_loss:.2f}, train_log_probs_sum {train_log_probs_sum:.2f}, time {(time.time() - batch_timer)/60:.2f}min')
                    # batch_timer = time.time()
                train_batch_num += 1
                
            self.epoch += 1

            train_log_prob_average = train_log_probs_sum / (
                len(train_loader) * train_loader.batch_size  # type: ignore
            )
            self._summary["training_log_probs"].append(train_log_prob_average)

            # Calculate validation performance.
            self._neural_net.eval()
            val_log_prob_sum = 0
            
            val_batch_num = 0
            with torch.no_grad():
                for batch in val_loader:
                    # theta_batch, x_batch, masks_batch = (
                    #     batch[0].to(self._device),
                    #     batch[1].to(self._device),
                    #     batch[2].to(self._device),
                    # )
                    x_batch, theta_batch = (
                        batch[0].to(self._device),
                        batch[1].to(self._device)
                    )
                    masks_batch = torch.ones_like(theta_batch[:, 0])
                    # Take negative loss here to get validation log_prob.
                    val_losses = self._loss(
                        theta_batch,
                        x_batch,
                        masks_batch,
                        proposal,
                        calibration_kernel,
                        force_first_round_loss=force_first_round_loss,
                    )
                    val_log_prob_sum -= val_losses.sum().item()
                    if val_batch_num % 5 == 0:
                        print(f'epoch {self.epoch}: val_log_prob_sum {val_log_prob_sum:.2f}')
                    val_batch_num += 1

            # Take mean over all validation samples.
            self._val_log_prob = val_log_prob_sum / (
                len(val_loader) * val_loader.batch_size  # type: ignore
            )
            # Log validation log prob for every epoch.
            self._summary["validation_log_probs"].append(self._val_log_prob)
            self._summary["epoch_durations_sec"].append(time.time() - epoch_start_time)

            self._maybe_show_progress(self._show_progress_bars, self.epoch, starting_time, train_log_prob_average, self._val_log_prob)

            self.check_posterior_behavior(config, limits, log_dir)
        # self._report_convergence_at_end(self.epoch, stop_after_epochs, max_num_epochs)
        # self._val_log_prob = self._best_val_log_prob
        
        # Update summary.
        self._summary["epochs_trained"].append(self.epoch)
        self._summary["best_validation_log_prob"].append(self._best_val_log_prob)

        # Update tensorboard and summary dict.
        self._summarize(round_=self._round)

        # Update description for progress bar.
        if show_train_summary:
            print(self._describe_round(self._round, self._summary))

        # load best model from state dict
        self._neural_net.load_state_dict(self._best_model_state_dict)
        self._val_log_prob = self._best_val_log_prob
        self._epochs_since_last_improvement = 0
        
        # Avoid keeping the gradients in the resulting network, which can
        # cause memory leakage when benchmarking.
        self._neural_net.zero_grad(set_to_none=True)

        figure = plt.figure(figsize=(16, 10))
        plt.plot(self._summary["training_log_probs"], label="training")
        plt.plot(self._summary["validation_log_probs"], label="validation")
        plt.xlabel("Epoch")
        plt.ylabel("Log prob")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(log_dir, f"model/round_{self._round}_training_curve.png"))
        plt.close(figure)
        
        return self, deepcopy(self._neural_net)
    
    def _describe_round(self, round_: int, summary: Dict[str, list]) -> str:
        epochs = summary["epochs_trained"][-1]
        best_validation_log_prob = summary["best_validation_log_prob"][-1]

        description = f"""
        -------------------------
        ||||| ROUND {round_} STATS |||||:
        -------------------------
        Epochs trained: {epochs}
        Best validation performance: {best_validation_log_prob:.4f}, from epoch {self._best_model_from_epoch:5}
        Model from best epoch {self._best_model_from_epoch} is loaded for further training
        -------------------------
        """

        return description

    def append_simulations_for_run(
        self,
        theta: Tensor,
        x: Tensor,
        current_round: int = 0,
        exclude_invalid_x: Optional[bool] = None,
        data_device: Optional[str] = None,
    ):
        """ update theta and x for the current round
        """
        if exclude_invalid_x is None:
            if current_round == 0:
                exclude_invalid_x = True
            else:
                exclude_invalid_x = False

        if data_device is None:
            data_device = self._device

        theta, x = validate_theta_and_x(
            theta, x, data_device=data_device, training_device=self._device
        )

        is_valid_x, num_nans, num_infs = handle_invalid_x(
            x, exclude_invalid_x=exclude_invalid_x
        )

        x = x[is_valid_x]
        theta = theta[is_valid_x]

        # Check for problematic z-scoring
        warn_if_zscoring_changes_data(x)
        if (
            type(self).__name__ == "SNPE_C"
            and current_round > 0
            and not self.use_non_atomic_loss
        ):
            nle_nre_apt_msg_on_invalid_x(
                num_nans, num_infs, exclude_invalid_x, "Multiround SNPE-C (atomic)"
            )
        else:
            npe_msg_on_invalid_x(
                num_nans, num_infs, exclude_invalid_x, "Single-round NPE"
            )

        prior_masks = mask_sims_from_prior(int(current_round > 0), theta.size(0))

        old_theta   = self._theta_roundwise[current_round]
        old_x       = self._x_roundwise[current_round]
        old_masks   = self._prior_masks[current_round]
        
        self._theta_roundwise[current_round] = torch.cat((old_theta, theta), dim=0)
        self._x_roundwise[current_round]     = torch.cat((old_x, x), dim=0)
        self._prior_masks[current_round]     = torch.cat((old_masks, prior_masks), dim=0)
        
        return self
    

class MySNPE_C(SNPE_C, MyPosteriorEstimator):
    def __init__(
        self,
        prior: Optional[Distribution] = None,
        density_estimator: Union[str, Callable] = "maf",
        device: str = "gpu",
        logging_level: Union[int, str] = "INFO",
        summary_writer: Optional[SummaryWriter] = None,
        show_progress_bars: bool = True,    
    ):
        kwargs = del_entries(locals(), entries=("self", "__class__"))
        # print(kwargs)
        super().__init__(**kwargs)
    
    def train(
        self,
        num_atoms: int = 10,
        training_batch_size: int = 50,
        learning_rate: float = 5e-4,
        validation_fraction: float = 0.1,
        stop_after_epochs: int = 20,
        max_num_epochs: int = 2**31 - 1,
        clip_max_norm: Optional[float] = 5.0,
        calibration_kernel: Optional[Callable] = None,
        resume_training: bool = False,
        force_first_round_loss: bool = False,
        discard_prior_samples: bool = False,
        use_combined_loss: bool = False,
        retrain_from_scratch: bool = False,
        show_train_summary: bool = False,
        seed: Optional[int] = 100,
        dataset_kwargs: Optional[Dict] = None,
        dataloader_kwargs: Optional[Dict] = None,
        config = None,
        limits = None,
        log_dir= None,
    ) -> nn.Module:
        r"""Return density estimator that approximates the distribution $p(\theta|x)$.

        Args:
            num_atoms: Number of atoms to use for classification.
            training_batch_size: Training batch size.
            learning_rate: Learning rate for Adam optimizer.
            validation_fraction: The fraction of data to use for validation.
            stop_after_epochs: The number of epochs to wait for improvement on the
                validation set before terminating training.
            max_num_epochs: Maximum number of epochs to run. If reached, we stop
                training even when the validation loss is still decreasing. Otherwise,
                we train until validation loss increases (see also `stop_after_epochs`).
            clip_max_norm: Value at which to clip the total gradient norm in order to
                prevent exploding gradients. Use None for no clipping.
            calibration_kernel: A function to calibrate the loss with respect to the
                simulations `x`. See Lueckmann, Gonçalves et al., NeurIPS 2017.
            resume_training: Can be used in case training time is limited, e.g. on a
                cluster. If `True`, the split between train and validation set, the
                optimizer, the number of epochs, and the best validation log-prob will
                be restored from the last time `.train()` was called.
            force_first_round_loss: If `True`, train with maximum likelihood,
                i.e., potentially ignoring the correction for using a proposal
                distribution different from the prior.
            discard_prior_samples: Whether to discard samples simulated in round 1, i.e.
                from the prior. Training may be sped up by ignoring such less targeted
                samples.
            use_combined_loss: Whether to train the neural net also on prior samples
                using maximum likelihood in addition to training it on all samples using
                atomic loss. The extra MLE loss helps prevent density leaking with
                bounded priors.
            retrain_from_scratch: Whether to retrain the conditional density
                estimator for the posterior from scratch each round.
            show_train_summary: Whether to print the number of epochs and validation
                loss and leakage after the training.
            dataloader_kwargs: Additional or updated kwargs to be passed to the training
                and validation dataloaders (like, e.g., a collate_fn)

        Returns:
            Density estimator that approximates the distribution $p(\theta|x)$.
        """

        # WARNING: sneaky trick ahead. We proxy the parent's `train` here,
        # requiring the signature to have `num_atoms`, save it for use below, and
        # continue. It's sneaky because we are using the object (self) as a namespace
        # to pass arguments between functions, and that's implicit state management.
        self._num_atoms = num_atoms
        self._use_combined_loss = use_combined_loss
        kwargs = del_entries(
            locals(), entries=("self", "__class__", "num_atoms", "use_combined_loss")
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
                and isinstance(self._neural_net._distribution, mdn)
                and check_dist_class(
                    self._prior, class_to_check=(Uniform, MultivariateNormal)
                )[0]
            )

            algorithm = "non-atomic" if self.use_non_atomic_loss else "atomic"
            print(f"Using SNPE-C with {algorithm} loss")

            if self.use_non_atomic_loss:
                # Take care of z-scoring, pre-compute and store prior terms.
                self._set_state_for_mog_proposal()

        return super().train_base(**kwargs)
    

