import sys
from sbi.inference.posteriors.direct_posterior import DirectPosterior
from torch.distributions import Distribution, MultivariateNormal, Uniform
from pyknos.mdn.mdn import MultivariateGaussianMDN as mdn
from sbi.inference import SNPE_C
from sbi.inference.snpe.snpe_base import PosteriorEstimator
from sbi.utils import del_entries, check_dist_class

sys.path.append("./src")
from utils.set_seed import setup_seed


class MyPosteriorEstimator_P4(PosteriorEstimator):
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

    def train_base_p4(
        self,
        config,
        prior_limits,
        dataloader_kwargs,
        continue_from_checkpoint=None,
        debug=False,
    ):
        self.config = config
        self.log_dir = config.log_dir
        self.prior_limits = prior_limits
        self.dataset_kwargs = self.config.dataset
        self.training_kwargs = self.config.train.training
        setup_seed(config.seed)

        # initialize values
        self.epoch = 0
        self._val_log_prob = float("-Inf")
        self._best_val_log_prob = float("-Inf")
        # self._best_model_from_epoch = -1

        # prepare train, val, test dataset and dataloader
        train_dataset
        val_dataset
        test_dataset

        train_dataloader

        val_dataloader
        test_dataloader

        # collect posterior sets

        # train until no validation improvement

        # avoid keeping gradients in resulting network
        # save best model


class MySNPE_C_P4(SNPE_C, MyPosteriorEstimator_P4):
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
        dataloader_kwargs,
        continue_from_checkpoint=None,
        debug=False,
    ):
        self.config = config
        self._num_atoms = self.config.train.training.num_atoms
        self._use_combined_loss = self.config.train.training.use_combined_loss
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
                and self._neural_net is not None
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

        return super().train_base_p4(**kwargs)  # type: ignore
