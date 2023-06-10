import matplotlib.pyplot as plt
import numpy as np

class ConvergeTest:
    
    def __init__(self, log_probs):
        print(log_probs)
        plt.plot(log_probs, '.-')
        plt.savefig('converge_test.png')
        
        self.log_probs = log_probs
        
        self._summary = {}
        self._summary["epoch_durations_sec"] = []
        self._summary["training_log_probs"] = []
        self._summary["validation_log_probs"] = []
        self._summary["test_log_probs"] = []
        self._summary["learning_rates"] = []

        self._summary = {}
        self._summary["best_from_epoch_of_current_dset"] = []
        self._summary["best_from_dset_of_current_dset"] = []
        self._summary["num_epochs_of_current_dset"] = []
        # ==================== initialize ==================== #
        # initialize log probs and counters
        self._init_log_prob()
        self.epoch_counter  = 0
        self._best_model_from_epoch = -1
        # initialize test dataset and test loader -> self.test_dataset, self.test_loader
        while (self.dset <= 10000
                and not self._converged_dset()
                ):
            
            self.epoch  = 0
            
            # train and validate until no validation performance improvement
            while (
                self.epoch <= 10000
                and not self._converged()
            ):  
                
                # train and validate for one epoch
                # if len(self.log_probs) > 0:
                self.train_valid_one_epoch()
                # else:
                #     break
                # print(f'epoch {self.epoch}: {self._val_log_prob}')
                self.epoch += 1
                self.epoch_counter += 1

                self._summary["validation_log_probs"].append(self._val_log_prob)
            
            self.dset += 1
            self._val_log_prob_dset = self._best_val_log_prob # 更新dset的val_log_prob
                
        plt.plot(self._summary["validation_log_probs"])
        # plt.plot(self._summary["best_from_dset_of_current_dset"], 'x')
        plt.grid()
        
        # print(self._summary["best_from_dset_of_current_dset"])
        # print(self._summary["num_epochs_of_current_dset"])
        # print(self._summary["best_from_epoch_of_current_dset"])
    
        best_dset_idx = np.array(self._summary["best_from_dset_of_current_dset"][1:], dtype=int)
        starting_epoch_dset = np.array(self._summary["num_epochs_of_current_dset"][1:-1], dtype=int)+1
        starting_epoch_dset = np.insert(starting_epoch_dset, 0, 0)
        best_epoch = np.array(self._summary["best_from_epoch_of_current_dset"][1:], dtype=int)
        
        best_idxs = [starting_epoch_dset[dset] + best_epoch for dset, best_epoch in zip( best_dset_idx, best_epoch)]
        self._summary["best_epoches"] = best_idxs
        
        y_values = [self._summary["validation_log_probs"][i] for i in best_idxs]
        plt.plot(best_idxs, y_values, 'v')
        
        plt.savefig('converge_test_after.png')

    def train_valid_one_epoch(self):
        self._val_log_prob = self.log_probs.pop(0)
        
    def _init_log_prob(self):
        
        # init log prob
        self.run    = 0
        self.dset   = 0
        self.epoch  = 0
        self._epoch_of_last_dset = 0
        self._val_log_prob, self._val_log_prob_dset = float("-Inf"), float("-Inf")
        self._best_val_log_prob, self._best_val_log_prob_dset = float("-Inf"), float("-Inf")
    
    def _converged(self):
        """Return whether the training converged yet and save best model state so far.

        Checks for improvement in validation performance over previous epochs.

        Args:
            epoch: Current epoch in training.
            stop_after_epochs: How many fruitless epochs to let pass before stopping.

        Returns:
            Whether the training has stopped improving, i.e. has converged.
        """
        epoch                   = self.epoch
        improvement_threshold   = 0.001
        min_num_epochs          = 3
        stop_after_epochs       = 3
        
        converged = False

        # assert self._neural_net is not None
        # neural_net = self._neural_net

        # (Re)-start the epoch count with the first epoch or any improvement. 
        improvement = self._val_log_prob - self._best_val_log_prob
        if epoch == 0 or ((self._val_log_prob > self._best_val_log_prob) and (improvement >= improvement_threshold)):
            
            self._epochs_since_last_improvement = 0
            
            self._best_val_log_prob     = self._val_log_prob
            # self._best_model_state_dict = deepcopy(neural_net.state_dict())
            self._best_model_from_epoch = epoch - 1
            
            # if epoch != 0: #and epoch%self.config['train']['posterior']['step'] == 0:
                # self._posterior_behavior_log(self.prior_limits) # plot posterior behavior when best model is updated
                # print_mem_info(f"{'gpu memory usage after posterior behavior log':46}", DO_PRINT_MEM)
            # torch.save(deepcopy(neural_net.state_dict()), f"{self.log_dir}/model/best_model_state_dict_run{self.run}.pt")
            
        else:
            self._epochs_since_last_improvement += 1

        # If no validation improvement over many epochs, stop training.
        if self._epochs_since_last_improvement > stop_after_epochs - 1 and epoch > min_num_epochs:
            # neural_net.load_state_dict(self._best_model_state_dict)
            converged = True
            # print(f"Converged after {epoch} epochs with validation log prob {self._val_log_prob:.4f} and best validation log prob {self._best_val_log_prob:.4f}.")
            
            # self._neural_net.load_state_dict(self._best_model_state_dict)
            self._val_log_prob = self._best_val_log_prob
            self._epochs_since_last_improvement = 0
            # self._epoch_of_last_dset = epoch - 1
            self._epoch_of_last_dset = self.epoch_counter-1
        
        # log info for this dset
        # self._summary_writer.add_scalar(f"run{self.run}/best_val_epoch_glob", self._best_model_from_epoch, self.epoch_counter-1)
        # self._summary_writer.add_scalar(f"run{self.run}/best_val_log_prob_glob", self._best_val_log_prob, self.epoch_counter-1)
        # self._summary_writer.add_scalar(f"run{self.run}/current_dset_glob", self.dset_counter, self.epoch_counter-1)
        # self._summary_writer.add_scalar(f"run{self.run}/num_chosen_dset_glob", self.num_train_sets, self.epoch_counter-1)
        # self._summary_writer.flush()
        return converged
    
    def _converged_dset(self):
        
        improvement_threshold = 0.001
        min_num_dsets         = 3
        stop_after_dsets      = 3
        
        converged = False
        # assert self._neural_net is not None
        
        # improvement = self._val_log_prob - self._best_val_log_prob
        self._summary["num_epochs_of_current_dset"].append(self._epoch_of_last_dset)
        if self.dset == 0 or (self._val_log_prob_dset > self._best_val_log_prob_dset):
            
            self._dset_since_last_improvement = 0
            
            self._best_val_log_prob_dset        = self._val_log_prob_dset
            # self._best_model_state_dict_dset    = deepcopy(self._neural_net.state_dict())
            self._best_model_from_dset          = self.dset - 1
            
            self._summary["best_from_epoch_of_current_dset"].append(self._best_model_from_epoch)
            self._summary["best_from_dset_of_current_dset"].append(self._best_model_from_dset)
            # torch.save(deepcopy(self._neural_net.state_dict()), f"{self.log_dir}/model/best_model_state_dict_run{self.run}.pt")
            # print(f"best val log prob dset {self.dset} is {self._best_val_log_prob_dset}")
        else:
            self._dset_since_last_improvement += 1
            # print(f"no improvement in dset {self.dset} since {self._dset_since_last_improvement} dsets")
        if self._dset_since_last_improvement > stop_after_dsets - 1 and self.dset > min_num_dsets - 1:
            
            converged = True
            # print(f"converged at dset {self.dset}")
            # self._neural_net.load_state_dict(self._best_model_state_dict_dset)
            self._val_log_prob_dset = self._best_val_log_prob_dset
            self._dset_since_last_improvement = 0
            
            # torch.save(deepcopy(self._neural_net.state_dict()), f"{self.log_dir}/model/best_model_state_dict_run{self.run}.pt")
        
        # use only the whole dataset as the training set, would train only once
        # if self.config.dataset.one_dataset == True and self.dset == 1:
        #     converged = True
        
        return converged

log_probs = [-12, -11.5, -11, -12, -12.5, -13, 
            # -12, -13, -14, -12,
            -10.8, -11, -12, -13,
            -10.5, -11, -12, -13,
            -11, -10.5, -10, -11, -11.5, -12,
            -9, -8, -6, -7, -8, -9,
            -11, -12, -13, -14, 
            -6, -7, -8, -9,
            -5, -7, -9, -11,
            -9, -8, -7, -8, 
            -7, -8, -9, -10,
            -12, -14, -16, -18,
            ]

ConvergeTest(log_probs)