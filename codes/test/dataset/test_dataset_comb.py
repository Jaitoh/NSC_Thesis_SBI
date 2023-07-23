import unittest
import torch
import os
import sys
from pathlib import Path

NSC_DIR = Path(__file__).resolve().parent.parent.parent.parent.as_posix()  # NSC dir
sys.path.append(f"{NSC_DIR}/codes/src")
from train_nle.Dataset import probR_Comb_Dataset, chR_Comb_Dataset


class TestDatasets(unittest.TestCase):
    def setUp(self):
        self.data_dir = "/home/ubuntu/tmp/NSC/data/dataset-comb"
        self.num_chosen_theta = 50
        self.chosen_dur = [3, 9]
        self.part_each_dur = [1, 1]
        self.max_theta = 500
        self.theta_chosen_mode = "random"
        self.num_probR_sample = 100
        self.chR_mode = "online"
        self.probR_Comb_Dataset = probR_Comb_Dataset(
            data_dir=self.data_dir,
            num_chosen_theta=self.num_chosen_theta,
            chosen_dur=self.chosen_dur,
            part_each_dur=self.part_each_dur,
            max_theta=self.max_theta,
            theta_chosen_mode=self.theta_chosen_mode,
        )
        self.chR_Comb_Dataset = chR_Comb_Dataset(
            data_dir=self.data_dir,
            num_chosen_theta=self.num_chosen_theta,
            chosen_dur=self.chosen_dur,
            part_each_dur=self.part_each_dur,
            max_theta=self.max_theta,
            theta_chosen_mode=self.theta_chosen_mode,
            num_probR_sample=self.num_probR_sample,
            chR_mode=self.chR_mode,
        )

    def test_probR_Comb_Dataset(self):
        seqC, theta, probR = self.probR_Comb_Dataset[0]
        self.assertIsInstance(seqC, torch.Tensor)
        self.assertIsInstance(theta, torch.Tensor)
        self.assertIsInstance(probR, torch.Tensor)
        self.assertEqual(len(seqC.shape), 1)
        self.assertEqual(len(theta.shape), 1)
        self.assertEqual(len(probR.shape), 1)

    def test_chR_Comb_Dataset(self):
        x, theta = self.chR_Comb_Dataset[0]
        self.assertIsInstance(x, torch.Tensor)
        self.assertIsInstance(theta, torch.Tensor)
        self.assertEqual(len(x.shape), 1)
        self.assertEqual(len(theta.shape), 1)


if __name__ == "__main__":
    unittest.main()
