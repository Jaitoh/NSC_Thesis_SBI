#!/bin/bash
cd ~/data/NSC/codes
source activate sbi

ROOT_DIR="$HOME/data/NSC"
code /home/wehe/data/NSC/codes/notebook/DM_inference.log

nice python3 /home/wehe/data/NSC/codes/notebook/DM_inference.py >/home/wehe/data/NSC/codes/notebook/DM_inference.log 2>&1
