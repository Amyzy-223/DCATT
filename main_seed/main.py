import argparse
import os
import time

import numpy as np
import pandas as pd
import torch

from config import set_config


import warnings

from train import Engine
from EEGdatasets import EEGDataset
from util import setup_seed, load_pkl, generate_train_test_set


warnings.filterwarnings('ignore')

"""
Note: For first-time use, please prepare the dataset as follows: 
1) Download the SEED dataset and place the Preprocessed_EEG folder into ./datas/rawdata. 
2) Run ./datas/LoadData_emotion.py to generate the SEED_DCATT dataset.
"""

if __name__ == '__main__':
    # set hyper parameter
    args, foldname = set_config()

    ## set random seed
    setup_seed(args.seed)

    ## load dataset
    print('mode:', args.SubTestFlag)
    print('dataset:', args.dataset)
    dataset = load_pkl(os.path.join(args.data_dir, args.dataset, 'seedDataset.pkl'))['seedDataset']
    idx_dict = load_pkl(os.path.join(args.data_dir, args.dataset, args.SubTestFlag + '.pkl'))
    if args.SubTestFlag in ['LOSO', 'INS']:
        subject = args.subject
    else:
        subject = 0
    PhyDataset = generate_train_test_set(dataset=dataset, idx_dict=idx_dict, sub=subject)



    ## train
    engine = Engine(args, PhyDataset, foldname)
    engine.one_stage_run()
    engine.two_stage_run()

