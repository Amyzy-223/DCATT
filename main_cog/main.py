import argparse
import argparse
import os
import time

import numpy as np
import pandas as pd
import torch

from loaddata import load_dataset, PhyDataset
from train import cross_validation
from util import setup_seed, result_visualization, write_log, save_pkl, test_and_visualization, load_pkl
from retrain import retrain_test

import warnings
warnings.filterwarnings('ignore')


if __name__ == '__main__':

    ## set hyperparameters
    t = time.localtime()
    foldname = str(t.tm_yday).zfill(3) + str(t.tm_hour).zfill(2) + str(t.tm_min).zfill(2) + str(t.tm_sec).zfill(3) + '/'
    ## hyperparameter

    parser = argparse.ArgumentParser(description='parser example')
    parser.add_argument('--lr', default=1E-2, type=float, help='learning rate 1E-1')
    parser.add_argument('--batch_size', default=64, type=int, help='batch size')
    parser.add_argument('--weight_decay', default=1E-2, type=float, help='weight decay')
    parser.add_argument('--epoch_num', default=100, type=int, help='num epochs')
    parser.add_argument('--optimizer', default='sgd', type=str, help='optimizer')
    parser.add_argument('--scheduler', default='cosine', type=str, help='scheduler: only cosine')
    parser.add_argument('--seed', default=3407, type=int, help='random seed 3407 42')
    parser.add_argument('--esPatience', default=15, type=int, help='early stopping patience')
    parser.add_argument('--dropout', default=0.3, type=float, help='drop out')
    parser.add_argument('--criterion', default='entropy', type=str, help='loss function: entropy/focal/weight:2.71')
    parser.add_argument('--alpha', default=3, type=float, help='ratio of major sample to minor sample')

    parser.add_argument('--SubTestFlag', default='dep', type=str, help='subtest flag: dep/ind/ccs (dependent/independent/cross case)')
    parser.add_argument('--sub_group', default=1, type=int, help='subject group LOSO, 0-3 or case group ccs, 1-4')
    parser.add_argument("--lenP", type=int, default=6, help="seq_length_x: 6", )
    parser.add_argument("--lenQ", type=int, default=4, help="seq_length_y", )
    parser.add_argument("--step", type=int, default=0, help="predict: step: step+lenQ") # 隔几步开始预测，连续预测就是隔0步

    parser.add_argument("--noise", default='-1', type=str, help="add noise, ratio between sigma and signal sd, 0 0.1 0.5 1")
    parser.add_argument("--dropCH", default=0, type=int, help="drop channel number, 1,2,3")
    parser.add_argument("--dropR", default=0, type=int, help="drop channel repeat number, 1-8")

    parser.add_argument('--modelname', default='DCATT', type=str, help='model name')
    parser.add_argument("--hidden_dim", type=int, default=64, help="hidden_dim")
    parser.add_argument("--end_dim", type=int, default=1024, help="end_dim")
    parser.add_argument("--layer", type=int, default=2, help="PhyModel5 layer")
    parser.add_argument('--fusionFlag', default='concat', type=str, help='fusionFlag: none,plus, concat, crossAtt, crossTimeAtt')
    parser.add_argument('--lstmLayer', default=2, type=int, help='lstm layer')
    parser.add_argument('--gcnFlag', default='gcn', type=str, help='spatial model flag for fnirs: gcn/conv/att/none')
    parser.add_argument('--tempFlag', default='att', type=str, help='temporal model flag for fnirs: att3d/att3_5d/att4d/none')
    parser.add_argument('--timeProj', default='att', type=str, help='time project method: att/lin/crossTimeAtt')
    parser.add_argument('--readout', default='mean', type=str)
    parser.add_argument('--predata', default='pre', type=str, help='predata method: pre/none')
    # do not change
    parser.add_argument("--root_dir", type=str, default="./train_result_2/", help="root dir")
    parser.add_argument('--fold_num', default=5, type=int, help='fold number')
    parser.add_argument('--device', default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), type=str, help='device')
    parser.add_argument("--result_dir", type=str, default="./result/", help="result dir")
    parser.add_argument("--model_dir", type=str, default="./model/", help="model dir")
    parser.add_argument("--plot_dir", type=str, default="./plot/", help="plot dir")
    parser.add_argument("--data_dir", type=str, default="./data/", help="Data directory.")
    parser.add_argument("--split_ratio_train", type=float, default=0.8, help="split ratio")

    args = parser.parse_args()
    args.criterion = (args.criterion, args.alpha)

    ## set random seed
    setup_seed(args.seed)

    ## load dataset
    if args.noise == '-1' and args.dropCH == 0 and args.dropR == 0:
        dataset_dir = os.path.join(args.data_dir, 'dataset', args.SubTestFlag +'_s'+str(args.sub_group)
                                   +'_P'+str(args.lenP)+'_Q'+str(args.lenQ)
                                   +'_S'+str(args.step)+'_seed'+str(args.seed)+'_D'+args.predata+'.pkl')
    elif args.noise != '-1' and args.dropCH == 0 and args.dropR == 0:
        dataset_dir = os.path.join(args.data_dir, 'data_noise_drop', 'noise', 'noise_'+str(args.noise),
                                   args.SubTestFlag + '_s' + str(args.sub_group) + '_N' + str(args.noise)
                                   + '_P' + str(args.lenP) + '_Q' + str(args.lenQ) + '_S' + str(args.step)
                                   + '_seed' + str(args.seed) + '_D' + args.predata + '.pkl')
    elif args.noise == '-1' and args.dropCH != 0 and args.dropR != 0:
        dataset_dir = os.path.join(args.data_dir, 'data_noise_drop', 'drop', 'drop_'+str(args.dropCH), 'd'+str(args.dropR),
                                   args.SubTestFlag + '_s' + str(args.sub_group) + '_Drop' + str(args.dropCH) + str(args.dropR)
                                   + '_P' + str(args.lenP) + '_Q' + str(args.lenQ) + '_S' + str(args.step)
                                   + '_seed' + str(args.seed) + '_D' + args.predata + '.pkl')
    else:
        print('error: args noise dropCH dropR')
    if os.path.exists(dataset_dir):
        print('exist dataset')
        PhyDataset = load_pkl(dataset_dir)
    else:
        print('new dataset')
        PhyDataset = load_dataset(args)
        save_pkl(PhyDataset, dataset_dir)


    ## train
    result_allfold, result_df, best_epoch = cross_validation(dataset=PhyDataset, args=args, foldname=foldname)

    ## result visualization
    df = pd.read_csv(os.path.join(args.root_dir, foldname, args.result_dir, 'Foldall_result.csv'))
    plot_dir = os.path.join(args.root_dir, foldname, args.plot_dir)
    lenQ = args.lenQ

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    result_visualization(df, plot_dir, lenQ)

    ## log
    write_log(df, foldname, args)

    ## test and visualization
    test_and_visualization(df, args.root_dir, foldname, dataset_dir, lenQ, True, args.criterion)

    ## retrain and test
    retrain_test(args, best_epoch, PhyDataset, foldname)



