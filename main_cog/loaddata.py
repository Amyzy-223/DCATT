import argparse
import os
import pickle
import math
import random

# import h5py
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
from torch.utils.data import Dataset, DataLoader, Subset, SubsetRandomSampler, ConcatDataset


from util import load_pkl, save_pkl

class PhyDataset(Dataset):
    def __init__(self, PhyDataset0, NormFlag, meanEcg, stdEcg, meanFnirs, stdFnirs):
        super(PhyDataset).__init__()

        self.labels = PhyDataset0['labels']
        self.labels_1d = PhyDataset0['labels_1d']
        self.ecgs = PhyDataset0['ecgs']
        self.fnirsxs = PhyDataset0['fnirsxs']
        self.fnirsA1s = PhyDataset0['fnirsA1s']
        self.fnirsA2s = PhyDataset0['fnirsA2s']
        self.fnirsA3s = PhyDataset0['fnirsA3s']
        self.fnirsA4s = PhyDataset0['fnirsA4s']
        self.NormFlag = NormFlag
        self.meanEcg = meanEcg
        self.stdEcg = stdEcg
        self.meanFnirs = meanFnirs
        self.stdFnirs = stdFnirs
        self.ecgs_norm = self.ecgs.copy()
        self.fnirsxs_norm = self.fnirsxs.copy()

        if self.NormFlag:  # 如果进行标准化
            # print('--norm--')
            for i, ecg in enumerate(self.ecgs):
                self.ecgs_norm[i] = (ecg - self.meanEcg[:, np.newaxis]) / self.stdEcg[:, np.newaxis]
            for i, fnirsx in enumerate(self.fnirsxs):
                self.fnirsxs_norm[i] = (fnirsx - self.meanFnirs[:, :, np.newaxis]) / self.stdFnirs[:, :, np.newaxis]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        label_1d = self.labels_1d[idx]
        ecg = self.ecgs_norm[idx]
        fnirsx = self.fnirsxs_norm[idx]
        fnirsA1 = self.fnirsA1s[idx]
        fnirsA2 = self.fnirsA2s[idx]
        fnirsA3 = self.fnirsA3s[idx]
        fnirsA4 = self.fnirsA4s[idx]

        return label, label_1d, ecg, fnirsx, fnirsA1, fnirsA2, fnirsA3, fnirsA4

def construct_dataset_subcase(subnum, casenum, ecg, fnirs, lenP, lenQ, step):
    """
    返回每个被试所有工况的数据集
    :param subnum: 被试编号
    :param cases: 工况集合
    :param ecg: 所有心电记录
    :param fnirs: 所有fnirs记录
    :param lenP: 训练步长
    :param lenQ: 预测步长
    :param step: 隔几步开始预测，预测 step: step+lenQ
    :return:
        PhyDataDict_sub： 单个被试的所有数据字典
    """
    labels, labels_1d, ecgs, fnirsxs, fnirsA1s, fnirsA2s, fnirsA3s, fnirsA4s = [], [], [], [], [], [], [], []

    ecg_subcase = ecg.loc[(ecg['sub'] == subnum) & (ecg['case'] == casenum), :]
    cuts_case = max(ecg_subcase['cut_case'])
    ecg_subcase.reset_index(drop=True, inplace=True)
    for j in range(cuts_case - lenP - lenQ - step + 1):
        # ! ECG handcraft features Z: shape [14, 12] [Dimension of ecg fearures, lengthP]
        ecg0 = ecg_subcase.loc[j: j + lenP - 1, ['RR_I', 'QRS', 'QTC', 'SDNN', 'SDSD', 'RMSSD',
                                                 'VeryLowFrequencyPower', 'LowFrequencyPower',
                                                 'HFFrequencyPower', 'VeryHighFrequencyPower',
                                                 'Sympathetic',
                                                 'Vagal', 'SympatheticToVagalRatio', 'RSA']]
        ecg0 = ecg0.values.transpose(1, 0)

        # ! label Y: shape [3, 4] [Dimension of one-hot category, lengthQ]
        label0 = ecg_subcase.loc[j + lenP + step: j + lenP + step + lenQ - 1,
                 ['label_0.0', 'label_1.0']].values.transpose(1, 0)
        label0_1d = ecg_subcase.loc[j + lenP + step: j + lenP + step + lenQ - 1, 'label'].values

        cuts = ecg_subcase.loc[j: j + lenP - 1, 'cuts'].values - 1

        # ! fnirsx X node vector: shape [6, 8, 12] [Dimension of fnirs alff features, node numbers, lengthP]
        # ! fnirsA A adj matrix: shape [8, 8, 12] [node numbers, node numbers, lengthP]
        fnirsx, fnirsA1, fnirsA2, fnirsA3, fnirsA4 = [], [], [], [], []
        for cutnum in cuts:
            try:
                if len(fnirs[cutnum]['ALFF']) == 8:  # 如果fnirs为空,则舍弃后面的
                    fnirs_nodevec = np.stack((fnirs[cutnum]['ALFF'], fnirs[cutnum]['mALFF'], fnirs[cutnum]['zALFF'],
                                              fnirs[cutnum]['fALFF'], fnirs[cutnum]['mfALFF'], fnirs[cutnum]['zfALFF']),
                                             axis=1)
                    fnirsx.append(np.squeeze(fnirs_nodevec))
                    fnirsA1.append(fnirs[cutnum]['corr'])
                    fnirsA2.append(fnirs[cutnum]['corrz'])
                    fnirsA3.append(fnirs[cutnum]['coh'])
                    fnirsA4.append(fnirs[cutnum]['plv'])
                else:
                    fnirsx, fnirsA1, fnirsA2, fnirsA3, fnirsA4 = [], [], [], [], []
                    break
            except Exception as e:
                # print('error1: ', e, ': ', subnum, casenum, j, cutnum)
                # print(np.array(fnirsx).shape)
                break
        try:
            fnirsx = np.array(fnirsx).transpose(2, 1, 0)
            fnirsA1 = np.array(fnirsA1).transpose(1, 2, 0)
            fnirsA2 = np.array(fnirsA2).transpose(1, 2, 0)
            fnirsA3 = np.array(fnirsA3).transpose(1, 2, 0)
            fnirsA4 = np.array(fnirsA4).transpose(1, 2, 0)

            labels.append(label0)
            labels_1d.append(label0_1d)
            ecgs.append(ecg0)
            fnirsxs.append(fnirsx)
            fnirsA1s.append(fnirsA1)
            fnirsA2s.append(fnirsA2)
            fnirsA3s.append(fnirsA3)
            fnirsA4s.append(fnirsA4)
        except Exception as e:
            # print('error2: ', e, ': ', subnum, casenum, j)
            break

    PhyDataDict_subcase = dict()
    PhyDataDict_subcase['labels'] = labels
    PhyDataDict_subcase['labels_1d'] = labels_1d
    PhyDataDict_subcase['ecgs'] = ecgs
    PhyDataDict_subcase['fnirsxs'] = fnirsxs
    PhyDataDict_subcase['fnirsA1s'] = fnirsA1s
    PhyDataDict_subcase['fnirsA2s'] = fnirsA2s
    PhyDataDict_subcase['fnirsA3s'] = fnirsA3s
    PhyDataDict_subcase['fnirsA4s'] = fnirsA4s

    return PhyDataDict_subcase

def concatDataDict(PhyDataDict0, PhyDataDict_sub):
    if len(PhyDataDict0) == 0:
        PhyDataDict0 = PhyDataDict_sub
    else:
        for key in PhyDataDict0.keys():
            PhyDataDict0[key] = PhyDataDict0[key] + PhyDataDict_sub[key]
    return PhyDataDict0


def construct_dataset(ecg, fnirs, lenP, lenQ, step, SubTestFlag, split_ratio_train, kfold, sub):
    subs = set(ecg['sub'])
    cases = set(ecg['case'])
    Dataset_norm_kfold, PhyDataDict_kfold = {}, {}
    sublist, caselist, sclist = [], [], []
    i = 0
    for subnum in subs:
        for casenum in cases:
            i += 1
            sublist.append(subnum)
            caselist.append(casenum)
            sclist.append(i)
    df_subcase = pd.DataFrame({"sub": sublist, "case": caselist, "scnum": sclist})
    if SubTestFlag == 'dep':  # scnum 随机kfold划分为 训练集 测试集 验证集
        kfold_ind = split_number_set_kfold_dep(set(sclist), (split_ratio_train, 1 - split_ratio_train), kfold)
    elif SubTestFlag == 'ind':  # scnum 先随机划分 训练被试集 测试被试集， 训练被试集内scnum再kfold划分 训练集与验证集
        kfold_ind = split_number_set_kfold_ind(df_subcase, (split_ratio_train, 1 - split_ratio_train), kfold, sub)
    elif SubTestFlag == 'ccs':  # scnum 先随机划分 训练工况集，测试工况集，训练工况集内scnum再kfold划分 训练集与验证集
        kfold_ind = split_number_set_kfold_ccs(df_subcase, (split_ratio_train, 1 - split_ratio_train), kfold, sub)

    for kfold_num in kfold_ind.keys():
        PhyDataDict_train, PhyDataDict_valid, PhyDataDict_test = {}, {}, {}
        ind_train_list = kfold_ind[kfold_num]['train']
        ind_valid_list = kfold_ind[kfold_num]['valid']
        ind_test_list = kfold_ind[kfold_num]['test']
        for subnum in subs:
            for casenum in cases:
                scnum = df_subcase[(df_subcase['sub'] == subnum) & (df_subcase['case'] == casenum)].scnum.values[0]
                PhyDataDict_subcase = construct_dataset_subcase(subnum, casenum, ecg, fnirs, lenP, lenQ, step)
                if scnum in ind_train_list:
                    PhyDataDict_train = concatDataDict(PhyDataDict_train, PhyDataDict_subcase)
                elif scnum in ind_valid_list:
                    PhyDataDict_valid = concatDataDict(PhyDataDict_valid, PhyDataDict_subcase)
                elif scnum in ind_test_list:
                    PhyDataDict_test = concatDataDict(PhyDataDict_test, PhyDataDict_subcase)
        PhyDataDict_kfold[kfold_num] = {'train': PhyDataDict_train,
                                        'valid': PhyDataDict_valid,
                                        'test': PhyDataDict_test}

    # 标准化
    trainDict = PhyDataDict_kfold[0]['train'].copy()
    validDict = PhyDataDict_kfold[0]['valid'].copy()
    PhyDatadict_train_all = concatDataDict(trainDict, validDict)
    PhyDataset_train_all = PhyDataset(PhyDatadict_train_all, False, 1, 1, 1, 1)
    meanEcg, stdEcg, meanFnirs, stdFnirs = cal_mean_std(PhyDataset_train_all)
    for kfold_num in kfold_ind.keys():
        Dataset_norm_kfold[kfold_num] = {'train': PhyDataset(PhyDataDict_kfold[kfold_num]['train'], True, meanEcg, stdEcg, meanFnirs, stdFnirs),
                                        'valid': PhyDataset(PhyDataDict_kfold[kfold_num]['valid'], True, meanEcg, stdEcg, meanFnirs, stdFnirs),
                                        'test': PhyDataset(PhyDataDict_kfold[kfold_num]['test'], True, meanEcg, stdEcg, meanFnirs, stdFnirs)}

    return Dataset_norm_kfold


def split_number_set(numset, split_ratio, sub=0):
    numarray = list(numset)[:]
    random.shuffle(numarray)
    split_index_start = math.ceil(len(numarray) * split_ratio[1]) * sub
    split_index_end = math.ceil(len(numarray) * split_ratio[1]) * (sub+1)
    return numarray[:split_index_start]+numarray[split_index_end:], numarray[split_index_start:split_index_end]

def split_number_set_kfold_dep(numset, split_ratio, kfold):
    kfold_ind = {}
    numarray = list(numset)[:]
    random.shuffle(numarray)
    split_index = math.ceil(len(numarray) * split_ratio[0])
    ind_trainall = numarray[:split_index]
    ind_test = numarray[split_index:]
    kfold_index = round(split_index / kfold)
    for i in range(kfold):
        ind_valid = ind_trainall[kfold_index * i: kfold_index * (i + 1)]
        ind_train = ind_trainall[0: kfold_index * i] + ind_trainall[kfold_index * (i + 1):]
        kfold_ind[i] = {'train': ind_train, 'valid': ind_valid, 'test': ind_test}
    return kfold_ind

def split_number_set_kfold_ind(df_subcase, split_ratio, kfold, sub):
    # 依次用 75%的被试训练，25%的被试测试
    split_ratio = (0.75, 0.25)
    kfold_ind = {}
    subset = set(df_subcase['sub'])
    train_sub, test_sub = split_number_set(subset, split_ratio, sub)
    df_sc_train = df_subcase[df_subcase['sub'].isin(train_sub)]
    df_sc_test = df_subcase[df_subcase['sub'].isin(test_sub)]
    sc_train = list(set(df_sc_train['scnum']))[:]
    sc_test = list(set(df_sc_test['scnum']))[:]
    random.shuffle(sc_train)
    kfold_index = round(len(sc_train) / kfold)
    for i in range(kfold):
        ind_valid = sc_train[kfold_index * i: kfold_index * (i + 1)]
        ind_train = sc_train[0: kfold_index * i] + sc_train[kfold_index * (i + 1):]
        kfold_ind[i] = {'train': ind_train, 'valid': ind_valid, 'test': sc_test}
    return kfold_ind

def split_number_set_kfold_ccs(df_subcase, split_ratio, kfold, case):
    # 依次用3个case训练，1个case测试
    kfold_ind = {}
    df_sc_train = df_subcase[df_subcase['case'] != case]
    df_sc_test = df_subcase[df_subcase['case'] == case]
    sc_train = list(set(df_sc_train['scnum']))[:]
    sc_test = list(set(df_sc_test['scnum']))[:]
    random.shuffle(sc_train)
    kfold_index = round(len(sc_train) / kfold)
    for i in range(kfold):
        ind_valid = sc_train[kfold_index * i: kfold_index * (i + 1)]
        ind_train = sc_train[0: kfold_index * i] + sc_train[kfold_index * (i + 1):]
        kfold_ind[i] = {'train': ind_train, 'valid': ind_valid, 'test': sc_test}
    return kfold_ind

def split_dataset(dataset, split_radio):
    train_size = int(len(dataset) * split_radio[0])
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(0))
    return train_dataset, test_dataset

def cal_mean_std(trainDataset):
    """
    计算训练集上 ecg 6个手工特征 和 fnirs 6个alff特征在8个通道上的 平均值 标准差
    :param trainDataset: 训练集
    :return:
        meanEcg, stdEcg: [14, ] 14-dimension of ecg handcraft features
        meanFnirs, stdFnirs [6, 8] 6-dimension of alff features 8-fnirs node numbers
    """
    ecg_trainset = []
    fnirs_trainset = []
    for i, (label, label_1d, ecg, fnirsx, fnirsA1, fnirsA2, fnirsA3, fnirsA4) in enumerate(trainDataset):
        ecg_trainset.append(ecg)
        fnirs_trainset.append(fnirsx)
    ecg_trainset = np.array(ecg_trainset).transpose(0,2,1).reshape(-1, 14)  # 沿着时间列拼接 [ele num, dimension of ecg features, lenP]
    fnirs_trainset = np.array(fnirs_trainset).transpose(0,3,1,2).reshape(-1, 6, 8)  # [ele num, Dimension of fnirs alff features, node numbers, lenP]
    meanEcg = ecg_trainset.mean(axis=0)
    stdEcg = ecg_trainset.std(axis=0)
    meanFnirs = fnirs_trainset.mean(axis=0)
    stdFnirs = fnirs_trainset.std(axis=0)
    return meanEcg, stdEcg, meanFnirs, stdFnirs


def load_dataset(args):
    if args.noise == '-1' and args.dropCH == 0 and args.dropR == 0:
        if args.predata == 'pre':
            raw_data_dict = load_pkl(os.path.join(args.data_dir, 'raw_data_dict_pre.pkl'))
        elif args.predata == 'none':
            raw_data_dict = load_pkl(os.path.join(args.data_dir, 'raw_data_dict_pre_none.pkl'))
    elif args.noise != '-1' and args.dropCH == 0 and args.dropR == 0:
        dataset_path = os.path.join(args.data_dir, 'data_noise_drop', 'noise', 'noise_' + str(args.noise))
        raw_data_dict = load_pkl(os.path.join(dataset_path, 'raw_data_dict_pre.pkl'))
    elif args.noise == '-1' and args.dropCH != 0 and args.dropR != 0:
        dataset_path = os.path.join(args.data_dir, 'data_noise_drop', 'drop', 'drop_' + str(args.dropCH), 'd' + str(args.dropR))
        raw_data_dict = load_pkl(os.path.join(dataset_path, 'raw_data_dict_pre.pkl'))
    if args.SubTestFlag == 'ind' or args.SubTestFlag == 'dep' or args.SubTestFlag == 'ccs':
        PhyDataset_norm = {}
        PhyDataset_norm['kfold'] = {}
        Dataset_norm_kfold1 = construct_dataset(raw_data_dict['ecg1_pre'], raw_data_dict['fnirs1_pre'], args.lenP,
                                                    args.lenQ, args.step, args.SubTestFlag, args.split_ratio_train,
                                                    args.fold_num, args.sub_group)
        Dataset_norm_kfold2 = construct_dataset(raw_data_dict['ecg2_pre'], raw_data_dict['fnirs2_pre'], args.lenP,
                                                    args.lenQ, args.step, args.SubTestFlag, args.split_ratio_train,
                                                    args.fold_num, args.sub_group)
        for kfold_num in range(args.fold_num):
            PhyDataset_norm['kfold'][kfold_num] = {'train': ConcatDataset([Dataset_norm_kfold1[kfold_num]['train'], Dataset_norm_kfold2[kfold_num]['train']]),
                                                'valid': ConcatDataset([Dataset_norm_kfold1[kfold_num]['valid'], Dataset_norm_kfold2[kfold_num]['valid']])}
        PhyDataset_norm['train'] = ConcatDataset([Dataset_norm_kfold1[0]['train'], Dataset_norm_kfold2[0]['train'], Dataset_norm_kfold1[0]['valid'], Dataset_norm_kfold2[0]['valid']])
        PhyDataset_norm['test'] = ConcatDataset([Dataset_norm_kfold1[0]['test'], Dataset_norm_kfold2[0]['test']])

    return PhyDataset_norm


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data/", help="Data directory.")
    parser.add_argument("--lenP", type=int, default=6, help="seq_length_x", )
    parser.add_argument("--lenQ", type=int, default=4, help="seq_length_y", )
    parser.add_argument("--step", type=int, default=0, help="step")
    parser.add_argument("--split_ratio_train", type=float, default=0.8, help="split ratio")
    parser.add_argument("--topkFlag", type=bool, default=True, help="topkFlag")
    parser.add_argument("--SubTestFlag", type=str, default='dep', help="dep ind dep2 ind2")
    parser.add_argument("-fold_num", type=int, default=5, help="fold_num")
    parser.add_argument("--predata", default='pre', type=str, help="predata")

    args = parser.parse_args()

    Dataset_norm_all = load_dataset(args)
    trainDataset_norm_all, testDataset_norm_all = Dataset_norm_all['train'], Dataset_norm_all['test']

    print('---### Dataset loaded ###---')
    print('train dataset length: ', len(trainDataset_norm_all), '; test dataset length: ', len(testDataset_norm_all))

