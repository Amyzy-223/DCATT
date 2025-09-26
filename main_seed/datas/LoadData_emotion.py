import os
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from torcheeg.datasets import SEEDDataset
from torcheeg import transforms
from torcheeg.transforms.graph import ToG
from torcheeg.datasets.constants import SEED_STANDARD_ADJACENCY_MATRIX


seed_channel_list = ['FP1','FPZ','FP2','AF3','AF4','F7','F5','F3','F1','FZ','F2','F4','F6','F8','FT7','FC5','FC3','FC1',
                     'FCZ','FC2','FC4','FC6','FT8','T7','C5','C3','C1','CZ','C2','C4','C6','T8','TP7','CP5','CP3','CP1',
                     'CPZ','CP2','CP4','CP6','TP8','P7','P5','P3','P1','PZ','P2','P4','P6','P8','PO7','PO5','PO3','POZ',
                     'PO4','PO6','PO8','CB1','O1','OZ','O2','CB2']
target_12_channels = ['FT7', 'FT8', 'T7', 'T8', 'C5', 'C6', 'TP7', 'TP8', 'CP5', 'CP6', 'P7', 'P8']

def save_pkl(saveDict, saveName):
    f_save = open(saveName, 'wb')
    pickle.dump(saveDict, f_save)
    f_save.close()

def load_pkl(saveName):
    f_read = open(saveName, 'rb')
    saveDict = pickle.load(f_read)
    f_read.close()
    return saveDict

def get_save_data(dataset):
    datas = []
    labels = []
    info = dataset.info
    for i in range(len(dataset)):
        datas.append(dataset[i][0])
        labels.append(dataset[i][1])
    datas = np.array(datas)
    labels = np.array(labels)
    return datas, labels, info

def prepare_raw_clip_data(root_path, select_channel_flag=False):
    if select_channel_flag:
        dataset = SEEDDataset(root_path=root_path,
                              chunk_size=200*4,  # 4s片段
                              online_transform=transforms.Compose([
                                  transforms.RearrangeElectrode(source=seed_channel_list, target = target_12_channels),
                                  transforms.ToTensor(),
                                  transforms.To2d()
                              ]),
                              label_transform=transforms.Compose([
                                  transforms.Select('emotion'),
                                  transforms.Lambda(lambda x: x + 1)
                              ]))
    else:
        dataset = SEEDDataset(root_path=root_path,
                              chunk_size=200 * 4,  # 4s片段
                              online_transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.To2d()
                              ]),
                              label_transform=transforms.Compose([
                                  transforms.Select('emotion'),
                                  transforms.Lambda(lambda x: x + 1)
                              ]))
    datas, labels, info = get_save_data(dataset)
    seed_clip_datas = {'datas': datas, 'labels': labels, 'info': info}
    return seed_clip_datas

def graph_norm(adj, topk=5):
    adj = abs(adj)
    topk_idx = (-adj).argsort(axis=-1)[:, :topk]
    mask = np.eye(adj.shape[0], dtype=bool)
    for i in range(0, topk_idx.shape[0]):
        for j in range(0, topk_idx.shape[1]):
            mask[i, topk_idx[i, j]] = 1
            mask[topk_idx[i, j], i] = 1
    adj = mask * adj

    ones = np.ones(adj.shape[0])
    D = np.diag(adj @ ones)
    Dinv = np.linalg.inv(D)
    Dinv5 = np.sqrt(Dinv)
    adj_stand = Dinv5 @ adj @ Dinv5
    return adj_stand

def cal_features(datas, sample_rate, normFlag=True):
    """
    计算微分熵、相关性、锁相值
    Parameters
    ----------
    data1： [288, 22, 1125] [trial, channel, time_sample]
    data2
    data3
    data4
    sample_rate

    Returns：
        feature [288, 4, 22, 4] [trial, bands, channel, time_step]
    -------

    """
    data1 = datas[:, :, :, 0:200].squeeze()
    data2 = datas[:, :, :, 200:400].squeeze()
    data3 = datas[:, :, :, 400:600].squeeze()
    data4 = datas[:, :, :, 600:800].squeeze()
    datas = [data1, data2, data3, data4]
    BDE = transforms.BandDifferentialEntropy(sampling_rate=sample_rate,
                                             band_dict={'delta': (1, 4),
                                                        'theta': (4, 8),
                                                        'alpha': (8, 14),
                                                        'beta': (14, 31),
                                                        'gamma': (31, 49)})
    CORR = transforms.PearsonCorrelation(absolute=True)
    PLV = transforms.PhaseLockingCorrelation()
    feature0 = BDE(eeg=datas[0][0, :, :])
    features_BDE, features_CORR, features_PLV = [], [], []
    for data in datas:
        print('calculating feature')
        feature_BDE = np.array([BDE(eeg=trail_data)['eeg'] for trail_data in data])  # [228, 22, 4] [trial, channel, bands]
        print('--bde')
        if normFlag:
            feature_CORR = np.array([graph_norm(CORR(eeg=trail_data)['eeg'].squeeze()) for trail_data in data])  # [228, 22, 22] [trial, channel, channel]
        else:
            feature_CORR = np.array([CORR(eeg=trail_data)['eeg'].squeeze() for trail_data in data])  # [228, 22, 22] [trial, channel, channel]
        print('--corr')
        if normFlag:
            feature_PLV = np.array([graph_norm(PLV(eeg=trail_data)['eeg'].squeeze()) for trail_data in data])  # [228, 22, 22] [trial, channel, channel]
        else:
            feature_PLV = np.array([PLV(eeg=trail_data)['eeg'].squeeze() for trail_data in data])  # [228, 22, 22] [trial, channel, channel]
        print('--plv')
        features_BDE.append(feature_BDE)
        features_CORR.append(feature_CORR)
        features_PLV.append(feature_PLV)
    features_BDE = np.array(features_BDE).transpose(1, 3, 2, 0)  # [trial, bands, channel, time_step]
    features_CORR = np.array(features_CORR).transpose(1, 2, 3, 0)  # [trial, channel, channel, time_step]
    features_PLV = np.array(features_PLV).transpose(1, 2, 3, 0)  # [trial, channel, channel, time_step]
    return features_BDE, features_CORR, features_PLV

def cal_psd_feature(datas, sample_rate):
    data_seq = []
    seq_num = 10
    seq_len = 2 # 2s
    for i in range(seq_num):
        start_at = seq_len*i*sample_rate
        end_at = (seq_len*(i+1))*sample_rate
        data = datas[:, :, :, start_at: end_at]
        data_seq.append(data)
    PSD = transforms.BandPowerSpectralDensity(sampling_rate=sample_rate, band_dict={'delta': (1, 4), 'theta': (4, 8),
                                                                                    'alpha': (8, 12), 'low beta': (12, 16),
                                                                                    'beta': (16, 20), 'high beta': (20, 28),
                                                                                    'gamma': (30, 45)})
    features_PSD = []
    for data in data_seq:
        print('calculating feature')
        feature_PSD = np.array([PSD(eeg=trial_data.squeeze())['eeg'].squeeze() for trial_data in data])
        features_PSD.append(feature_PSD)
    features_PSD = np.array(features_PSD).transpose(1, 3, 2, 0) # [trial, bands, channel, time_step]
    PSD_sum = features_PSD.sum(axis=1, keepdims=True)
    features_rPSD = features_PSD / PSD_sum
    return features_rPSD

def cal_psd_feature_overlap(datas, sample_rate):
    data_seq = []
    seq_num = 36
    seq_len = 2 # 2s
    overlap_rate = 0.75
    overlap = seq_len * (1-overlap_rate) # 0.5s
    for i in range(seq_num):
        start_at = int(i * overlap * sample_rate)
        end_at = int(start_at + seq_len * sample_rate)
        data = datas[:, :, :, start_at: end_at]
        data_seq.append(data)
    PSD = transforms.BandPowerSpectralDensity(sampling_rate=sample_rate, band_dict={'delta': (1, 4), 'theta': (4, 8),
                                                                                    'alpha': (8, 12), 'low beta': (12, 16),
                                                                                    'beta': (16, 20), 'high beta': (20, 28),
                                                                                    'gamma': (30, 45)})
    features_PSD = []
    for data in data_seq:
        print('calculating feature')
        feature_PSD = np.array([PSD(eeg=trial_data.squeeze())['eeg'].squeeze() for trial_data in data])
        features_PSD.append(feature_PSD)
    features_PSD = np.array(features_PSD).transpose(1, 3, 2, 0) # [trial, bands, channel, time_step]
    PSD_sum = features_PSD.sum(axis=1, keepdims=True)
    features_rPSD = features_PSD / PSD_sum
    return features_rPSD

class EEGDataset(Dataset):
    def __init__(self, clip_datas, sample_rate=200, normFlag=True):
        self.datas = clip_datas['datas']
        self.labels = clip_datas['labels']
        self.info = clip_datas['info']
        self.sample_rate = sample_rate
        self.num_class = len(set(self.info['emotion']))
        self.bdes, self.corrs, self.plvs = cal_features(self.datas, self.sample_rate, normFlag)
        # print('feature cal ok')

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        ## TODO 分4个timestep 计算特征上移，不要每次取值时重复计算
        # x = self.BDE_func(eeg=raw_data)['eeg'].squeeze()
        # corr = self.CORR_func(eeg=raw_data)['eeg'].squeeze()
        # plv = self.PLV_func(eeg=raw_data)['eeg'].squeeze()
        x = self.bdes[index, :, :, :]
        corr = self.corrs[index, :, :, :]
        plv = self.plvs[index, :, :, :]
        y = self.labels[index]
        y_onehot = np.eye(self.num_class)[y]
        return x, corr, plv, y, y_onehot

class EEGDataset_e2e(Dataset):
    def __init__(self, clip_datas, sample_rate=200):
        self.datas = clip_datas['datas']
        self.labels = clip_datas['labels']
        self.info = clip_datas['info']
        self.sample_rate = sample_rate
        self.num_class = len(set(self.info['emotion']))

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        x = self.datas[index, :, :, :].squeeze()
        y = self.labels[index]
        y_onehot = np.eye(self.num_class)[y]
        return x, y, y_onehot

class EEGDataset_EmT(Dataset):
    def __init__(self, clip_datas, sample_rate=200):
        self.datas = clip_datas['datas']
        self.labels = clip_datas['labels']
        self.info = clip_datas['info']
        self.sample_rate = sample_rate
        self.num_class = len(set(self.info['emotion']))
        # self.psds = cal_psd_feature(self.datas, self.sample_rate)
        self.psds = cal_psd_feature_overlap(self.datas, self.sample_rate)
        # print('feature cal ok')

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        x = self.psds[index, :, :, :]
        y = self.labels[index]
        y_onehot = np.eye(self.num_class)[y]
        return x, y, y_onehot

def generate_train_test_idx(df, mode, subs):
    df['session_id'] = 0
    df = df[(df['subject_id'] >= 1) & (df['subject_id'] <= subs)]
    df['trial_id_int'] = df['trial_id'].str.extract('(\d+)')[0].astype(int)

    for sub in range(subs):
        dates = sorted(list(set(df[df['subject_id']==(sub+1)]['date'])))
        for i, date in enumerate(dates):
            df.loc[(df['subject_id']==(sub+1)) & (df['date']==date), 'session_id'] = i
    idx_dict = dict()
    if mode == 'LOSO':  # 跨被试，被试留一做
        for sub in range(subs):
            sub_id = sub + 1
            test_idx = df.index[df['subject_id'] == sub_id].tolist()
            train_idx = df.index[df['subject_id'] != sub_id].tolist()
            idx_dict[sub] = {'train_idx': train_idx, 'test_idx': test_idx}
    elif mode == 'DEP':  # 跨session 所有被试第1、2session做train，valid 最后一个session做test
        train_idx = df.index[(df['session_id']==0) | (df['session_id']==1)].tolist()
        test_idx = df.index[(df['session_id']==2)].tolist()
        idx_dict[0] = {'train_idx': train_idx, 'test_idx': test_idx}
    elif mode == 'TRA': # 所有被试所有数据放在一起随机划分 train test valid
        train_idx = df.sample(frac=0.8, random_state=3407).index
        test_idx = df.index.difference(train_idx)
        idx_dict[0] = {'train_idx': train_idx, 'test_idx': test_idx}
    elif mode == 'INS':  # 所有被试内，前9个trial做train valid 后6个trial做test
        for sub in range(subs):
            sub_id = sub + 1
            train_idx = df.index[(df['subject_id']==sub_id) & (df['trial_id_int']>=1) & (df['trial_id_int']<=9)]
            test_idx = df.index[(df['subject_id']==sub_id) & (df['trial_id_int']>=10) & (df['trial_id_int']<=15)]
            idx_dict[sub] = {'train_idx': train_idx, 'test_idx': test_idx}
    return idx_dict

def generate_train_test_set(dataset, idx_dict, mode, name, save_dir, save_flag=False):
    for sub, idxs in idx_dict.items():
        train_idx, test_idx = idxs['train_idx'], idxs['test_idx']
        trainset = Subset(dataset, train_idx)
        testset = Subset(dataset, test_idx)
        PhyDataset = {'train': trainset, 'test': testset}
        if save_flag:
            if mode == 'LOSO':
                save_name = os.path.join(save_dir, name+'_LOSO_sub'+str(sub)+'.pkl')
                save_pkl(PhyDataset, save_name)
            elif mode == 'DEP':
                save_name = os.path.join(save_dir, name+'_DEP.pkl')
                save_pkl(PhyDataset, save_name)
            return 0
        else:
            return PhyDataset



def get_data(seed_clip_datas, seed_dataset, mode, name, save_dir, subs):
    save_dir = os.path.join(save_dir, mode)
    info = seed_clip_datas['info']
    idx_dict = generate_train_test_idx(info, mode, subs=subs)
    generate_train_test_set(seed_dataset, idx_dict, mode, name, save_dir, save_flag=True)

def save_dataset(seed_clip_datas, save_dir, save_name):
    seedDataset = EEGDataset(seed_clip_datas, sample_rate=200)
    save_pkl({'seedDataset': seedDataset}, os.path.join(save_dir, save_name+'.pkl'))

def save_train_test_idx(info, subs, save_dir):
    modes = ['LOSO', 'DEP', 'TRA', 'INS']
    # modes = ['INS']
    for mode in modes:
        idx_dict = generate_train_test_idx(info, mode=mode, subs=subs)
        save_pkl({'info': info, 'mode': mode, 'idx_dict': idx_dict}, os.path.join(save_dir, mode+'.pkl'))


if __name__ == '__main__':

    # 利用torcheeg读取原始eeg时序信号
    seed_clip_datas = prepare_raw_clip_data(root_path='./rawdata/Preprocessed_EEG', select_channel_flag=True)
    save_pkl(seed_clip_datas, './loaddata/SeedClip_12chan.pkl')


    # SEED_DCATT
    seed_clip_datas = load_pkl('./loaddata/SeedClip_12chan.pkl')
    seedDataset = EEGDataset(seed_clip_datas, sample_rate=200)
    save_pkl({'seedDataset': seedDataset}, './loaddata/SEED_DCATT/seedDataset.pkl')
    seed_clip_datas = load_pkl('./loaddata/SeedClip_12chan.pkl')
    save_train_test_idx(info=seed_clip_datas['info'], subs=15, save_dir='./loaddata/SEED_DCATT')






    
