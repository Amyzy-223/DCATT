from torch.utils.data import Dataset
import numpy as np



class EEGDataset(Dataset):
    def __init__(self, clip_datas, sample_rate=200):
        self.datas = clip_datas['datas']
        self.labels = clip_datas['labels']
        self.info = clip_datas['info']
        self.sample_rate = sample_rate
        self.num_class = len(set(self.info['emotion']))
        self.bdes, self.corrs, self.plvs = cal_features(self.datas, self.sample_rate)
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

