import copy
import json
import os
import pickle
import random

import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops
from torch.utils.data import Dataset, DataLoader, Subset, SubsetRandomSampler
from sklearn.manifold import TSNE
from sklearn import metrics

from fontTools.misc.cython import returns
from matplotlib import pyplot as plt, image as mpimg
from sklearn.metrics import roc_curve, auc, roc_auc_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, \
    classification_report, brier_score_loss
from scipy import interp
from scipy.special import softmax
from itertools import cycle




def save_pkl(saveDict, saveName):
    f_save = open(saveName, 'wb')
    pickle.dump(saveDict, f_save)
    f_save.close()

def load_pkl(saveName):
    f_read = open(saveName, 'rb')
    saveDict = pickle.load(f_read)
    f_read.close()
    return saveDict

def save_args(args, save_dir):
    """
    保存 args 字典到txt
    :param args:
    :param save_dir:
    :return:
    """
    argsDict = args.__dict__
    with open(os.path.join(save_dir, 'args.txt'), 'w') as f:
        for eachArg, value in argsDict.items():
            f.write(f'{eachArg}: {value}\n')

def setup_seed(seed):
    """
    固定随机数种子
    :param seed:
    :return:
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def cal_roc_stand(trues, scores):
    """
    计算给定一维 vector的auc
    :param trues:
    :param scores:
    :return:
    """
    num_class = 2
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    trues = torch.from_numpy(trues).to(torch.int64)
    trues_ones = F.one_hot(trues, num_classes=num_class)

    for i in range(num_class):
        fpr[i], tpr[i], _ = roc_curve(trues_ones[:, i], scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(trues_ones.ravel(), scores.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_class)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_class):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= num_class
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    return fpr, tpr, roc_auc

def cal_roc(labellist, scorelist, predlist):
    """
    分别计算 所有lenQ的auc，和每一个step的auc
    :param labellist: array [batch_size, lenQ]
    :param scorelist: array [batch_size, cate_nums, lenQ]
    :param predlist: array [batch_size, lenQ]
    :return:
        auc_list: list lenQ+1, [all, each step]
        f1_list: list lenQ+1, [all, each step]
    """
    auc_list, f1_list = [], []
    labellist_allsteps = labellist.reshape(-1)
    scorelist_allsteps = scorelist.transpose(0, 2, 1).reshape(-1, 2)
    predlist_allsteps = predlist.reshape(-1)
    # print(scorelist_allsteps)
    # print(labellist_allsteps)
    fpr, tpr, roc_auc = cal_roc_stand(labellist_allsteps, scorelist_allsteps)
    f1 = f1_score(labellist_allsteps, predlist_allsteps, average='macro')
    auc_list.append(roc_auc)
    f1_list.append(f1)
    for q in range(labellist.shape[-1]):
        fpr, tpr, roc_auc = cal_roc_stand(labellist[:, q].squeeze(), scorelist[:, :, q].squeeze())
        auc_list.append(roc_auc)
        f1 = f1_score(labellist[:, q].squeeze(), predlist[:, q].squeeze(), average='macro')
        f1_list.append(f1)
    return auc_list, f1_list

def cal_multistep(labellist, scorelist, predlist):
    """
    分别计算 所有lenQ的PR-AUC, Brier score和每一个step的
    """
    prauc_list, brier_list  = [], []
    labellist_allsteps = labellist.reshape(-1)
    scorelist_allsteps = scorelist.transpose(0, 2, 1).reshape(-1, 2)
    predlist_allsteps = predlist.reshape(-1)

    # 计算 Precision-Recall曲线下的面积
    precision, recall, _ = metrics.precision_recall_curve(labellist_allsteps, scorelist_allsteps[:, 1])
    auc_pr = metrics.auc(recall, precision)
    prauc_list.append(auc_pr)

    # 计算 brier分数
    probs = softmax(scorelist_allsteps, axis=1)  # score转为分类概率 # 将0作为正类，计算分数
    brier = brier_score_loss(labellist_allsteps, probs[:, 1])  # 将1作为正类，计算分数
    brier_list.append(brier)

    for q in range(labellist.shape[-1]):
        # 计算 Precision-Recall曲线下的面积
        precision, recall, _ = metrics.precision_recall_curve(labellist[:, q].squeeze(), scorelist[:, 1, q].squeeze())
        auc_pr = metrics.auc(recall, precision)
        prauc_list.append(auc_pr)
        # 计算 brier分数
        probs = softmax(scorelist[:, :, q].squeeze(), axis=1)
        brier = brier_score_loss(labellist[:, q].squeeze(), probs[:, 1])
        brier_list.append(brier)
    return prauc_list, brier_list




def cal_steps_loss(outputs, label, criterion):
    """
    计算多步预测损失，测试发现循环增加loss和分别计算loss相加没区别
    :param outputs: [batch_size, out_dim, lenQ]
    :param label: [batch_size, out_dim, lenQ]
    :param criterion: cross entropy loss
    :return:
    """
    loss_steps = []
    loss = 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for q in range(outputs.shape[-1]):
        if criterion[0] == 'entropy':
            criterion_func = nn.CrossEntropyLoss()
            lossq = criterion_func(outputs[:, :, q].squeeze(), label[:, :, q].squeeze())
        elif criterion[0] == 'focal':
            lossq = torchvision.ops.sigmoid_focal_loss(outputs[:, :, q].squeeze(), label[:, :, q].squeeze(), alpha=0.75, gamma=0, reduction='mean')
            # print(lossq)
        elif criterion[0] == 'weight':
            # alpha越大，表明样本越不均衡，越关注小样本的损失，压低大样本的影响，2.71是根据实际负正样本数计算得出
            alpha = criterion[1]
            w1 = 1/(alpha/(alpha+1)*2)
            w2 = 1/(1/(alpha+1)*2)
            # weight = torch.tensor([0.685, 1.853]).to(device)
            weight = torch.tensor([w1, w2]).to(device)
            criterion_func = nn.BCEWithLogitsLoss(weight=weight)
            lossq = criterion_func(outputs[:, :, q].squeeze(), label[:, :, q].squeeze())
        elif criterion[0] == 'weight2':
            alpha = criterion[1]
            pos_weight = torch.tensor([1/alpha, alpha]).to(device)
            criterion_func = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            lossq = criterion_func(outputs[:, :, q].squeeze(), label[:, :, q].squeeze())
        elif criterion[0] == 'weight3':
            alpha = criterion[1]
            pos_weight = torch.tensor([alpha]).to(device)
            criterion_func = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            out_vec = outputs[:, -1, q].squeeze()
            lab_vec = label[:, -1, q].squeeze()
            lossq = criterion_func(outputs[:, -1, q].squeeze(), label[:, -1, q].squeeze())
        elif criterion[0] == 'weight4':
            alpha = criterion[1]
            weight = torch.tensor([1, alpha]).to(device)
            criterion_func = nn.CrossEntropyLoss(weight=weight)
            lossq = criterion_func(outputs[:, :, q].squeeze(), label[:, :, q].squeeze())
        loss_steps.append(lossq)
        loss += lossq
    # print('loss:', loss)
    return loss_steps, loss

'''
def cal_steps_loss(outputs, label, criterion):
    loss_steps = []
    loss = 0
    lossq1 = criterion(outputs[:, :, 0].squeeze(), label[:, :, 0].squeeze())
    lossq2 = criterion(outputs[:, :, 1].squeeze(), label[:, :, 1].squeeze())
    lossq3 = criterion(outputs[:, :, 2].squeeze(), label[:, :, 2].squeeze())
    lossq4 = criterion(outputs[:, :, 3].squeeze(), label[:, :, 3].squeeze())
    loss = lossq1 + lossq2 + lossq3 + lossq4
    loss_steps = [lossq1, lossq2, lossq3, lossq4]
    return loss_steps, loss
'''

def dictlistlist2df(dictlistlist, name):
    df0 = pd.DataFrame()
    for epoch in range(len(dictlistlist)):
        df_epoch = pd.DataFrame()
        for q in range(len(dictlistlist[epoch])):
            for key in dictlistlist[epoch][q].keys():
                col_name = name + '_' + str(q) + '_' + str(key)
                df_epoch.loc[epoch, col_name] = dictlistlist[epoch][q][key]
        df0 = pd.concat([df0, df_epoch])
    return df0


def listlist2df(listlist, name):
    df0 = pd.DataFrame()
    for epoch in range(len(listlist)):
        df_epoch = pd.DataFrame()
        for q in range(len(listlist[epoch])):
            col_name = name + '_' + str(q)
            df_epoch.loc[epoch, col_name] = listlist[epoch][q]
        df0 = pd.concat([df0, df_epoch])
    return df0

def dict_all2df(dict_all):
    list1 = ['foldnum', 'epoch', 'best_auc', 'best_epoch', 'lr_epoch', 'loss_train', 'loss_valid',
             'loss_test', 'acc_train', 'acc_valid', 'acc_test']
    list2 = ['f1_train', 'f1_valid', 'f1_test', 'loss_steps_train', 'loss_steps_valid',
             'loss_steps_test', 'acc_steps_train', 'acc_steps_valid', 'acc_steps_test']
    list3 = ['roc_train', 'roc_valid', 'roc_test']
    dict_all0 = {k: dict_all[k] for k in list1}
    df0 = pd.DataFrame(dict_all0)
    for name in list2:
        df00 = listlist2df(dict_all[name], name)
        df0 = pd.concat([df0, df00], axis=1)
    for name in list3:
        df00 = dictlistlist2df(dict_all[name], name)
        df0 = pd.concat([df0, df00], axis=1)
    return df0


def plot_eval_curve(df, name, name_list, plot_dir, pic_name):
    """
    绘制 三个集的评价指标曲线
    :param df:
    :param name:
    :param name_list: 绘制指标的名字列表
    :param plot_dir:
    :param pic_name:
    :return:
    """
    df_train = df[['epoch','foldnum', name_list[0]]]
    df_valid = df[['epoch','foldnum', name_list[1]]]
    df_test = df[['epoch','foldnum', name_list[2]]]
    df_train['set'] = 'train'
    df_valid['set'] = 'valid'
    df_test['set'] = 'test'
    df_train.rename(columns={name_list[0]: name}, inplace=True)
    df_valid.rename(columns={name_list[1]: name}, inplace=True)
    df_test.rename(columns={name_list[2]: name}, inplace=True)
    df = pd.concat([df_train, df_valid, df_test])
    sns.relplot(x='epoch', y=name, hue='set', data=df, kind='line', palette='tab10')
    plt.title(name)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, str(pic_name) + '.png'), dpi=300); plt.clf();    plt.close()
    # plt.show()

def plot_steps_curve(df, name, plot_dir, lenQ, pic_name):
    """
    绘制多步评价指标曲线
    :param df:
    :param name:
    :param plot_dir:
    :param lenQ:
    :param pic_name:
    :return:
    """
    dfq_all = []
    for q in range(lenQ):
        if 'roc' in name:
            name_list = name.rsplit('_')
            nameq = name_list[0] + '_' + name_list[1] + '_' + str(q+1) + '_' + name_list[2]
        elif 'f1' in name:
            nameq = name + '_' + str(q+1)
        else:
            nameq = name + '_' + str(q)
        dfq = df[['epoch', 'foldnum', nameq]]
        dfq['set'] = q+1
        dfq.rename(columns={nameq: name}, inplace=True)
        dfq_all.append(dfq)
    dfq_all = pd.concat(dfq_all)
    sns.relplot(x='epoch', y=name, hue='set', data=dfq_all, kind='line', palette='tab10')
    plt.title(name)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, str(pic_name) + '.png'), dpi=300); plt.clf();    plt.close()
    # plt.show()

def result_visualization(df, plot_dir, lenQ):
    """
    根据结果df进行结果可视化
    :param df:
    :param plot_dir:
    :param lenQ:
    :return:
    """
    temp_dir = os.path.join(plot_dir, 'temp')
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    ## plot each visualization
    plot_eval_curve(df, 'loss', ['loss_train', 'loss_valid', 'loss_test'], temp_dir, 1)
    plot_eval_curve(df, 'acc', ['acc_train', 'acc_valid', 'acc_test'], temp_dir, 2)
    plot_eval_curve(df, 'f1', ['f1_train_0', 'f1_valid_0', 'f1_test_0'], temp_dir, 3)

    sns.relplot(x='epoch', y='lr_epoch', data=df, kind='line', palette='tab10')
    plt.title('lr')
    plt.tight_layout()
    plt.savefig(os.path.join(temp_dir, '4.png'), dpi=300); plt.clf();    plt.close()

    plot_eval_curve(df, 'roc_macro', ['roc_train_0_macro', 'roc_valid_0_macro', 'roc_test_0_macro'], temp_dir, 5)
    plot_eval_curve(df, 'roc_micro', ['roc_train_0_micro', 'roc_valid_0_micro', 'roc_test_0_micro'], temp_dir, 6)
    plot_eval_curve(df, 'roc_0', ['roc_train_0_0', 'roc_valid_0_0', 'roc_test_0_0'], temp_dir, 7)
    plot_eval_curve(df, 'roc_1', ['roc_train_0_1', 'roc_valid_0_1', 'roc_test_0_1'], temp_dir, 8)

    plot_steps_curve(df, 'f1_train', temp_dir, lenQ, 9)
    plot_steps_curve(df, 'f1_valid', temp_dir, lenQ, 10)
    plot_steps_curve(df, 'f1_test', temp_dir, lenQ, 11)

    plot_steps_curve(df, 'loss_steps_train', temp_dir, lenQ, 12)
    plot_steps_curve(df, 'loss_steps_valid', temp_dir, lenQ, 13)
    plot_steps_curve(df, 'loss_steps_test', temp_dir, lenQ, 14)

    plot_steps_curve(df, 'acc_steps_train', temp_dir, lenQ, 15)
    plot_steps_curve(df, 'acc_steps_valid', temp_dir, lenQ, 16)
    plot_steps_curve(df, 'acc_steps_test', temp_dir, lenQ, 17)

    plot_steps_curve(df, 'roc_train_macro', temp_dir, lenQ, 18)
    plot_steps_curve(df, 'roc_valid_macro', temp_dir, lenQ, 19)
    plot_steps_curve(df, 'roc_test_macro', temp_dir, lenQ, 20)

    plot_steps_curve(df, 'roc_train_micro', temp_dir, lenQ, 21)
    plot_steps_curve(df, 'roc_valid_micro', temp_dir, lenQ, 22)
    plot_steps_curve(df, 'roc_test_micro', temp_dir, lenQ, 23)

    plot_steps_curve(df, 'roc_train_0', temp_dir, lenQ, 24)
    plot_steps_curve(df, 'roc_valid_0', temp_dir, lenQ, 25)
    plot_steps_curve(df, 'roc_test_0', temp_dir, lenQ, 26)

    plot_steps_curve(df, 'roc_train_1', temp_dir, lenQ, 27)
    plot_steps_curve(df, 'roc_valid_1', temp_dir, lenQ, 28)
    plot_steps_curve(df, 'roc_test_1', temp_dir, lenQ, 29)


    ## concat to a whole pic
    nrows = 10
    ncols = 3
    fig, axarr = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows))
    pic = 1
    breakFlag = False
    for row in range(nrows):
        for col in range(ncols):
            try:
                axarr[row, col].imshow(mpimg.imread(os.path.join(temp_dir, str(pic) + '.png')))
                pic += 1
                if pic == 34:
                    breakFlag = True
                    break
            except:
                pass
        if breakFlag:
            break
    [ax.set_axis_off() for ax in axarr.ravel()]
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'eval.png'), dpi=300); plt.clf();    plt.close()

def write_log(df, foldname, args):
    """
    训练结果写入每个文件夹下的 log.txt 并补充进总文件夹下的 sum.txt
    :param df:
    :param foldname:
    :param args:
    :return:
    """
    # 5折上的best auc的均值和标准差 best model对应的fold和epoch
    best_auc = []
    for foldnum in range(5):
        best_auc_fold = df.loc[(df['foldnum'] == foldnum + 1) & (df['epoch'] == 1), 'best_auc']
        best_auc.extend(best_auc_fold)

    best_auc_avg = np.mean(np.array(best_auc))
    best_auc_std = np.std(np.array(best_auc))
    max_best_auc = np.max(np.array(best_auc))
    max_best_fold = df.loc[(df['best_auc'] == max_best_auc) & (df['epoch'] == 1), 'foldnum'].values[0]
    max_best_epoch = df.loc[(df['best_auc'] == max_best_auc) & (df['epoch'] == 1), 'best_epoch'].values[0]

    # 保留4位小数
    best_auc_avg = '%.4f' % best_auc_avg
    best_auc_std = '%.4f' % best_auc_std
    max_best_auc = '%.4f' % max_best_auc

    # 写入log文件和summary文件 txt
    log_name = os.path.join(args.root_dir, foldname, 'log.txt')
    sum_name = os.path.join(args.root_dir, 'summary.txt')

    f1 = open(log_name, 'w')
    f2 = open(sum_name, 'a')

    hyper_para = {'foldname': foldname,
                  'auc_avg': best_auc_avg,
                  'auc_std': best_auc_std,
                  'max_auc': max_best_auc,
                  'max_fold': max_best_fold,
                  'max_epo': max_best_epoch,
                  'lr': args.lr,
                  'bs': args.batch_size,
                  'wd': args.weight_decay,
                  'epos': args.epoch_num,
                  'lenP': args.lenP,
                  'lenQ': args.lenQ,
                  'step': args.step,
                  'layer': args.layer,
                  'model': args.modelname,
                  'seed': args.seed,
                  'opt': args.optimizer,
                  'sch': args.scheduler,
                  'fusion': args.fusionFlag,
                  'lstmL': args.lstmLayer,
                  'STF': args.SubTestFlag,
                  'sub': args.sub_group,
                  'noise': args.noise,
                  'dropCH': args.dropCH,
                  'dropR': args.dropR,
                  'hid_dim': args.hidden_dim,
                  'end_dim': args.end_dim,
                  'gcnF': args.gcnFlag,
                  'tempF': args.tempFlag,
                  'timeP': args.timeProj,
                  'read': args.readout,
                  'predata': args.predata,
                  'drop': args.dropout,
                  'cri': args.criterion[0],
                  'alpha': args.criterion[1]
                  }
    # 如果summary文件为空，则写入表头，如果不为空则直接写入结果
    if not os.path.getsize(sum_name):
        for key in hyper_para.keys():
            f2.write(key + '\t')
            if key == 'foldname':
                f2.write('\t')
        f2.write('\n')
    # 写入 log 表头结果 和 sum结果
    for key in hyper_para.keys():
        f1.write(key + '\t')
        if key == 'foldname':
            f1.write('\t')
    f1.write('\n')
    for key, value in hyper_para.items():
        f1.write(str(value) + '\t')
        f2.write(str(value) + '\t')
        # if key == 'foldname':
        #     f1.write('\t')
        #     f2.write('\t')
    f2.write('\n')

    f1.close()
    f2.close()


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 10
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_auc):

        score = val_auc

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


def valid(model, valid_loader, criterion, lenQ, device):
    """
    测试
    :param model:
    :param valid_loader:
    :param criterion:
    :param lenQ:
    :param device:
    :return:
    """
    model.eval()
    total_loss = 0.0
    total_loss_steps = [0.0 for q in range(lenQ)]
    total_correct = 0
    total_correct_steps = [0 for q in range(lenQ)]
    predlist = []
    labellist = []
    scorelist = []

    for i, data in enumerate(valid_loader):
        with torch.no_grad():
            label, label_1d, ecg, fnirsx, fnirsA1, fnirsA2, fnirsA3, fnirsA4 = data
            label = label.float().to(device)  # [batch_size, cate_nums, lenQ]
            label_1d = label_1d.float().to(device)  # [batch_size, lenQ]
            ecg = ecg.float().to(device)  # [batch_size, ecg_feature_nums, lenP]
            fnirsx = fnirsx.float().to(device)  # [batch_size, fnirs_feature_nums, node_nums, lenP]
            fnirsA1 = fnirsA1.float().to(device)  # [batch_size, node_nums, node_nums, lenP]
            fnirsA2 = fnirsA2.float().to(device)
            fnirsA3 = fnirsA3.float().to(device)
            fnirsA4 = fnirsA4.float().to(device)
            As = [fnirsA1, fnirsA2, fnirsA3, fnirsA4]

            outputs = model(fnirsx, As, ecg)  # [batch_size, cate_nums, lenQ]
            loss_steps, loss = cal_steps_loss(outputs, label, criterion)  # list (lenQ), float
            _, predictions = torch.max(outputs, 1)  # prediction -> sum.indices [batch_size, lenQ]

            total_loss += loss.item() * fnirsx.size(0)
            total_correct += torch.sum(predictions == label_1d.data)
            total_correct_cmpr = torch.sum(predictions == label_1d.data, dim=0)
            for q in range(lenQ):
                total_loss_steps[q] += (loss_steps[q].item() * fnirsx.size(0))
                total_correct_steps[q] += (total_correct_cmpr[q])
            predlist.extend(predictions.cpu().detach().numpy())
            labellist.extend(label_1d.data.cpu().detach().numpy())
            scorelist.extend(outputs.cpu().detach().numpy())

    epoch_loss = total_loss / len(valid_loader.dataset)
    epoch_loss_steps = [loss / len(valid_loader.dataset) for loss in total_loss_steps]
    epoch_acc = total_correct.double() / (len(valid_loader.dataset) * lenQ)
    epoch_acc_steps = [correct.cpu().detach().numpy() / len(valid_loader.dataset) for correct in total_correct_steps]
    labellist = np.array(labellist)  # [batch_size, lenQ]
    scorelist = np.array(scorelist)  # [batch_size, cate_nums, lenQ]
    predlist = np.array(predlist)  # [batch_size, lenQ]
    auc_list, f1_list = cal_roc(labellist, scorelist, predlist)
    prauc_list, brier_list = cal_multistep(labellist, scorelist, predlist)

    return epoch_loss, epoch_loss_steps, epoch_acc.item(), epoch_acc_steps, auc_list, f1_list, predlist, labellist, scorelist, prauc_list, brier_list

def valid_tsne(model, valid_loader, device):
    """
    获取 decoder 前的output
    :param model:
    :param valid_loader:
    :param device:
    :return:
    """
    model.eval()
    labellist = []
    scorelist = []

    for i, data in enumerate(valid_loader):
        with torch.no_grad():
            label, label_1d, ecg, fnirsx, fnirsA1, fnirsA2, fnirsA3, fnirsA4 = data
            label = label.float().to(device)  # [batch_size, cate_nums, lenQ]
            label_1d = label_1d.float().to(device)  # [batch_size, lenQ]
            ecg = ecg.float().to(device)  # [batch_size, ecg_feature_nums, lenP]
            fnirsx = fnirsx.float().to(device)  # [batch_size, fnirs_feature_nums, node_nums, lenP]
            fnirsA1 = fnirsA1.float().to(device)  # [batch_size, node_nums, node_nums, lenP]
            fnirsA2 = fnirsA2.float().to(device)
            fnirsA3 = fnirsA3.float().to(device)
            fnirsA4 = fnirsA4.float().to(device)
            As = [fnirsA1, fnirsA2, fnirsA3, fnirsA4]

            outputs = model(fnirsx, As, ecg)  # [batch_size, cate_nums, lenQ]
            labellist.extend(label_1d.data.cpu().detach().numpy())
            scorelist.extend(outputs.cpu().detach().numpy())

    labellist = np.array(labellist)  # [batch_size, lenQ]
    scorelist = np.array(scorelist)  # [batch_size, cate_nums, lenQ]

    return labellist, scorelist

def plot_tsne(plot_list, label_list, temp_dir, name, pal, title, perplexity):
    """
    绘制tsne降维可视化结果
    :param plot_list: [batch_size, hidden_dim]
    :param label_list: [batch_size]
    :param temp_dir: 保存路径
    :param name: 图片名称
    :param pal: 颜色卡
    :param title: 图标题
    :param perplexity: tsne参数
    :return:
    """
    tsne = TSNE(perplexity=perplexity, n_components=2, init='pca', n_iter=5000)
    tsne_list = tsne.fit_transform(plot_list)
    tsne_data = {'x': tsne_list[:, 0], 'y': tsne_list[:, 1], 'label': label_list}
    tsne_df = pd.DataFrame(tsne_data)
    g = sns.jointplot(data=tsne_df, x='x', y='y', hue='label', palette=pal, s=10, marginal_kws={'common_norm': False})
    handles, labels = g.ax_joint.get_legend_handles_labels()
    g.ax_joint.legend(handles=handles, labels=['normal', 'abnormal'], fontsize=15)
    g.ax_joint.xaxis.set_visible(False)
    g.ax_joint.yaxis.set_visible(False)
    g.fig.suptitle(title, y=1.00, fontsize=20)
    plt.savefig(os.path.join(temp_dir, name + '.png'), dpi=300, bbox_inches='tight'); plt.clf();    plt.close()


def plot_matrix(true_list, pred_list, temp_dir, name, vmax):
    """
    绘制混淆矩阵
    :param true_list: 真实标签列表，[batch_size]
    :param pred_list: 预测标签列表， [batch_size]
    :param temp_dir: 保存路径
    :param name: 图片名称
    :return:
    """
    matrix = confusion_matrix(true_list, pred_list, labels=[0,1], normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=['normal', 'abnormal'])
    disp.plot(cmap=plt.cm.Reds)
    plt.title('Confusion matrix')
    plt.savefig(os.path.join(temp_dir, name + '.png'), dpi=300, bbox_inches='tight'); plt.clf(); plt.close()

    # plt.figure(dpi=300)
    # plt.matshow(matrix, cmap=plt.cm.Reds) # 根据最下面的图按自己需求更改颜色 #YlGn
    # plt.colorbar()
    # plt.clim(0, vmax)
    #
    # for i in range(len(matrix)):
    #     for j in range(len(matrix)):
    #         color = (1, 1, 1) if i == j else (0, 0, 0)  # 对角线字体白色，其他黑色
    #         label = round(matrix[j, i]/np.sum(matrix)*100) #最后一个参量表示保留小数位数
    #         plt.annotate(label, xy=(i, j), horizontalalignment='center', verticalalignment='center', fontsize=15, color=color)
    #
    # plt.tick_params(labelsize=15) # 设置左边和上面的label类别如0,1,2,3,4的字体大小。
    #
    # # plt.ylabel('True label')
    # # plt.xlabel('Predicted label')
    # plt.ylabel('True label', fontdict={'family': 'Arial', 'size': 20}) # 设置字体大小。
    # plt.xlabel('Predicted label', fontdict={'family': 'Arial', 'size': 20})
    # plt.xticks(range(0,2), labels=['cate1', 'cate2']) # 将x轴或y轴坐标，刻度 替换为文字/字符
    # plt.yticks(range(0,2), labels=['cate1', 'cate2'], rotation=90)
    #
    # plt.savefig(os.path.join(temp_dir, name +'.png'), dpi=300, bbox_inches = 'tight'); plt.clf();    plt.close()
    # # plt.show()


def plot_roc(trues, scores, temp_dir, name):
    """
    绘制 micro macro cate1 cate2 cate3 的 ROC 曲线
    :param trues: 真实标签列表 [batch_size]
    :param scores: 预测分数列表 [batch_size, cate_nums]
    :param temp_dir:
    :param name:
    :return:
    """
    num_class = 2
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    trues = torch.from_numpy(trues).to(torch.int64)
    trues_ones = F.one_hot(trues, num_classes=num_class)

    for i in range(num_class):
        fpr[i], tpr[i], _ = roc_curve(trues_ones[:, i], scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(trues_ones.ravel(), scores.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_class)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_class):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= num_class
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    lw = 2
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(num_class), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of category {0} (area = {1:0.2f})'
                       ''.format(i + 1, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC for model' + name)
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(temp_dir, name + '.png'), dpi=300, bbox_inches='tight'); plt.clf();    plt.close()
    # plt.show()
    return fpr["macro"], tpr["macro"], roc_auc["macro"]

def plot_roc_all_steps(fpr_list, tpr_list, roc_auc_list, temp_dir, name):
    lw = 2
    plt.figure()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    for i in range(len(fpr_list)):
        plt.plot(fpr_list[i], tpr_list[i],
                 label='macro-average ROC curve for step {0} (area = {1:0.4f})'
                       ''.format(i+1, roc_auc_list[i]),
                 color=colors[i], lw=lw)
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC for model' + name)
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(temp_dir, name + '.png'), dpi=300, bbox_inches='tight'); plt.clf(); plt.close()



def eval_visualization(predlist, labellist, scorelist, plotlist, temp_dir, nameflag, lenQ):
    # 合并所有时间步list
    plotlist0 = np.transpose(plotlist, (0, 2, 1))
    plotlist0 = plotlist0.reshape(-1, plotlist0.shape[-1])  # [batch_size * lenQ, hidden_dim]
    labellist0 = labellist.reshape(-1).squeeze()  # [batch_size * lenQ]
    predlist0 = predlist.reshape(-1).squeeze()  # [batch_size * lenQ]
    scorelist0 = np.transpose(scorelist, (0, 2, 1))
    scorelist0 = scorelist0.reshape(-1, scorelist0.shape[-1])  # [batch_size * lenQ, cate_nums]

    ## 用sklearn的 classification report验证结果
    repo_dict = dict()
    repo_dict['all'] = classification_report(labellist0, predlist0, target_names=['normal', 'abnormal'])
    for q in range(labellist.shape[-1]):
        repo_dict['step'+str(q+1)] = classification_report(labellist[:, q].squeeze(), predlist[:, q].squeeze(), target_names=['normal', 'abnormal'])
    print(repo_dict)

    ## 设置绘图参数
    color = ["#1E90FF", "#FF7256"]
    pal = sns.color_palette(color)
    sns.set_palette(pal)

    ## 绘制 roc曲线
    fpr_list, tpr_list, roc_auc_list = [], [], []
    plot_roc(labellist0, scorelist0, temp_dir, nameflag + '_roc_0')
    for q in range(labellist.shape[-1]):
        fpr_i, tpr_i, auc_i = plot_roc(labellist[:, q].squeeze(), scorelist[:, :, q].squeeze(), temp_dir, nameflag + '_roc_' + str(q + 1))
        fpr_list.append(fpr_i); tpr_list.append(tpr_i); roc_auc_list.append(auc_i)
    plot_roc_all_steps(fpr_list, tpr_list, roc_auc_list, temp_dir, nameflag + '_roc_allstep')

    ## 绘制混淆矩阵
    plot_matrix(labellist0, predlist0, temp_dir, nameflag + '_cm_0', predlist0.shape[0])
    for q in range(predlist.shape[-1]):
        plot_matrix(labellist[:, q].squeeze(), predlist[:, q].squeeze(), temp_dir, nameflag + '_cm_' + str(q + 1),
                    predlist.shape[0])

    ## 绘制tsne降维结果
    plot_tsne(plotlist0, labellist0, temp_dir, nameflag+'_tsne_0', pal, '0', 30)
    # for q in range(plotlist.shape[-1]):
    #     plot_tsne(plotlist[:, :, q].squeeze(), labellist[:, q].squeeze(), temp_dir,
    #               nameflag + '_tsne_' + str(q + 1), pal)

    # ## 整理合并绘图
    # fig, axarr = plt.subplots(3, lenQ + 1, figsize=((lenQ + 1) * 10, 3 * 10))
    # iname = ['roc', 'cm', 'tsne']
    # for i in range(3):
    #     for j in range(lenQ + 1):
    #         try:
    #             axarr[i, j].imshow(
    #                 mpimg.imread(os.path.join(temp_dir, nameflag + '_' + iname[i] + '_' + str(j) + '.png')))
    #         except:
    #             pass
    # [ax.set_axis_off() for ax in axarr.ravel()]
    # plt.tight_layout()
    # plt.savefig(os.path.join(temp_dir.replace('temp', ''), nameflag + '.png'), dpi=600);
    # plt.clf();
    # plt.close()
    return repo_dict


def test_pre_data(model, loader, lenQ, device, criterion, test_dir, temp_dir, nameflag, plotFlag):
    """
    获取最优模型的测试结果，即 混淆矩阵 ROC tsne可视化结果
    :param best_model:
    :param loader: 数据loader
    :param lenQ:
    :param device:
    :param criterion:
    :param temp_dir:
    :param nameflag: 图片名前缀
    :return:
        test_df：评价指标整理结果df
    """
    ## 预测结果
    best_model = copy.deepcopy(model)
    pred_model = best_model.to(device)
    loss, loss_steps, acc, acc_steps, auc, f1, predlist, labellist, scorelist, prauc_list, brier_list = valid(pred_model, loader, criterion, lenQ, device)

    ## tsne降维结果
    plot_model = best_model.to(device)
    # plot_model.end_conv_1 = torch.nn.Sequential()
    plot_model.end_conv_2 = torch.nn.Sequential()
    labellist, plotlist = valid_tsne(plot_model, loader, device)

    ## 保存 predlist labellist scorelist plotlist 画图所需数据
    plot_data_dict = {'predlist': predlist, 'labellist': labellist, 'scorelist': scorelist, 'plotlist': plotlist}
    save_pkl(plot_data_dict, os.path.join(test_dir, nameflag+'_plot_data_dict.pkl'))


    ## 将评价指标转为df形式
    test_result = {'nameflag': nameflag, 'loss': loss}
    for q in range(len(loss_steps)):
        test_result['loss_'+str(q+1)] = loss_steps[q]
    test_result['acc'] = acc
    for q in range(len(acc_steps)):
        test_result['acc_'+str(q+1)] = acc_steps[q]
    for q in range(len(f1)):
        test_result['f1_'+str(q)] = f1[q]
    for q in range(len(auc)):
        for cate in auc[q]:
            test_result['auc_'+str(q)+'_'+str(cate)] = auc[q][cate]
    for q in range(len(prauc_list)):
        test_result['prauc_'+str(q)] = prauc_list[q]
    for q in range(len(brier_list)):
        test_result['brier_'+str(q)] = brier_list[q]
    test_df = pd.DataFrame(test_result, index=[0])

    repo_dict = dict()
    if plotFlag:
        repo_dict = eval_visualization(predlist, labellist, scorelist, plotlist, temp_dir, nameflag, lenQ)
    return test_df, repo_dict

def log_test_result(test_df_train, test_df_test, repo_dict_train, repo_dict_test, test_dir, root_dir, fold_name, test_sum_name):
    # 整理结果
    df_final = pd.concat([test_df_train, test_df_test], axis=0)
    columns = list(df_final)
    columns.insert(2, columns.pop(columns.index('acc')))
    columns.insert(3, columns.pop(columns.index('auc_0_macro')))
    columns.insert(4, columns.pop(columns.index('f1_0')))
    df_final = df_final.loc[:, columns]
    df_final.reset_index(drop=True, inplace=True)
    df_final.to_csv(os.path.join(test_dir, 'df_final.csv'), index=False)

    # test结果写入 test_summary.txt
    sum_name = os.path.join(root_dir, test_sum_name+'.txt')
    f1 = open(sum_name, 'a')
    if not os.path.getsize(sum_name):
        f1.write('foldname'); f1.write('\t\t')
        for c in df_final.columns:
            f1.write(c); f1.write('\t')
        f1.write('\n')
    f1.write(fold_name)
    f1.write('\t')
    for c in df_final.columns:
        value = df_final[c][1]
        try:
            value = '%.4f' % float(value)
            value = str(value)
        except:
            pass
        f1.write(value); f1.write('\t')
    f1.write('\n'); f1.close()

    # repo_dict 写入txt
    repo_name = os.path.join(test_dir, 'repo_dict_test.txt')
    f1 = open(repo_name, 'w')
    for key, value in repo_dict_test.items():
        f1.write(key + '\n' + value + '\n'*2)
    f1.close()



def test_and_visualization(df, root_dir, fold_name, dataset_dir, lenQ, plotFlag, criterion):
    """
    测试 valid上的最优模型，test各个指标  tsne降维结果 混淆矩阵 ROC curve
    :param df: 结果csv
    :param fold_dir: 训练根目录， 如 r'E:\AMY\研究\11时序建模\2train\train_result\0161656001'
    :param lenQ:
    :return:
    """
    fold_dir = os.path.join(root_dir, fold_name)
    # 获取 kfold 中最优模型
    best_auc = []
    for foldnum in range(5):
        best_auc_fold = df.loc[(df['foldnum'] == foldnum + 1) & (df['epoch'] == 1), 'best_auc']
        best_auc.extend(best_auc_fold)
    max_best_auc = np.max(np.array(best_auc))
    max_best_fold = df.loc[(df['best_auc'] == max_best_auc) & (df['epoch'] == 1), 'foldnum'].values[0]
    best_model_dir = os.path.join(fold_dir, 'model', 'Fold' + str(max_best_fold) + '_best_model.pt')
    best_model = torch.load(best_model_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 准备数据集
    datasetPhy = load_pkl(dataset_dir)
    train_set = datasetPhy['train']
    test_set = datasetPhy['test']
    train_loader = DataLoader(dataset=train_set, batch_size=len(train_set), shuffle=False)
    test_loader = DataLoader(dataset=test_set, batch_size=len(test_set), shuffle=False)

    # 获取评价结果和可视化结果
    test_dir = os.path.join(fold_dir, 'test')
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    temp_dir = os.path.join(test_dir, 'temp')
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    test_df_test, repo_dict_test = test_pre_data(best_model, test_loader, lenQ, device, criterion, test_dir, temp_dir, 'test', True)
    test_df_train, repo_dict_train = test_pre_data(best_model, train_loader, lenQ, device, criterion, test_dir, temp_dir, 'train', False)

    log_test_result(test_df_train, test_df_test, repo_dict_train, repo_dict_test, test_dir, root_dir, fold_name, 'test_summary')


def only_test_visualization(df, fold_dir, lenQ):
    # 加载 predlist labellist scorelist plotlist 画图所需数据
    plot_data_dict = load_pkl(os.path.join(fold_dir, 'test', 'test_plot_data_dict.pkl'))
    eval_visualization(plot_data_dict['predlist'], plot_data_dict['labellist'], plot_data_dict['scorelist'],
                       plot_data_dict['plotlist'], os.path.join(fold_dir, 'test', 'temp'), 'test', lenQ)

    plot_data_dict = load_pkl(os.path.join(fold_dir, 'test', 'train_plot_data_dict.pkl'))
    eval_visualization(plot_data_dict['predlist'], plot_data_dict['labellist'], plot_data_dict['scorelist'],
                       plot_data_dict['plotlist'], os.path.join(fold_dir, 'test', 'temp'), 'train', lenQ)
