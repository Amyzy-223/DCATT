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

from fontTools.misc.cython import returns
from matplotlib import pyplot as plt, image as mpimg
from sklearn.metrics import roc_curve, auc, roc_auc_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, \
    classification_report
from scipy.interpolate import interp1d
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
    :param trues: [batch_size, cate_nums]
    :param scores: [batch_size, cate_nums]
    :return:
    """
    num_class = scores.shape[-1]
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    scores = softmax(scores, dim=1)

    for i in range(num_class):
        fpr[i], tpr[i], _ = roc_curve(trues[:, i], scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(trues.ravel(), scores.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    fpr_grid = np.linspace(0, 1, 1000)
    mean_tpr = np.zeros_like(fpr_grid)
    for i in range(num_class):
        mean_tpr += np.interp(fpr_grid, fpr[i], tpr[i])
    mean_tpr /= num_class
    fpr["macro"] = fpr_grid
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    return fpr, tpr, roc_auc

def softmax(x, dim):
    e_x = np.exp(x - np.max(x, axis=dim, keepdims=True))
    return e_x / e_x.sum(axis=dim, keepdims=True)

def cal_roc(labellist, scorelist, predlist, labellist_onehot):
    """
    分别计算 所有lenQ的auc，和每一个step的auc
    :param labellist: array [batch_size]
    :param scorelist: array [batch_size, cate_nums]
    :param predlist: array [batch_size]
    :param labellist_onehot: array [batch_size, cate_nums]
    :return:
        auc: array
        f1: array
    """

    # fpr, tpr, roc_auc = cal_roc_stand(labellist, scorelist)
    _, _, roc_auc = cal_roc_stand(labellist_onehot, scorelist)
    f1 = f1_score(labellist, predlist, average='macro')
    return roc_auc, f1




def dictlist2df(dictlist, name):
    df0 = pd.DataFrame()
    for epoch in range(len(dictlist)):
        df_epoch = pd.DataFrame()
        for key, value in dictlist[epoch].items():
            col_name = name + '_' + str(key)
            df_epoch.loc[epoch, col_name] = value
        df0 = pd.concat([df0, df_epoch])
    return df0

def dict_all2df(dict_all, stage):
    if stage == 1:
        list1 = ['foldnum', 'epoch', 'best_auc', 'best_epoch', 'lr_epoch', 'loss_train', 'loss_valid',
                 'loss_test', 'acc_train', 'acc_valid', 'acc_test','f1_train', 'f1_valid', 'f1_test']
        list2 = ['roc_train', 'roc_valid', 'roc_test']
    else:
        list1 = ['foldnum', 'epoch', 'best_auc', 'best_epoch', 'lr_epoch', 'loss_train', 'loss_test',
                 'acc_train', 'acc_test','f1_train', 'f1_test']
        list2 = ['roc_train', 'roc_test']
    dict_all0 = {k: dict_all[k] for k in list1}
    df0 = pd.DataFrame(dict_all0)
    for name in list2:
        df00 = dictlist2df(dict_all[name], name)
        df0 = pd.concat([df0, df00], axis=1)
    return df0


def plot_eval_curve(df, name, name_list, plot_dir, pic_name, stage):
    """
    绘制 三个集的评价指标曲线
    :param df:
    :param name:
    :param name_list: 绘制指标的名字列表
    :param plot_dir:
    :param pic_name:
    :return:
    """
    if stage == 1:
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
    else:
        df_train = df[['epoch', 'foldnum', name_list[0]]]
        df_test = df[['epoch', 'foldnum', name_list[1]]]
        df_train['set'] = 'train'
        df_test['set'] = 'test'
        df_train.rename(columns={name_list[0]: name}, inplace=True)
        df_test.rename(columns={name_list[1]: name}, inplace=True)
        df = pd.concat([df_train, df_test])
    sns.relplot(x='epoch', y=name, hue='set', data=df, kind='line', palette='tab10')
    plt.title(name)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, str(pic_name) + '.png'), dpi=300); plt.clf();    plt.close()
    # plt.show()


def result_visualization(df, plot_dir, num_class, stage):
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

    if stage == 1:
        namelist = ['train', 'valid', 'test']
    else:
        namelist = ['train', 'test']

    i = 1
    ## plot each visualization
    plot_eval_curve(df, 'loss', ['loss_'+name for name in namelist], temp_dir, i, stage); i+=1
    plot_eval_curve(df, 'acc', ['acc_'+name for name in namelist], temp_dir, i, stage); i+=1
    plot_eval_curve(df, 'f1', ['f1_'+name for name in namelist], temp_dir, i, stage); i+=1

    sns.relplot(x='epoch', y='lr_epoch', data=df, kind='line', palette='tab10')
    plt.title('lr')
    plt.tight_layout()
    plt.savefig(os.path.join(temp_dir, str(i)+'.png'), dpi=300); plt.clf();    plt.close(); i+=1

    plot_eval_curve(df, 'roc_macro', ['roc_' + name + '_macro' for name in namelist], temp_dir, i, stage); i+=1
    plot_eval_curve(df, 'roc_micro', ['roc_' + name + '_micro' for name in namelist], temp_dir, i, stage); i+=1
    for c in range(num_class):
        plot_eval_curve(df, 'roc_'+str(c), ['roc_'+name+'_'+str(c) for name in namelist], temp_dir, i, stage); i+=1

    ## concat to a whole pic
    nrows = int(np.ceil(i/3))
    ncols = 3
    fig, axarr = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows))
    pic = 1
    breakFlag = False
    for row in range(nrows):
        for col in range(ncols):
            try:
                axarr[row, col].imshow(mpimg.imread(os.path.join(temp_dir, str(pic) + '.png')))
                pic += 1
            except:
                breakFlag = True
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
                  'model': args.modelname,
                  'data': args.dataset,
                  'mode': args.SubTestFlag,
                  'sub': args.subject,
                  'layer': args.layer,
                  'seed': args.seed,
                  'opt': args.optimizer,
                  'sch': args.scheduler,
                  'hid_dim': args.hidden_dim,
                  'end_dim': args.end_dim,
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


def plot_tsne(plot_list, label_list, temp_dir, name, title, perplexity):
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
    ## 设置绘图参数
    # color = ["#1E90FF", "#FF7256"]
    # pal = sns.color_palette(color)
    # sns.set_palette(pal)
    try:
        tsne = TSNE(perplexity=perplexity, n_components=2, init='pca', n_iter=5000)
    except:
        tsne = TSNE(perplexity=perplexity, n_components=2, init='pca', max_iter=5000)
    tsne_list = tsne.fit_transform(plot_list)
    tsne_data = {'x': tsne_list[:, 0], 'y': tsne_list[:, 1], 'label': label_list}
    tsne_df = pd.DataFrame(tsne_data)
    g = sns.jointplot(data=tsne_df, x='x', y='y', hue='label', s=10, marginal_kws={'common_norm': False}, palette="bright")
    # handles, labels = g.ax_joint.get_legend_handles_labels()
    # g.ax_joint.legend(handles=handles, labels=['normal', 'abnormal'], fontsize=15)
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
    matrix = confusion_matrix(true_list, pred_list, normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=matrix)
    disp.plot(cmap=plt.cm.Reds)
    plt.title('Confusion matrix')
    plt.savefig(os.path.join(temp_dir, name + '.png'), dpi=300, bbox_inches='tight'); plt.clf(); plt.close()



def plot_roc(trues, scores, temp_dir, name):
    """
    绘制 micro macro cate1 cate2 cate3 的 ROC 曲线
    :param trues: 真实标签列表 [batch_size]
    :param scores: 预测分数列表 [batch_size, cate_nums]
    :param temp_dir:
    :param name:
    :return:
    """
    fpr, tpr, roc_auc = cal_roc_stand(trues, scores)
    num_class = scores.shape[-1]
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

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green'])
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


def eval_visualization(predlist, labellist, labellist_onehot, scorelist, plotlist_tsne, labellist_tsne, temp_dir, nameflag):
    ## 用sklearn的 classification report验证结果
    num_class = predlist.shape[-1]
    repo_dict = dict()
    repo_dict= classification_report(labellist, predlist)
    print(repo_dict)

    ## 绘制 roc曲线
    fpr_list, tpr_list, roc_auc_list = [], [], []
    plot_roc(labellist_onehot, scorelist, temp_dir, nameflag + '_roc')

    ## 绘制混淆矩阵
    plot_matrix(labellist, predlist, temp_dir, nameflag + '_cm', predlist.shape[0])

    ## 绘制tsne降维结果
    if nameflag == 'test':
        plot_tsne(plotlist_tsne, labellist_tsne, temp_dir, nameflag+'_tsne', nameflag, 30)
    elif nameflag == 'train':
        pass

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

def log_test_result(test_df_train, test_df_test, repo_dict_train, repo_dict_test, test_dir, root_dir, fold_name, test_sum_name):
    # 整理结果
    df_final = pd.concat([test_df_train, test_df_test], axis=0)
    columns = list(df_final)
    columns.insert(2, columns.pop(columns.index('acc')))
    columns.insert(3, columns.pop(columns.index('auc_macro')))
    columns.insert(4, columns.pop(columns.index('f1')))
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
    f1.write(repo_dict_test)
    f1.close()



def extract_kfold_best(df):
    foldnum_set = set(df['foldnum'])
    best_auc = max(set(df['best_auc']))
    best_epoch = df[(df['best_auc'] == best_auc) & (df['epoch']==1)]['best_epoch'].values[0]
    df_final = pd.DataFrame()
    for foldnum in foldnum_set:
        best_epoch_fold = df[(df['foldnum']==foldnum) & (df['epoch']==1)]['best_epoch'].values[0]
        df_fold_best = df[(df['foldnum']==foldnum) & (df['epoch']==best_epoch_fold)][['foldnum', 'best_epoch', 'acc_valid', 'acc_test',
                                                                                      'roc_valid_macro', 'roc_test_macro',
                                                                                      'f1_valid', 'f1_test']]
        df_final = pd.concat([df_final, df_fold_best])
    return df_final, best_epoch

def generate_train_test_set(dataset, idx_dict, sub=0):
    idxs = idx_dict['idx_dict'][sub]
    train_idx, test_idx = idxs['train_idx'], idxs['test_idx']
    PhyDataset = {'train': Subset(dataset, train_idx), 'test': Subset(dataset, test_idx)}
    return PhyDataset