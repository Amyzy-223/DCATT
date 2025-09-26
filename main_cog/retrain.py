import os

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset, SubsetRandomSampler

from train_util import model_init, train_one_epoch, valid_one_epoch
from util import test_pre_data, log_test_result


def extract_kfold_best(df):
    foldnum_set = set(df['foldnum'])
    best_auc = max(set(df['best_auc']))
    best_epoch = df[(df['best_auc'] == best_auc) & (df['epoch']==1)]['best_epoch'].values[0]
    df_final = pd.DataFrame()
    for foldnum in foldnum_set:
        best_epoch_fold = df[(df['foldnum']==foldnum) & (df['epoch']==1)]['best_epoch'].values[0]
        df_fold_best = df[(df['foldnum']==foldnum) & (df['epoch']==best_epoch_fold)][['foldnum', 'best_epoch', 'acc_valid', 'acc_test',
                                                                                      'roc_valid_0_macro', 'roc_test_0_macro',
                                                                                      'f1_valid_0', 'f1_test_0']]
        df_final = pd.concat([df_final, df_fold_best])
    return df_final, best_epoch

def retrain_test(args, best_epoch, dataset, foldname):
    train_loader = DataLoader(dataset=dataset['train'], batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(dataset=dataset['test'], batch_size=args.batch_size, shuffle=False)
    model, optimizer, scheduler, criterion = model_init(args)
    model = model.to(args.device)
    print("device:", args.device)

    test_dir = os.path.join(args.root_dir, foldname, 'test2')
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    temp_dir = os.path.join(test_dir, 'temp')
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    print('=' * 40)
    for epoch in range(best_epoch):
        ## train one epoch
        print('epoch:{:d}/{:d}'.format(epoch, best_epoch))
        print('*' * 20)
        train_loss, train_loss_steps, train_acc, train_acc_steps, train_auc_list, train_f1_list, train_predlist, train_labellist, train_scorelist \
            = train_one_epoch(model, train_loader, optimizer, criterion, args)
        print("training: {:.4f}, {:.4f}, {:.4f}".format(train_loss, train_acc, train_auc_list[0]["macro"]))
        test_loss, test_loss_steps, test_acc, test_acc_steps, test_auc_list, test_f1_list, test_predlist, test_labellist, test_scorelist \
            = valid_one_epoch(model, test_loader, criterion, args)
        print("testing: {:.4f}, {:.4f}, {:.4f}".format(test_loss, test_acc, test_auc_list[0]["macro"]))
        ## scheduler update
        if args.scheduler == 'cosine':
            scheduler.step()
    best_model = model
    torch.save(best_model, os.path.join(test_dir, 'best_model.pt'))

    train_loader = DataLoader(dataset=dataset['train'], batch_size=len(dataset['train']), shuffle=False)
    test_loader = DataLoader(dataset=dataset['test'], batch_size=len(dataset['test']), shuffle=False)
    test_df_test, repo_dict_test = test_pre_data(best_model, test_loader, args.lenQ, args.device, criterion, test_dir, temp_dir, 'test', True)
    test_df_train, repo_dict_train = test_pre_data(best_model, train_loader, args.lenQ, args.device, criterion, test_dir, temp_dir, 'train', False)
    log_test_result(test_df_train, test_df_test, repo_dict_train, repo_dict_test, test_dir, args.root_dir, foldname, 'test_summary2')

