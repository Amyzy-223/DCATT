import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from genericpath import exists
import torch.optim as optim
from sklearn.model_selection import KFold
from torch.utils.data import Dataset, DataLoader, Subset, SubsetRandomSampler


from train_util import model_init, train_one_epoch, valid_one_epoch
from util import cal_roc, cal_steps_loss, dict_all2df, save_args, EarlyStopping
from retrain import extract_kfold_best


def cross_validation(dataset, args, foldname):
    result_allfold_df = pd.DataFrame()
    result_allfold = {}
    test_loader = DataLoader(dataset=dataset['test'], batch_size=args.batch_size, shuffle=False)

    ## make essential dir
    save_dir = os.path.join(args.root_dir, foldname, args.result_dir)
    model_dir = os.path.join(args.root_dir, foldname, args.model_dir)
    args_dir = os.path.join(args.root_dir, foldname)
    if not exists(save_dir):
        os.makedirs(save_dir)
    if not exists(model_dir):
        os.makedirs(model_dir)
    if not exists(args_dir):
        os.makedirs(args_dir)
    save_args(args, args_dir)

    for fold in range(args.fold_num):
        print("\n========== Fold " + str(fold + 1) + " ==========")

        ## get dataset
        train_dataset = dataset['kfold'][fold]['train']
        valid_dataset = dataset['kfold'][fold]['valid']
        print("Training on", str(len(train_dataset)), "examples, Validation on", str(len(valid_dataset)),
              "examples", " Testing on", str(len(dataset['test'])))

        ## get dataloader
        train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
        valid_loader = DataLoader(dataset=valid_dataset, batch_size=args.batch_size, shuffle=True)
        print("num of train loader: ", len(train_loader), "num of valid loader: ", len(valid_loader), "num of test loader: ", len(test_loader))

        ## model initialization
        model, optimizer, scheduler, criterion = model_init(args)
        model = model.to(args.device)
        print("device:", args.device)

        ## early stop
        early_stopping = EarlyStopping(patience=args.esPatience, verbose=True)

        ## train
        result_all, result_df, best_model = train_model(model, train_loader, valid_loader, test_loader, criterion,
                                                        optimizer, scheduler, early_stopping, args, fold+1,
                                                        foldname, save_dir, model_dir)
        result_allfold_df = pd.concat([result_allfold_df, result_df])
        result_allfold.update({'Fold' + str(fold + 1): result_all})


    # save result
    result_allfold_df.to_csv(os.path.join(save_dir, 'Foldall_result.csv'), sep=',')
    torch.save(result_allfold, os.path.join(save_dir, 'result_allfold.pt'))
    df_result_kfold, best_epoch = extract_kfold_best(result_allfold_df)
    df_result_kfold.to_csv(os.path.join(save_dir, 'result_kfold.txt'), sep='\t', float_format='%.2f', index=False)
    return result_allfold, result_allfold_df, best_epoch


def train_model(model, train_loader, valid_loader, test_loader, criterion, optimizer, scheduler, early_stopping, args,
                foldnum, foldname, save_dir, model_dir):

    best_auc = 0.0
    best_epoch = 0

    loss_train, loss_valid, loss_test = [], [], []
    acc_train, acc_valid, acc_test = [], [], []
    roc_train, roc_valid, roc_test = [], [], []
    f1_train, f1_valid, f1_test = [], [], []
    loss_steps_train, loss_steps_valid, loss_steps_test = [], [], []
    acc_steps_train, acc_steps_valid, acc_steps_test = [], [], []

    lr_epoch = []

    for epoch in range(args.epoch_num):
        ## train one epoch
        print('epoch:{:d}/{:d}'.format(epoch, args.epoch_num))
        print('*' * 20)
        train_loss, train_loss_steps, train_acc, train_acc_steps, train_auc_list, train_f1_list, train_predlist, train_labellist, train_scorelist \
            = train_one_epoch(model, train_loader, optimizer, criterion, args)
        print("training: {:.4f}, {:.4f}, {:.4f}".format(train_loss, train_acc, train_auc_list[0]["macro"]))
        valid_loss, valid_loss_steps, valid_acc, valid_acc_steps, valid_auc_list, valid_f1_list, valid_predlist, valid_labellist, valid_scorelist\
            = valid_one_epoch(model, valid_loader, criterion, args)
        print("validation: {:.4f}, {:.4f}, {:.4f}".format(valid_loss, valid_acc, valid_auc_list[0]["macro"]))
        test_loss, test_loss_steps, test_acc, test_acc_steps, test_auc_list, test_f1_list, test_predlist, test_labellist, test_scorelist \
            = valid_one_epoch(model, test_loader, criterion, args)
        print("testing: {:.4f}, {:.4f}, {:.4f}".format(test_loss, test_acc, test_auc_list[0]["macro"]))

        ## record evaluation indicators
        valid_auc = valid_auc_list[0]["macro"]

        loss_train.append(train_loss)
        acc_train.append(train_acc)
        roc_train.append(train_auc_list)
        f1_train.append(train_f1_list)
        loss_steps_train.append(train_loss_steps)
        acc_steps_train.append(train_acc_steps)

        loss_valid.append(valid_loss)
        acc_valid.append(valid_acc)
        roc_valid.append(valid_auc_list)
        f1_valid.append(valid_f1_list)
        loss_steps_valid.append(valid_loss_steps)
        acc_steps_valid.append(valid_acc_steps)

        loss_test.append(test_loss)
        acc_test.append(test_acc)
        roc_test.append(test_auc_list)
        f1_test.append(test_f1_list)
        loss_steps_test.append(test_loss_steps)
        acc_steps_test.append(test_acc_steps)

        lr_epoch.append(optimizer.param_groups[0]['lr'])
        lr_temp = optimizer.param_groups[0]['lr'] * 1000
        print("lr:{:.4f}".format(lr_temp))

        ## scheduler update
        if args.scheduler == 'cosine':
            scheduler.step()
        elif args.scheduler == 'reduce':
            scheduler.step(valid_loss)

        ## save best model
        if valid_auc > best_auc:
            best_auc = valid_auc
            best_epoch = epoch + 1
            best_model = model
            torch.save(best_model, os.path.join(model_dir, 'Fold' + str(foldnum) + '_best_model.pt'))

        ## early stopping
        early_stopping(valid_auc)
        if early_stopping.early_stop:
            print('Early stopping')
            break

    result_all = {
        'foldnum': [foldnum for _ in range(len(loss_train))],
        'epoch': [i+1 for i in range(len(loss_train))],
        'best_auc': [best_auc for _ in range(len(loss_train))],
        'best_epoch': [best_epoch for _ in range(len(loss_train))],
        'lr_epoch': lr_epoch,
        'loss_train': loss_train,
        'loss_valid': loss_valid,
        'loss_test': loss_test,
        'acc_train': acc_train,
        'acc_valid': acc_valid,
        'acc_test': acc_test,
        'roc_train': roc_train,
        'roc_valid': roc_valid,
        'roc_test': roc_test,
        'f1_train': f1_train,
        'f1_valid': f1_valid,
        'f1_test': f1_test,
        'loss_steps_train': loss_steps_train,
        'loss_steps_valid': loss_steps_valid,
        'loss_steps_test': loss_steps_test,
        'acc_steps_train': acc_steps_train,
        'acc_steps_valid': acc_steps_valid,
        'acc_steps_test': acc_steps_test
    }
    ## result dict -> dataframe save
    result_df = dict_all2df(result_all)
    print("Fold", str(foldnum), "Best epoch at", str(best_epoch), "Best macro auc", str(best_auc))

    result_df.to_csv(os.path.join(save_dir, 'Fold' + str(foldnum) + "_result.csv"), sep=',')
    return result_all, result_df, best_model


