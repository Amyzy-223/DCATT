## TODO 还原EEGnet参数看效果

import os
import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from genericpath import exists

import torch.optim as optim
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import Dataset, DataLoader, Subset

from models.DCATT import DCATT, Conv2dWithConstraint, Conv1dWithConstraint

from util import cal_roc, dict_all2df, save_args, result_visualization, write_log, save_pkl, eval_visualization, \
    log_test_result, extract_kfold_best

class Engine():
    def __init__(self, args, dataset, foldname):
        self.result_allfold_df = pd.DataFrame()
        self.result_allfold = {}
        self.args = args
        self.dataset = dataset
        self.foldname = foldname
        self.test_loader = DataLoader(dataset=dataset['test'], batch_size=args.batch_size, shuffle=False)

        ## make essential dir
        self.args_dir = os.path.join(args.root_dir, foldname)
        self.save_dir = os.path.join(args.root_dir, foldname, args.result_dir)
        self.model_dir = os.path.join(args.root_dir, foldname, args.model_dir)
        self.plot_dir = os.path.join(args.root_dir, foldname, args.plot_dir)
        if not exists(self.save_dir):
            os.makedirs(self.save_dir)
        if not exists(self.model_dir):
            os.makedirs(self.model_dir)
        if not exists(self.args_dir):
            os.makedirs(self.args_dir)
        if not exists(self.plot_dir):
            os.makedirs(self.plot_dir)
        save_args(args, self.args_dir)

        self.test_dir = os.path.join(self.args.root_dir, self.foldname, 'test2')
        if not os.path.exists(self.test_dir):
            os.makedirs(self.test_dir)
        self.temp_dir = os.path.join(self.test_dir, 'temp')
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)


    def one_stage_run(self):
        # train all fold
        if self.args.SubTestFlag in ['LOSO', 'TRA']:
            index_list = np.arange(len(self.dataset['train']))
            train_index, valid_index = train_test_split(index_list, test_size=0.2, random_state=self.args.seed, shuffle=True)
            run_fold = RunOneFold(fold=1, stage=1, train_index=train_index, valid_index=valid_index, args=self.args,
                                  dataset=self.dataset, foldname=self.foldname, test_loader= self.test_loader,
                                  save_dir=self.save_dir, model_dir=self.model_dir, args_dir=self.args_dir)
            result_all, result_df, _, cate_num = run_fold.run()
            self.result_allfold_df = pd.concat([self.result_allfold_df, result_df])
            self.result_allfold.update({'Fold' + str(1): result_all})

        elif self.args.SubTestFlag in ['DEP', 'INS']:
            kfold_ = KFold(n_splits=self.args.kfold, shuffle=True, random_state=self.args.seed)
            kfold_index = kfold_.split(self.dataset['train'])
            for fold, (train_index, valid_index) in enumerate(kfold_index):
                print("\n========== Fold " + str(fold + 1) + " / " + str(self.args.kfold)+ " ==========")
                run_fold = RunOneFold(fold=fold+1, stage=1, train_index=train_index, valid_index=valid_index, args=self.args,
                                      dataset=self.dataset, foldname=self.foldname, test_loader=self.test_loader,
                                      save_dir=self.save_dir, model_dir=self.model_dir, args_dir=self.args_dir)
                result_all, result_df, _, cate_num = run_fold.run()
                self.result_allfold_df = pd.concat([self.result_allfold_df, result_df])
                self.result_allfold.update({'Fold' + str(fold+1): result_all})


        # save train result
        self.result_allfold_df.to_csv(os.path.join(self.save_dir, 'Foldall_result.csv'), sep=',')
        torch.save(self.result_allfold, os.path.join(self.save_dir, 'result_allfold.pt'))
        df_result_kfold, self.best_epoch = extract_kfold_best(self.result_allfold_df)
        df_result_kfold.to_csv(os.path.join(self.save_dir, 'result_kfold.txt'), sep='\t', float_format='%.2f',
                               index=False)

        ## train curve visualization
        result_visualization(self.result_allfold_df, self.plot_dir, cate_num, stage=1)

        ## log
        write_log(self.result_allfold_df, self.foldname, self.args)

    def two_stage_run(self):
        ## 2-stage retrain
        run_fold = RunOneFold(stage=2, args=self.args, best_epoch=self.best_epoch, dataset=self.dataset,
                              foldname=self.foldname, test_dir=self.test_dir, test_loader=self.test_loader)
        result_all, result_df, best_model, cate_num = run_fold.run()
        torch.save(result_all, os.path.join(self.test_dir, 'result_all.pt'))

        ## train curve visualization
        result_visualization(result_df, self.test_dir, cate_num, stage=2)

        ## result visualization roc confusion matrix tsne
        test_df_test, repo_dict_test = run_fold.pre_test_data(model=best_model, nameFlag='test', plotFlag=True)
        test_df_train, repo_dict_train = run_fold.pre_test_data(model=best_model, nameFlag='train', plotFlag=True)
        log_test_result(test_df_train, test_df_test, repo_dict_train, repo_dict_test, self.test_dir, self.args.root_dir, self.foldname,
                        'test_summary2')




class RunOneFold():
    def __init__(self, fold=1, stage=1, best_epoch=0, train_index=None, valid_index=None, args=None,
                 dataset=None, foldname=None, test_loader=None, save_dir=None, model_dir=None, args_dir=None, test_dir=None):
        self.stage = stage
        self.args = args
        self.dataset = dataset
        self.foldname = foldname
        self.test_loader = test_loader

        if stage == 1:
            self.fold = fold
            self.epoch = self.args.epoch_num
            self.save_dir = save_dir
            self.model_dir = model_dir
            self.args_dir = args_dir

            print('=' * 40 + 'STAGE 1'+ '=' * 40)
            ## get dataset
            self.train_dataset = Subset(self.dataset['train'], train_index)
            self.valid_dataset = Subset(self.dataset['train'], valid_index)
            print("Training on", str(len(self.train_dataset)), "examples, Validation on", str(len(self.valid_dataset)),
                  "examples", " Testing on", str(len(self.dataset['test'])))

            ## get dataloader
            self.train_loader = DataLoader(dataset=self.train_dataset, batch_size=self.args.batch_size, shuffle=True)
            self.valid_loader = DataLoader(dataset=self.valid_dataset, batch_size=self.args.batch_size, shuffle=True)
            print("num of train loader: ", len(self.train_loader), "num of valid loader: ", len(self.valid_loader),
                  "num of test loader: ", len(self.test_loader))

            ## early stop
            self.early_stopping = EarlyStopping(patience=self.args.esPatience, verbose=True)

        elif stage == 2:
            print('=' * 40 + 'STAGE 2' + '=' * 40)
            self.fold = 0
            self.epoch = best_epoch
            self.train_loader = DataLoader(dataset=dataset['train'], batch_size=args.batch_size, shuffle=True)
            self.test_dir = test_dir

        ## model init
        self.model_init()

    def model_init(self):
        ## dataset info
        if self.args.dataset in ['SEED_DCATT']:
            self.input_len = 4
            self.channel = 12
            self.input_dim = 5
            self.cate_num = 3
            self.freq = 200


        if self.args.SubTestFlag in ['DEP', 'LOSO', 'TRA']:
            self.drop = self.args.dropout
        else:
            self.drop = self.args.dropout/2

        ## model init
        if self.args.modelname == 'DCATT':
            print('\nDCATT')  # dataset SEED_12chan_5band
            self.model = DCATT(dropout=self.drop, lenP=self.input_len, lenQ=1, layer = self.args.layer,
                               node_num=self.channel, fnirs_dim=self.input_dim, out_dim=self.cate_num,
                               hidden_dim=self.args.hidden_dim, end_dim=self.args.end_dim)


        self.initialize_weights()
        self.model = self.model.to(self.args.device)


        print("device:", self.args.device)

        ## optimizer
        if self.args.optimizer == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=0.9,
                                       weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr, betas=(0.9, 0.99),
                                        weight_decay=self.args.weight_decay)

        ## scheduler
        if self.args.scheduler == 'reduce':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', factor=0.5,
                                                                        patience=5,
                                                                        cooldown=5)
        elif self.args.scheduler == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.args.epoch_num)
            # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.epoch)
        ## loss function
        self.criterion = self.args.criterion
        if self.args.criterion[0] == 'entropy':
            self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    def initialize_weights(self):
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, Conv2dWithConstraint) or isinstance(m, Conv1dWithConstraint):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def run(self):
        ## train
        result_all, result_df, best_model = self.train_model()
        return result_all, result_df, best_model, self.cate_num

    def train_model(self):

        best_auc = 0.0
        best_epoch = 0

        if self.stage == 1:
            loss_train, acc_train, roc_train, f1_train = [], [], [], []
            loss_valid, acc_valid, roc_valid, f1_valid = [], [], [], []
            loss_test, acc_test, roc_test, f1_test = [], [], [], []
        else:
            loss_train, acc_train, roc_train, f1_train = [], [], [], []
            loss_test, acc_test, roc_test, f1_test = [], [], [], []

        lr_epoch = []

        for epoch in range(self.epoch):
            ## train one epoch
            print('epoch:{:d}/{:d}'.format(epoch, self.epoch))
            print('*' * 20, 'Fold', str(self.fold))
            train_loss, train_acc, train_auc, train_f1, train_predlist, train_labellist, train_scorelist, train_labellist_onehot \
                = self.train_one_epoch(self.train_loader)
            print("training: {:.4f}, {:.4f}, {:.4f}".format(train_loss, train_acc, train_auc["macro"]))
            if self.stage == 1:
                valid_loss, valid_acc, valid_auc, valid_f1, valid_predlist, valid_labellist, valid_scorelist, valid_labellist_onehot \
                    = self.valid_one_epoch(self.valid_loader)
                print("validation: {:.4f}, {:.4f}, {:.4f}".format(valid_loss, valid_acc, valid_auc["macro"]))
            test_loss, test_acc, test_auc, test_f1, test_predlist, test_labellist, test_scorelist, test_labellist_onthot \
                = self.valid_one_epoch(self.test_loader)
            print("testing: {:.4f}, {:.4f}, {:.4f}".format(test_loss, test_acc, test_auc["macro"]))

            ## record evaluation indicators

            loss_train.append(train_loss)
            acc_train.append(train_acc)
            roc_train.append(train_auc)
            f1_train.append(train_f1)

            if self.stage == 1:
                valid_index = valid_auc["macro"]
                loss_valid.append(valid_loss)
                acc_valid.append(valid_acc)
                roc_valid.append(valid_auc)
                f1_valid.append(valid_f1)
            else:
                test_index = test_auc["macro"]

            loss_test.append(test_loss)
            acc_test.append(test_acc)
            roc_test.append(test_auc)
            f1_test.append(test_f1)

            lr_epoch.append(self.optimizer.param_groups[0]['lr'])
            lr_temp = self.optimizer.param_groups[0]['lr'] * 1000
            print("lr:{:.4f}".format(lr_temp))

            ## scheduler update
            if self.args.scheduler == 'cosine':
                self.scheduler.step()
            elif self.args.scheduler == 'reduce':
                if self.stage ==1:
                    self.scheduler.step(valid_loss)

            ## save best model
            if self.stage == 1:  # 根据valid指标选择最优模型
                if valid_index > best_auc:
                    best_auc = valid_index
                    best_epoch = epoch + 1
                    best_model = self.model
                    torch.save(best_model, os.path.join(self.model_dir, 'Fold' + str(self.fold) + '_best_model.pt'))

                ## early stopping
                self.early_stopping(valid_index)
                if self.early_stopping.early_stop:
                    print('Early stopping')
                    break

            else:  ## 最后一个epoch模型为最优模型
                if test_index > best_auc:
                    best_auc = test_index
                    best_epoch = epoch + 1
                if epoch == self.epoch - 1:
                    best_model = self.model
                    torch.save(best_model, os.path.join(self.test_dir, 'best_model.pt'))

        if self.stage == 1:
            result_all = {
                'foldnum': [self.fold for _ in range(len(loss_train))],
                'epoch': [i + 1 for i in range(len(loss_train))],
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
                'f1_test': f1_test
            }
            ## result dict -> dataframe save
            result_df = dict_all2df(result_all, stage=1)
            print("Fold", str(self.fold), "Best epoch at", str(best_epoch), "Best macro auc", str(best_auc))

            result_df.to_csv(os.path.join(self.save_dir, 'Fold' + str(self.fold) + "_result.csv"), sep=',')
        else:
            result_all = {
                'foldnum': [self.fold for _ in range(len(loss_train))],
                'epoch': [i + 1 for i in range(len(loss_train))],
                'best_auc': [best_auc for _ in range(len(loss_train))],
                'best_epoch': [best_epoch for _ in range(len(loss_train))],
                'lr_epoch': lr_epoch,
                'loss_train': loss_train,
                'loss_test': loss_test,
                'acc_train': acc_train,
                'acc_test': acc_test,
                'roc_train': roc_train,
                'roc_test': roc_test,
                'f1_train': f1_train,
                'f1_test': f1_test
            }
            ## result dict -> dataframe save
            result_df = dict_all2df(result_all, stage=2)
            print("Last auc", str(roc_test[-1]['macro']), "Best epoch at", str(best_epoch), "Best macro auc", str(best_auc))

            result_df.to_csv(os.path.join(self.test_dir, "retrain_result.csv"), sep=',')
        return result_all, result_df, best_model

    def train_one_epoch(self, train_loader):
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        predlist = []
        labellist = []
        scorelist = []
        labellist_onehot = []

        for i, data in enumerate(train_loader):
            # print('loader:', str(i), '/', len(train_loader))
            try:  # raw eeg data / eeg de feature, no edge features
                eeg, label_1d, label = data
                label = label.float().to(self.args.device)  # [batch_size, cate_nums]
                label_1d = label_1d.float().to(self.args.device)  # [batch_size]
                try:  # raw eeg data
                    eeg = eeg.float().to(self.args.device).permute(0, 2, 1)  # [batch_size, time_sample, channel] [64, 1125, 22]
                    # eeg = eeg[:, : self.input_len, :]  # 截断时间点，保证时序长度是 2^（layer）的整数倍
                except:  # EEG DE feature [batch_size, bands, channels, time_step]
                    eeg = eeg.float().to(self.args.device)
            except:
                eeg, corr, plv, label_1d, label = data
                eeg = eeg.float().to(self.args.device)
                corr = corr.float().to(self.args.device)
                plv = plv.float().to(self.args.device)
                label = label.float().to(self.args.device)
                label_1d = label_1d.float().to(self.args.device)

            self.optimizer.zero_grad()
            if 'DCATT' in self.args.modelname:  # DCATT
                outputs = self.model(eeg, [corr, plv]).squeeze()
            else:
                outputs = self.model(eeg).squeeze()  # [batch_size, cate_nums]
            loss = self.criterion(outputs, label)  # list (lenQ), float
            _, predictions = torch.max(outputs, 1)  # prediction -> sum.indices [batch_size, lenQ]
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * eeg.size(0)
            total_correct += torch.sum(predictions == label_1d.data)
            predlist.extend(predictions.cpu().detach().numpy())
            labellist.extend(label_1d.data.cpu().detach().numpy())
            labellist_onehot.extend(label.data.cpu().detach().numpy())
            scorelist.extend(outputs.cpu().detach().numpy())

        epoch_loss = total_loss / len(train_loader.dataset)
        epoch_acc = total_correct.double() / len(train_loader.dataset)

        labellist = np.array(labellist)  # [batch_size]
        labellist_onehot = np.array(labellist_onehot)  # [batch_size, cate_num]
        scorelist = np.array(scorelist)  # [batch_size, cate_nums]
        predlist = np.array(predlist)  # [batch_size]
        auc, f1 = cal_roc(labellist, scorelist, predlist, labellist_onehot)

        return epoch_loss, epoch_acc.item(), auc, f1, predlist, labellist, scorelist, labellist_onehot

    def valid_one_epoch(self, valid_loader):
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        predlist = []
        labellist = []
        scorelist = []
        labellist_onehot = []

        for i, data in enumerate(valid_loader):
            with torch.no_grad():
                try:  # raw eeg data / eeg de feature, no edge features
                    eeg, label_1d, label = data
                    label = label.float().to(self.args.device)  # [batch_size, cate_nums]
                    label_1d = label_1d.float().to(self.args.device)  # [batch_size]
                    try:  # raw eeg data
                        eeg = eeg.float().to(self.args.device).permute(0, 2,
                                                                       1)  # [batch_size, time_sample, channel] [64, 1125, 22]
                        # eeg = eeg[:, : self.input_len, :]  # 截断时间点，保证时序长度是 2^（layer）的整数倍
                    except:  # EEG DE feature [batch_size, bands, channels, time_step]
                        eeg = eeg.float().to(self.args.device)
                except:
                    eeg, corr, plv, label_1d, label = data
                    eeg = eeg.float().to(self.args.device)
                    corr = corr.float().to(self.args.device)
                    plv = plv.float().to(self.args.device)
                    label = label.float().to(self.args.device)
                    label_1d = label_1d.float().to(self.args.device)

                if 'DCATT' in self.args.modelname:  # DCATT
                    outputs = self.model(eeg, [corr, plv]).squeeze()
                else:
                    outputs = self.model(eeg).squeeze()  # [batch_size, cate_nums]
                loss = self.criterion(outputs, label)  # list (lenQ), float
                _, predictions = torch.max(outputs, 1)  # prediction -> sum.indices [batch_size, lenQ]

                total_loss += loss.item() * eeg.size(0)
                total_correct += torch.sum(predictions == label_1d.data)
                predlist.extend(predictions.cpu().detach().numpy())
                labellist.extend(label_1d.data.cpu().detach().numpy())
                labellist_onehot.extend(label.data.cpu().detach().numpy())
                scorelist.extend(outputs.cpu().detach().numpy())


        epoch_loss = total_loss / len(valid_loader.dataset)
        epoch_acc = total_correct.double() / len(valid_loader.dataset)

        labellist = np.array(labellist)  # [batch_size, lenQ]
        labellist_onehot = np.array(labellist_onehot)
        scorelist = np.array(scorelist)  # [batch_size, cate_nums, lenQ]
        predlist = np.array(predlist)  # [batch_size, lenQ]
        auc, f1 = cal_roc(labellist, scorelist, predlist, labellist_onehot)

        return epoch_loss, epoch_acc.item(), auc, f1, predlist, labellist, scorelist, labellist_onehot

    def valid_tsne(self, valid_loader):
        self.model.eval()
        labellist = []
        scorelist = []

        for i, data in enumerate(valid_loader):
            with torch.no_grad():
                try:  # raw eeg data / eeg de feature, no edge features
                    eeg, label_1d, label = data
                    label = label.float().to(self.args.device)  # [batch_size, cate_nums]
                    label_1d = label_1d.float().to(self.args.device)  # [batch_size]
                    try:  # raw eeg data
                        eeg = eeg.float().to(self.args.device).permute(0, 2,
                                                                       1)  # [batch_size, time_sample, channel] [64, 1125, 22]
                        # eeg = eeg[:, : self.input_len, :]  # 截断时间点，保证时序长度是 2^（layer）的整数倍
                    except:  # EEG DE feature [batch_size, bands, channels, time_step]
                        eeg = eeg.float().to(self.args.device)
                except:
                    eeg, corr, plv, label_1d, label = data
                    eeg = eeg.float().to(self.args.device)
                    corr = corr.float().to(self.args.device)
                    plv = plv.float().to(self.args.device)
                    label = label.float().to(self.args.device)
                    label_1d = label_1d.float().to(self.args.device)

                if 'DCATT' in self.args.modelname:  # DCATT
                    outputs = self.model(eeg, [corr, plv]).squeeze()
                else:
                    outputs = self.model(eeg).squeeze()  # [batch_size, cate_nums]
                labellist.extend(label_1d.data.cpu().detach().numpy())
                scorelist.extend(outputs.cpu().detach().numpy())

        labellist = np.array(labellist)  # [batch_size, lenQ]
        scorelist = np.array(scorelist)  # [batch_size, cate_nums, lenQ]

        return labellist, scorelist

    def pre_test_data(self, model, nameFlag, plotFlag):
        if nameFlag == 'train':
            loader = self.train_loader
        elif nameFlag == 'test':
            loader = self.test_loader
        ## 预测结果
        best_model = copy.deepcopy(model)
        self.model = best_model.to(self.args.device)
        loss, acc, auc, f1, predlist, labellist, scorelist, labellist_onehot = self.valid_one_epoch(loader)

        ## tsne降维结果
        self.model.last = torch.nn.Sequential()
        labellist_tsne, plotlist_tsne = self.valid_tsne(loader)
        print('tsne shape:', plotlist_tsne.shape)

        ## 保存 predlist labellist scorelist plotlist 画图所需数据
        plot_data_dict = {'predlist': predlist, 'labellist': labellist, 'scorelist': scorelist, 'labellist_onehot': labellist_onehot,
                          'plotlist_tsne': plotlist_tsne,  'labellist_tsne': labellist_tsne}
        save_pkl(plot_data_dict, os.path.join(self.test_dir, nameFlag + '_plot_data_dict.pkl'))

        ## 将评价指标转为df形式
        test_result = {'nameflag': nameFlag, 'loss': loss}
        test_result['acc'] = acc
        test_result['f1'] = f1
        for cate in auc:
            test_result['auc_' + str(cate)] = auc[cate]
        test_df = pd.DataFrame(test_result, index=[0])

        repo_dict = dict()
        if plotFlag:
            temp_dir = os.path.join(self.test_dir, 'temp')
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)
            repo_dict = eval_visualization(predlist, labellist, labellist_onehot, scorelist, plotlist_tsne, labellist_tsne, temp_dir, nameFlag)
        return test_df, repo_dict



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

