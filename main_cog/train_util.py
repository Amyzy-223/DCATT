import os

import numpy as np
import torch
import torch.nn as nn

import torch.optim as optim
import torchvision.ops
from model import DCATT
from util import cal_roc, cal_steps_loss

def model_init(args):
    if args.modelname == 'DCATT':
        print('\nDCATT')
        model = DCATT(fusionFlag=args.fusionFlag,lenP=args.lenP, lenQ=args.lenQ, gcnFlag=args.gcnFlag,
                          tempFlag=args.tempFlag, timeProj=args.timeProj, dropout=args.dropout, readout=args.readout,
                          hidden_dim=args.hidden_dim, end_dim=args.end_dim, layer=args.layer, out_dim=2)

    ## optimizer
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=args.weight_decay)

    ## scheduler
    if args.scheduler == 'reduce':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5, cooldown=5)
    elif args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epoch_num)

    ## loss function
    criterion = args.criterion

    return model, optimizer, scheduler, criterion

def train_one_epoch(model, train_loader, optimizer, criterion, args):
    model.train()
    total_loss = 0.0
    total_loss_steps = [0.0 for q in range(args.lenQ)]
    total_correct = 0
    total_correct_steps = [0 for q in range(args.lenQ)]
    predlist = []
    labellist = []
    scorelist = []

    for i, data in enumerate(train_loader):
        label, label_1d, ecg, fnirsx, fnirsA1, fnirsA2, fnirsA3, fnirsA4 = data
        label = label.float().to(args.device)  # [batch_size, cate_nums, lenQ]
        label_1d = label_1d.float().to(args.device) # [batch_size, lenQ]
        ecg = ecg.float().to(args.device)  # [batch_size, ecg_feature_nums, lenP]
        fnirsx = fnirsx.float().to(args.device)  # [batch_size, fnirs_feature_nums, node_nums, lenP]
        fnirsA1 = fnirsA1.float().to(args.device)  # [batch_size, node_nums, node_nums, lenP]
        fnirsA2 = fnirsA2.float().to(args.device)
        fnirsA3 = fnirsA3.float().to(args.device)
        fnirsA4 = fnirsA4.float().to(args.device)
        As = [fnirsA1, fnirsA2, fnirsA3, fnirsA4]

        optimizer.zero_grad()
        outputs = model(fnirsx, As, ecg)  # [batch_size, cate_nums, lenQ]
        loss_steps, loss = cal_steps_loss(outputs, label, criterion)  # list (lenQ), float
        _, predictions = torch.max(outputs, 1)  # prediction -> sum.indices [batch_size, lenQ]
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * fnirsx.size(0)
        total_correct += torch.sum(predictions == label_1d.data)
        total_correct_cmpr = torch.sum(predictions == label_1d.data, dim=0)
        for q in range(args.lenQ):
            total_loss_steps[q] += (loss_steps[q].item() * fnirsx.size(0))
            total_correct_steps[q] += (total_correct_cmpr[q])
        predlist.extend(predictions.cpu().detach().numpy())
        labellist.extend(label_1d.data.cpu().detach().numpy())
        scorelist.extend(outputs.cpu().detach().numpy())

    epoch_loss = total_loss / len(train_loader.dataset)
    epoch_loss_steps = [loss / len(train_loader.dataset) for loss in total_loss_steps]
    epoch_acc = total_correct.double() / (len(train_loader.dataset) * args.lenQ)
    epoch_acc_steps = [correct.cpu().detach().numpy() / len(train_loader.dataset) for correct in total_correct_steps]
    labellist = np.array(labellist)  # [batch_size, lenQ]
    scorelist = np.array(scorelist)  # [batch_size, cate_nums, lenQ]
    predlist = np.array(predlist)  # [batch_size, lenQ]
    auc_list, f1_list = cal_roc(labellist, scorelist, predlist)

    return epoch_loss, epoch_loss_steps, epoch_acc.item(), epoch_acc_steps, auc_list, f1_list, predlist, labellist, scorelist

def valid_one_epoch(model, valid_loader, criterion, args):
    model.eval()
    total_loss = 0.0
    total_loss_steps = [0.0 for q in range(args.lenQ)]
    total_correct = 0
    total_correct_steps = [0 for q in range(args.lenQ)]
    predlist = []
    labellist = []
    scorelist = []

    for i, data in enumerate(valid_loader):
        with torch.no_grad():
            label, label_1d, ecg, fnirsx, fnirsA1, fnirsA2, fnirsA3, fnirsA4 = data
            label = label.float().to(args.device)  # [batch_size, cate_nums, lenQ]
            label_1d = label_1d.float().to(args.device)  # [batch_size, lenQ]
            ecg = ecg.float().to(args.device)  # [batch_size, ecg_feature_nums, lenP]
            fnirsx = fnirsx.float().to(args.device)  # [batch_size, fnirs_feature_nums, node_nums, lenP]
            fnirsA1 = fnirsA1.float().to(args.device)  # [batch_size, node_nums, node_nums, lenP]
            fnirsA2 = fnirsA2.float().to(args.device)
            fnirsA3 = fnirsA3.float().to(args.device)
            fnirsA4 = fnirsA4.float().to(args.device)
            As = [fnirsA1, fnirsA2, fnirsA3, fnirsA4]

            outputs = model(fnirsx, As, ecg)  # [batch_size, cate_nums, lenQ]
            loss_steps, loss = cal_steps_loss(outputs, label, criterion)  # list (lenQ), float
            _, predictions = torch.max(outputs, 1)  # prediction -> sum.indices [batch_size, lenQ]

            total_loss += loss.item() * fnirsx.size(0)
            total_correct += torch.sum(predictions == label_1d.data)
            total_correct_cmpr = torch.sum(predictions == label_1d.data, dim=0)
            for q in range(args.lenQ):
                total_loss_steps[q] += (loss_steps[q].item() * fnirsx.size(0))
                total_correct_steps[q] += (total_correct_cmpr[q])
            predlist.extend(predictions.cpu().detach().numpy())
            labellist.extend(label_1d.data.cpu().detach().numpy())
            scorelist.extend(outputs.cpu().detach().numpy())

    epoch_loss = total_loss / len(valid_loader.dataset)
    epoch_loss_steps = [loss / len(valid_loader.dataset) for loss in total_loss_steps]
    epoch_acc = total_correct.double() / (len(valid_loader.dataset) * args.lenQ)
    epoch_acc_steps = [correct.cpu().detach().numpy() / len(valid_loader.dataset) for correct in total_correct_steps]
    labellist = np.array(labellist)  # [batch_size, lenQ]
    scorelist = np.array(scorelist)  # [batch_size, cate_nums, lenQ]
    predlist = np.array(predlist)  # [batch_size, lenQ]
    auc_list, f1_list = cal_roc(labellist, scorelist, predlist)

    return epoch_loss, epoch_loss_steps, epoch_acc.item(), epoch_acc_steps, auc_list, f1_list, predlist, labellist, scorelist
