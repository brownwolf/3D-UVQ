#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from data import *
import torch
import torch.utils.data
import args
import os
import pickle
import datetime
from encoder_Q import Quantization
import time
import _pickle as pkl
import numpy as np
from RCCA import RCCAModule
from triplet_selector import BatchHardTripletSelector,TripletLoss, HardestNegativeTripletSelector, OnlineTripletLoss, AllTripletSelector
import torch.nn as nn
from center_loss import cluster_loss
from sklearn.cluster.k_means_ import KMeans,MiniBatchKMeans
from radam import RAdam
from mean_model import model_mean

def train_ccblock(model_options):
    # get train&valid datasets' paths
    if model_options.trainset_num > 1:
        train_file_paths = [model_options.trainset_path.format(i) for i in range(1, model_options.trainset_num + 1)]
    else:
        train_file_paths = [model_options.trainset_path]

    # load datasets
    print(train_file_paths)
    label_paths = "/home/langruimin/BLSTM_pytorch/data/fcv/fcv_train_labels.mat"
    videoset = VideoDataset(train_file_paths, label_paths)
    print(len(videoset))


    # create model
    model = RCCAModule(1,1)
    model_quan = Quantization(model_options.subLevel, model_options.subCenters, model_options.dim)

    params_path = os.path.join(model_options.model_save_path, model_options.params_filename)
    params_path_Q = os.path.join(model_options.model_save_path, model_options.Qparams_filename)
    if model_options.reload_params:
        print('Loading model params...')
        model.load_state_dict(torch.load(params_path))
        print('Done.')

    model = model.cuda()
    model_quan = model_quan.cuda()
    # optimizer
    optimizer = RAdam(
        model.parameters(),
        lr=1e-3,
        betas=(0.9, 0.999),
        weight_decay=1e-4
    )
    optimizer2 = RAdam(
        model_quan.parameters(),
        lr=1e-3, # 7e-6
        betas=(0.9, 0.999),
        weight_decay=1e-4
    )

    lr_C = ""
    lr_Q = ""
    # milestones = []
    # lr_schduler_C = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1, last_epoch=-1)
    # lr_schduler_Q = torch.optim.lr_scheduler.MultiStepLR(optimizer2, milestones, gamma=0.6, last_epoch=-1)


    selector = AllTripletSelector()
    triplet_loss = OnlineTripletLoss(margin=512, triplet_selector=selector)

    batch_idx = 1
    train_loss_rec = open(os.path.join(model_options.records_save_path, model_options.train_loss_filename), 'w')
    error_ = 0.
    loss_ = 0.
    num = 0
    print("##########start train############")
    trainloader = torch.utils.data.DataLoader(videoset, batch_size=9, shuffle=True,num_workers=4, pin_memory=True)
    model.train()
    model_quan.train()

    init_train_label = np.load("/home/langruimin/BLSTM_pytorch/data/fcv/init_train_labels.npy")

    for l in range(100):
        # lr_schduler_C.step(l)
        # milestones.append(l+2)
        # lr_schduler_Q.step(l)

        # training
        for i, (data, index, _, _) in enumerate(trainloader):
            data = data.to(model_options.default_dtype)
            data = data.unsqueeze(1)
            data = data.cuda()
            # cc_block
            output_ccblock_mean = torch.tanh(model(data))

            # quantization block
            Qhard, Qsoft, SoftDistortion, HardDistortion, JointCenter, error,_ = model_quan(output_ccblock_mean)
            Q_loss = 0.1 * SoftDistortion + HardDistortion + 0.1 * JointCenter

            tri_loss, tri_num = triplet_loss(output_ccblock_mean, init_train_label[index])

            optimizer2.zero_grad()
            Q_loss.backward(retain_graph=True)
            optimizer2.step()

            optimizer.zero_grad()
            tri_loss.backward()
            optimizer.step()

            error_ += error.item()
            loss_ += tri_loss.item()
            num += 1
            if batch_idx % model_options.disp_freq == 0:
                info = "epoch{0} Batch {1} loss:{2:.3f}  distortion:{3:.3f} " \
                    .format(l, batch_idx, loss_/ num, error_ / num)
                print(info)
                train_loss_rec.write(info + '\n')

            batch_idx += 1
        batch_idx = 0
        error_ = 0.
        loss_ = 0.
        num = 0

        if (l+1) % model_options.save_freq == 0:
            print('epoch: ', l ,'New best model. Saving model ...')
            torch.save(model.state_dict(), params_path)
            torch.save(model_quan.state_dict(), params_path_Q)

            for param_group in optimizer.param_groups:
                lr_C = param_group['lr']
            for param_group in optimizer2.param_groups:
                lr_Q = param_group['lr']
            record_inf ="saved model at epoch {0} lr_C:{1} lr_Q:{2}".format(l, lr_C, lr_Q)
            train_loss_rec.write(record_inf + '\n')
        print("##########epoch done##########")

    print('train done. Saving model ...')
    torch.save(model.state_dict(), params_path)
    torch.save(model_quan.state_dict(), params_path_Q)
    print("##########train done##########")

if __name__ == "__main__":
    new_model = True
    options = args.BLSTM_model_options()
    currenttime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(int(time.time())))
    if new_model:
        options.reload_params = False
        this_path = os.path.join(str(options.projected_size), "no_neighbor", currenttime)
        options.records_save_path = os.path.join('records', this_path)
        options.model_save_path = os.path.join('model', this_path)
        os.makedirs(options.records_save_path, exist_ok=True)
        os.makedirs(options.model_save_path, exist_ok=True)
    else:
        pass
    torch.set_default_dtype(options.default_dtype)
    train_ccblock(options)
