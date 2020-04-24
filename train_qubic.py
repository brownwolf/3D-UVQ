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
from Qubic_attention import RCCAModule
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
    model = RCCAModule(1,2)

    model_quan = Quantization(16, 256, 1024)

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
        lr=1e-4,
        betas=(0.9, 0.999),
        weight_decay=1e-4
    )
    optimizer2 = RAdam(
        model_quan.parameters(),
        lr=1e-3,
        betas=(0.9, 0.999),
        weight_decay=1e-4
    )

    lr_C= ''
    lr_Q= ''

     # load the similarity matrix
    print("+++++++++loading similarity+++++++++")
    f = open("/home/langruimin/BLSTM_pytorch/data/fcv/SimilarityInfo/Sim_K1_10_K2_5_fcv.pkl", "rb")
    similarity = pkl.load(f)
    similarity = torch.ByteTensor(similarity.astype(np.uint8))
    f.close()
    print("++++++++++similarity loaded+++++++")
    # '''

    batch_idx = 1
    train_loss_rec = open(os.path.join(model_options.records_save_path, model_options.train_loss_filename), 'w')
    error_ = 0.
    loss_ = 0.
    num = 0
    neighbor_num = 0
    neighbor = True
    neighbor_freq = 2
    total_batchs = len(videoset) // model_options.train_batch_size
    print("##########start train############")
    trainloader = torch.utils.data.DataLoader(videoset, batch_size=8, shuffle=True,num_workers=4, pin_memory=True)
    model.train()
    model_quan.train()

    neighbor_loss = 0.0
    for l in range(80):

        if neighbor == True:
            # training
            for i, (data, index, _, _) in enumerate(trainloader):
                data = data.to(model_options.default_dtype)
                data = data.unsqueeze(1)
                data = data.cuda()

                output_ccblock_mean = torch.tanh(model(data))

                # quantization block
                Qhard, Qsoft, SoftDistortion, HardDistortion, JointCenter, error,_ = model_quan(output_ccblock_mean)
                Q_loss = 0.1 * SoftDistortion + HardDistortion + 0.1 * JointCenter

                optimizer2.zero_grad()
                Q_loss.backward(retain_graph=True)
                optimizer2.step()

                # if batch_idx < total_batchs * 0.6:
                # if l % neighbor_freq == 0:
                    # neighbor loss
                similarity_select = torch.index_select(similarity, 0, index)
                similarity_select = torch.index_select(similarity_select, 1, index).float().cuda()
                neighbor_loss = torch.sum((torch.mm(output_ccblock_mean, output_ccblock_mean.transpose(0,1)) / output_ccblock_mean.shape[-1] - similarity_select).pow(2))

                loss_ += neighbor_loss.item()
                neighbor_num  += 1

                optimizer.zero_grad()
                neighbor_loss.backward()
                optimizer.step()

                error_ += error.item()
                num += 1
                if batch_idx % model_options.disp_freq == 0:
                    info = "epoch{0} Batch {1} loss:{2:.3f}  distortion:{3:.3f} " \
                        .format(l, batch_idx, loss_/ neighbor_num, error_ / num)
                    print(info)
                    train_loss_rec.write(info + '\n')

                batch_idx += 1
            batch_idx = 0
            error_ = 0.
            loss_ = 0.
            num = 0
            neighbor_num = 0

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
        this_path = os.path.join(str(128),"qubic2d",currenttime)
        options.records_save_path = os.path.join('records', this_path)
        options.model_save_path = os.path.join('model', this_path)
        os.makedirs(options.records_save_path, exist_ok=True)
        os.makedirs(options.model_save_path, exist_ok=True)
    else:
        pass
    torch.set_default_dtype(options.default_dtype)
    train_ccblock(options)
