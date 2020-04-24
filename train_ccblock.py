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
    model = RCCAModule(1,1,recurrence=2)

    model_quan = Quantization(12, 1024, model_options.dim)

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
    milestones = []
    # lr_schduler_C = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1, last_epoch=-1)
    lr_schduler_Q = torch.optim.lr_scheduler.MultiStepLR(optimizer2, milestones, gamma=0.6, last_epoch=-1)

    # triplet selector, triplet loss
    # selector = BatchHardTripletSelector()
    # triplet_loss = TripletLoss(margin=0.1)
    # selector = HardestNegativeTripletSelector(margin=0.1, cpu=False)

    '''
    selector = AllTripletSelector()
    triplet_loss = OnlineTripletLoss(margin=512, triplet_selector=selector)
    '''

    # corss_entroypyloss
    # criterion = nn.CrossEntropyLoss()

    # centers = np.load(options.centers_path)
    # centers = torch.Tensor(centers).cuda()

    # neighborLoss
    # '''
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
    neighbor = True
    neighbor_freq = 2
    print("##########start train############")
    trainloader = torch.utils.data.DataLoader(videoset, batch_size=model_options.train_batch_size, shuffle=True,num_workers=4, pin_memory=True)
    model.train()
    model_quan.train()

    neighbor_loss = 0.0
    for l in range(60):
        # lr_schduler_C.step(l)
        # milestones.append(l+2)
        # lr_schduler_Q.step(l)

        if neighbor == True:
            # training
            for i, (data, index, _, _) in enumerate(trainloader):
                data = data.to(model_options.default_dtype)
                data = data.unsqueeze(1)
                data = data.cuda()
                # if l > 0:
                #     print("data shape: ",data.shape)
                # cc_block
                output_ccblock_mean = torch.tanh(model(data))


                # triplet_loss
                # tri_loss, tri_num = triplet_loss(output_ccblock_mean, init_train_label[index])
                # print("triplets num: ",tri_num)

                # cross_entropy_loss
                # loss = criterion(output_crossEntropy,labels.cuda())

                # cluster_loss
                # center_loss, _ = cluster_loss(centers, output_ccblock_mean, init_train_label[index], margin=0.5)

                # quantization block
                Qhard, Qsoft, SoftDistortion, HardDistortion, JointCenter, error,_ = model_quan(output_ccblock_mean)
                Q_loss = 0.1 * SoftDistortion + HardDistortion + 0.1 * JointCenter

                optimizer2.zero_grad()
                Q_loss.backward(retain_graph=True)
                optimizer2.step()

                if l % neighbor_freq == 0:
                    # neighbor loss
                    similarity_select = torch.index_select(similarity, 0, index)
                    similarity_select = torch.index_select(similarity_select, 1, index).float().cuda()
                    neighbor_loss = torch.sum((torch.mm(output_ccblock_mean, output_ccblock_mean.transpose(0,1)) / 1024 - similarity_select).pow(2))

                    optimizer.zero_grad()
                    neighbor_loss.backward()
                    optimizer.step()

                error_ += error.item()
                loss_ += neighbor_loss.item()
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
        this_path = os.path.join(str(120), currenttime)
        options.records_save_path = os.path.join('records', this_path)
        options.model_save_path = os.path.join('model','fcvid', this_path)
        os.makedirs(options.records_save_path, exist_ok=True)
        os.makedirs(options.model_save_path, exist_ok=True)
    else:
        pass
    torch.set_default_dtype(options.default_dtype)
    train_ccblock(options)
