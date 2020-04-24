#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 18 10:01:59 2018

@author: vhash
"""
from data import *
import layers
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
# torch.set_default_tensor_type(torch.DoubleTensor)
# from cc_attention import RCCAModule

def train_BLSTM(model_options):
    # get train&valid datasets' paths
    if model_options.trainset_num > 1:
        train_file_paths = [model_options.trainset_path.format(i) for i in range(1, model_options.trainset_num+1)]
    else:
        train_file_paths = [model_options.trainset_path]
    if model_options.validset_num > 1:
        valid_file_paths = [model_options.validset_path.format(i) for i in range(1, model_options.validset_num+1)]
    else:
        valid_file_paths = [model_options.validset_path]
    
    # load datasets
    print(train_file_paths)
    videoset = VideoDataset(train_file_paths)
    print(len(videoset))
    if model_options.use_validset:
        trainset = videoset
        validset = VideoDataset(valid_file_paths)
        train_len = len(trainset)
        valid_len = len(validset)
        print("train_len:", train_len)
    
    # load model
    model = layers.PlayAndRewind(model_options.feature_size, model_options.projected_size, model_options.sequence_len)
    model_quan = Quantization(model_options.subLevel, model_options.subCenters, model_options.dim, model_options.train_batch_size)
    params_path = os.path.join(model_options.model_save_path, model_options.params_filename)
    if model_options.reload_params:
        print('Loading model params...')
        model.load_state_dict(torch.load(params_path))
        print('Done.')
    model = model.cuda()
    model_quan = model_quan.cuda()

    model_non_local = RCCAModule()
    model_non_local = model_non_local.cuda()

    # loss1=torch.nn.MSELoss(size_average=False,reduce=True)
    # loss2=torch.nn.MSELoss(size_average=False,reduce=True)
    # loss3=torch.nn.MSELoss(size_average=False,reduce=True)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=model_options.lrate,
        weight_decay=model_options.l2_decay,
        betas=(0.5, 0.999),
    )
    optimizer2 = torch.optim.Adam(
        model_quan.parameters(),
        lr=5e-3,
        betas=(0.5, 0.999),
    )

    optimizer3 = torch.optim.Adam(
        model_non_local.parameters(),
        lr = 5e-3,
    )

    # train model
    with open(os.path.join(model_options.model_save_path, model_options.options_filename), 'wb') as f:
        pickle.dump(model_options, f, -1)
    
    best_valid_loss = float('inf')
    batch_idx = 0
    train_loss_rec = open(os.path.join(model_options.records_save_path, model_options.train_loss_filename), 'w')
    valid_loss_rec = open(os.path.join(model_options.records_save_path, model_options.valid_loss_filename), 'w')

    # load the similarity matrix
    f = open("/home/langruimin/BLSTM_pytorch/data/fcv/SimilarityInfo/Sim_K1_10_K2_5_fcv.pkl", "rb")
    similarity = pkl.load(f)
    similarity = torch.ByteTensor(similarity.astype(np.uint8))
    print("sim matrix:", similarity)
    f.close()

    error_ = 0.
    neighbor_loss_ = 0.
    total_loss_ = 0.
    num = 0

    for l in range(50):
        # split train & valid set
        if not model_options.use_validset:
            train_len = int(len(videoset)*0.90)
            valid_len = len(videoset)-train_len
            trainset, validset = torch.utils.data.random_split(videoset, [train_len, valid_len])
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=model_options.train_batch_size, shuffle=True, num_workers=4, pin_memory=True)
        validloader = torch.utils.data.DataLoader(validset, batch_size=model_options.valid_batch_size, shuffle=False, num_workers=1, pin_memory=True)
        
        # training
        for i, (data, index) in enumerate(trainloader):
            data = data.to(model_options.default_dtype)
            data = data.cuda()

            # for_recon,back_recon,mean_recon=model(data)
            # for_loss=loss1(for_recon,data)
            # back_loss=loss2(back_recon,data)
            # mean_loss=200.*loss3(mean_recon,torch.mean(data,1))
            # total_loss=1/3.*for_loss+1/3.*back_loss+1/3.*mean_loss

            bhx, bhx_last = model(data)  # data 100*25*4096
            similarity_select = torch.index_select(similarity, 0, index)
            similarity_select = torch.index_select(similarity_select, 1, index).float().cuda()
            neighbor_loss = (torch.mm(bhx_last, bhx_last.transpose(0, 1)) / 256 - similarity_select).pow(2)
            '''
            for index_i in range(bhx_last.size()[0]):
                for index_j in range(bhx_last.size()[0]):
                    if index_i != index_j:
                        loss_ij = pow((torch.mm(bhx_last[index_i, :].unsqueeze(0), bhx_last[index_j, :].unsqueeze(1)) / 256 - int(similarity[index_i, index_j])), 2)
                        neighbor_loss += loss_ij.mean()
            neighbor_loss = neighbor_loss / (bhx_last.size()[0] ** 2)
            '''
            Qhard, Qsoft, SoftDistortion, HardDistortion, JointCenter, error = model_quan(bhx)  # bhx 100*256
            quan_loss = 0.1 * SoftDistortion + HardDistortion + 0.1 * JointCenter

            # total_loss = 0.5 * neighbor_loss + quan_loss
            total_loss = quan_loss

            # optimizer.zero_grad()
            optimizer2.zero_grad()
            total_loss.backward()
            # optimizer.step()
            optimizer2.step()

            error_ += error.item()
            neighbor_loss_ += neighbor_loss.item()
            total_loss_ += total_loss.item()
            num += 1

            if batch_idx % model_options.disp_freq == 0:
                info = "epoch{0} Batch {1} loss:{2:.2f}  error:{3:.2f} Neighbor_loss:{4:.2f}"\
                  .format(l, batch_idx, total_loss, error_/num, neighbor_loss)
                print(info)
                train_loss_rec.write(info+'\n')

            # validation
            if (batch_idx + 1) % model_options.valid_freq == 0:
                print("Validating ...")
                avg_total_loss = 0.
                for j, valid in enumerate(validloader):
                    valid = valid.to(model_options.default_dtype)
                    valid = valid.cuda()
                    with torch.no_grad():
                        bhx = model(valid)
                        Qhard, Qsoft = model_quan(bhx)
                        SoftDistortion = torch.mean((bhx - Qsoft)*(bhx - Qsoft))
                        HardDistortion = torch.mean((bhx - Qhard)*(bhx - Qhard))
                        JointCenter = torch.mean((Qsoft - Qhard)*(Qsoft - Qhard))
                        avg_total_loss = avg_total_loss + (SoftDistortion.item() + HardDistortion.item() + 0.1 * JointCenter.item())
                    del valid
                avg_total_loss = avg_total_loss / (valid_len * model_options.sequence_len)
                info = "Batch {0}\tValid Cost: {1}".format(batch_idx, avg_total_loss)
                print(info)
                train_loss_rec.write(info+'\n')
                valid_loss_rec.write(info+'\n')
                if avg_total_loss < best_valid_loss:
                    best_valid_loss = avg_total_loss
                    print('New best model. Saving model ...')
                    torch.save(model.state_dict(), params_path)
            batch_idx += 1
        batch_idx = 0
        error_ = 0.
        neighbor_loss_ = 0.
        total_loss_ = 0.
        num = 0


new_model = True
current_time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
options = args.BLSTM_model_options()
if new_model:
    options.reload_params = False
    this_path = os.path.join(str(options.projected_size), current_time)
    options.records_save_path = os.path.join('records', this_path)
    options.model_save_path = os.path.join('model', this_path)
    os.makedirs(options.records_save_path, exist_ok=True)
    os.makedirs(options.model_save_path, exist_ok=True)
else:
    pass
options.lrate = 0.0002
torch.set_default_dtype(options.default_dtype)
train_BLSTM(options)
