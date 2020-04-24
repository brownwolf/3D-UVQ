#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 11:09:49 2018

@author: vhash
"""

import pickle
import h5py
import torch
import torch.utils.data as data
import pdb
import scipy.io as sio


class VideoDataset(data.Dataset):
    def __init__(self, feature_h5_paths ,label_paths):
        # pdb.set_trace()
        self.video_files = []
        self.video_cnt = []
        for path in feature_h5_paths:
            h5_file = h5py.File(path, 'r')
            self.video_files.append(h5_file)
            self.video_cnt.append(h5_file['feats'].shape[0])
        self.len = sum(self.video_cnt)
        self.videoset_cnt = len(self.video_cnt)
        self.labels = sio.loadmat(label_paths)['labels'] #45585*239

    def __getitem__(self, index):
        pos = 0
        for i in range(self.videoset_cnt):
            if index >= self.video_cnt[i]:
                index -= self.video_cnt[i]
            else:
                pos = i
                break
        video_feat = torch.from_numpy(self.video_files[pos]['feats'][index])
        labels = torch.from_numpy(self.labels[index].astype(float)) # 239
        labels_onehot = labels
        labels = torch.argmax(labels)
        return video_feat, index, labels, labels_onehot

    def __len__(self):
        return self.len


if __name__ == '__main__':
    file_paths=['/home/langruimin/BLSTM_pytorch/data/fcv/fcv_train_feats.h5']
    label_paths = "/home/langruimin/BLSTM_pytorch/data/fcv/fcv_train_labels.mat"
    trainset=VideoDataset(file_paths, label_paths)
    print(len(trainset))
    trainloader=data.DataLoader(trainset,batch_size=128,shuffle=True,num_workers=1,pin_memory=True)
    for i,(data, index, labels) in enumerate(trainloader):
        print(i, data[:,0,:].size())
        print(i, index)
        print(i, labels)