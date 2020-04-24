#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time

import numpy as np
import torch.utils.data

import args
import mAP
from RCCA import RCCAModule
from data import *
from encoder_Q import Quantization


def encode(model_options):
    file_path = [model_options.testset_path]
    testlabel_path = model_options.testset_label_path
    testset = VideoDataset(file_path, testlabel_path)
    testloader = torch.utils.data.DataLoader(testset, batch_size=model_options.test_batch_size,
                                             shuffle=False, num_workers=4, pin_memory=True)
    model = RCCAModule(1,1,recurrence=2)
    model_quan = Quantization(12, 1024, 1024)
    model.eval()
    model_quan.eval()
    print('Test set size: {0}'.format(len(testset)))

    print('Loading model params...')
    params_path = os.path.join(model_options.model_save_path, model_options.params_filename)
    Qparams_path = os.path.join(model_options.model_save_path, model_options.Qparams_filename)
    print("param path",params_path,Qparams_path)
    model.load_state_dict(torch.load(params_path))
    model_quan.load_state_dict((torch.load(Qparams_path)))
    model = model.cuda()
    model_quan = model_quan.cuda()
    print('load Done.')

    print("########encoding##########")
    res = []
    labels = []
    features = []
    Qhards = []
    for i, (data, index, testlabel, labels_onehot) in enumerate(testloader):
        data = data.to(model_options.default_dtype)
        data = data.unsqueeze(1)
        data = data.cuda()

        with torch.no_grad():
            output_ccblock = torch.tanh(model(data))
            Qhard, _, _, _, _, _, HardCode = model_quan(output_ccblock)
            res.append(HardCode.cpu())
            features.append(output_ccblock.cpu())
            labels.append(labels_onehot)
            Qhards.append(Qhard.cpu())
    features = np.concatenate(features, axis=0)
    print("features shape:",features.shape)
    labels = np.concatenate(labels, axis=0)
    print('Labels shape:', labels.shape)
    res = np.concatenate(res, axis=0)
    print("B shape: ", res.shape)
    Qhards = np.concatenate(Qhards,axis=0)
    print("Qhard shape:",Qhards.shape)
    print("C shape: ", model_quan.Codebook.size())

    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(int(time.time())))
    this_path = os.path.join("saved","fcvid",str(120),"ccnet",current_time)
    os.makedirs(this_path)

    # quantized = reconstruct(res,model_quan.Codebook.detach().cpu().numpy())
    # np.save(os.path.join(this_path, "quantized.npy"),quantized)

    # save B
    # np.save("./saved/fcvid/B.npy", res)
    np.save(os.path.join(this_path,"B.npy"),res)
    # save C
    # np.save("./saved/fcvid/C.npy", model_quan.Codebook.detach().cpu().numpy())
    np.save(os.path.join(this_path, "C.npy"), model_quan.Codebook.detach().cpu().numpy())
    # save labels
    # np.save("./saved/fcvid/Y.npy", labels)
    np.save(os.path.join(this_path, "Y.npy"), labels)
    # save features
    # np.save("./saved/fcvid/X.npy", features)
    np.save(os.path.join(this_path, "X.npy"), features)

    np.save(os.path.join(this_path, "Qhard.npy"), Qhards)
    print("########encode and save done##########")

def reconstruct(B, C):
    M = C.shape[0]
    D = C[0].shape[-1]
    q = np.zeros([B.shape[0], D])
    for i in range(M):
        c = C[i]
        q += c[B[:, i]]
    return q

def evaluation(projected_size, encode_time, r):
    print("########start evalution##########")
    abl= "ccnet"

    C = np.load("./saved/fcvid/{1}/{2}/{0}/C.npy".format(encode_time,projected_size,abl))
    B = np.load("./saved/fcvid/{1}/{2}/{0}/B.npy".format(encode_time,projected_size,abl))

    X = np.load("./saved/fcvid/{1}/{2}/{0}/X.npy".format(encode_time,projected_size,abl))

    Qhard = np.load("./saved/fcvid/{1}/{2}/{0}/Qhard.npy".format(encode_time,projected_size,abl))
    b_label = np.load("./saved/fcvid/{1}/{2}/{0}/Y.npy".format(encode_time,projected_size,abl))

    print("Quantizatoin error: %.6e" % np.mean(np.sum((X - reconstruct(B, C)) ** 2, 1)))
    x_norm=np.linalg.norm(X, ord=None, axis=None, keepdims=False)
    print("fea l2:",x_norm)

    '''
    print("X :",X[0])
    print("sum",np.sum(X[0]))
    quit()
    '''

    """ When using center loss or classification, use l2 eval"""
    # mAP.raw_l2(X, b_label, X, np.copy(b_label), R=r)
    # b_label = np.load("./saved/fcvid/{1}/{2}/{0}/Y.npy".format(encode_time,projected_size,abl))
    # mAP.raw_l2(Quan, b_label, X, np.copy(b_label), R=r)
    mAP.AQD_l2_single(X[6700][None,:], b_label[6700][None,:], C, B, np.copy(b_label), R=r)

    """ When using semantic loss or cosine norm, use cos eval """
    # mAP.raw(X, b_label, X, np.copy(b_label), model_options.r)
    # b_label = np.load("./saved/fcvid/{1}/{0}/Y.npy".format(encode_time,model_options.projected_size))
    # mAP.AQD(X, b_label, C, B, np.copy(b_label), R=model_options.r)

    print("########evalution done##########")

def test(projected_size):
    print("#########test########")
    encode_time = "2019-10-12 09:44:41"
    C = np.load("./saved/fcvid/{1}/{0}/C.npy".format(encode_time, projected_size))
    B = np.load("./saved/fcvid/{1}/{0}/B.npy".format(encode_time, projected_size))
    X = np.load("./saved/fcvid/{1}/{0}/X.npy".format(encode_time, projected_size))
    aver = np.mean(np.sum((X - reconstruct(B, C)) ** 2, 1))
    for i in range(B.shape[0]):
        if np.mean(np.sum((X[i] - reconstruct(B[i][np.newaxis,:], C)) ** 2, 1)) > aver:
            print("X::", X[i])
            print("B::", B[i])
            print("Quantizatoin error: %.6e" % np.mean(np.sum((X[i] - reconstruct(B[i][np.newaxis,:], C)) ** 2, 1)))

if __name__ == "__main__":
    options = args.BLSTM_model_options()
    options.model_save_path = 'model/fcvid/128/2020-01-17 13:27:27'
    projected_size = 128
    encode_time = "2020-01-17 10:31:53"

    Encode = False
    if Encode == True:
        encode(options)
    else:
        for r in [6]:
            evaluation(projected_size, encode_time, r)