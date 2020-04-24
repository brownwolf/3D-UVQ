from __future__ import  absolute_import
import torch
import torch.nn as nn

import torch.autograd as autograd
import torch.cuda.comm as comm
import torch.nn.functional as F
from torch.autograd.function import once_differentiable
import time
import functools

import torch.utils.data

# from libs import InPlaceABN, InPlaceABNSync
# BatchNorm2d = functools.partial(InPlaceABNSync, activation='none')

import _ext

def _check_contiguous(*args):
    if not all([mod is None or mod.is_contiguous() for mod in args]):
        raise ValueError("Non-contiguous input")


class CA_Weight(autograd.Function):
    @staticmethod
    def forward(ctx, t, f):
        # Save context
        n, c, h, w = t.size()
        size = (n, h+w-1, h, w)
        weight = torch.zeros(size, dtype=t.dtype, layout=t.layout, device=t.device)

        _ext.ca_forward_cuda(t, f, weight)
        
        # Output
        ctx.save_for_backward(t, f)

        return weight

    @staticmethod
    @once_differentiable
    def backward(ctx, dw):
        t, f = ctx.saved_tensors

        dt = torch.zeros_like(t)
        df = torch.zeros_like(f)

        _ext.ca_backward_cuda(dw.contiguous(), t, f, dt, df)

        _check_contiguous(dt, df)

        return dt, df

class CA_Map(autograd.Function):
    @staticmethod
    def forward(ctx, weight, g):
        # Save context
        out = torch.zeros_like(g)
        _ext.ca_map_forward_cuda(weight, g, out)
        
        # Output
        ctx.save_for_backward(weight, g)

        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, dout):
        weight, g = ctx.saved_tensors

        dw = torch.zeros_like(weight)
        dg = torch.zeros_like(g)

        _ext.ca_map_backward_cuda(dout.contiguous(), weight, g, dw, dg)

        _check_contiguous(dw, dg)

        return dw, dg

ca_weight = CA_Weight.apply
ca_map = CA_Map.apply


class CrissCrossAttention(nn.Module):
    """ Criss-Cross Attention Module"""
    def __init__(self,in_dim):
        super(CrissCrossAttention,self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self,x):
        proj_query = self.query_conv(x)
        proj_key = self.key_conv(x)
        proj_value = self.value_conv(x)

        energy = ca_weight(proj_query, proj_key)
        attention = F.softmax(energy, 1)
        out = ca_map(attention, proj_value)
        out = self.gamma*out + x

        return out

class RCCAModule(nn.Module):
    def __init__(self):
        super(RCCAModule, self).__init__()
        self.cca = CrissCrossAttention(1)

    def forward(self, x, recurrence=2):
        output = x
        for i in range(recurrence):
            output = self.cca(output)
        return output




__all__ = ["CrissCrossAttention", "ca_weight", "ca_map"]

if __name__ == '__main__':
    train_file_paths = "/home/langruimin/BLSTM_pytorch/data/fcv/fcv_train_feats.h5"
    videoset = VideoDataset(train_file_paths)
    trainloader = torch.utils.data.DataLoader(videoset, batch_size=100, shuffle=True,
                                              num_workers=4, pin_memory=True)
    model = RCCAModule()
    for i, (data, index) in enumerate(trainloader):
        data = data.to(model_options.default_dtype)
        data = data.cuda()
        output = model(data)
        print(output.size())
