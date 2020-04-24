from __future__ import  absolute_import
import torch.nn as nn

import torch.autograd as autograd
import torch.nn.functional as F
from torch.autograd.function import once_differentiable
from libs import InPlaceABNSync

# from libs import InPlaceABN, InPlaceABNSync
# BatchNorm2d = functools.partial(InPlaceABNSync, activation='none')

from data import *
from cc_attention import _ext

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
        # self.fc = nn.Linear(1*4096, 1024, bias=True)
        # self.fc2 = nn.Linear(1024, 4096, bias=True)

    def forward(self,x):
        proj_query = self.query_conv(x)
        # proj_query = F.relu(self.fc(proj_query))
        proj_key = self.key_conv(x)
        # proj_key = F.relu(self.fc(proj_key))
        proj_value = self.value_conv(x)
        # proj_value = F.relu(self.fc(proj_value))

        energy = ca_weight(proj_query, proj_key)
        attention = F.softmax(energy, 1)
        out = ca_map(attention, proj_value)
        # out = self.fc2(out)
        out = self.gamma*out + x

        return out

class RCCAModule(nn.Module):
    def __init__(self, in_channels, out_channels, recurrence =2):
        super(RCCAModule, self).__init__()
        self.recurrence = recurrence
        self.fc = nn.Linear(1 * 4096, 1024, bias=True)
        # self.fc2 = nn.Linear(1024, 239, bias=True)
        inter_channels = (in_channels // 4) + 1
        self.conva = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   # InPlaceABNSync(inter_channels)
                                   nn.BatchNorm2d(inter_channels))
        self.cca = CrissCrossAttention(inter_channels)
        self.convb = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                   # InPlaceABNSync(inter_channels)
                                    nn.BatchNorm2d(inter_channels) )

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels + inter_channels, out_channels, kernel_size=3, padding=1, dilation=1, bias=False),
            # InPlaceABNSync(out_channels),
            nn.BatchNorm2d(inter_channels),
            nn.Dropout2d(0.1),
            # nn.Conv2d(1, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        )

    def forward(self, x):
        if x.shape[-1] == 1024:
            output = F.relu(x)
        else:
            output = F.relu(self.fc(x))
        output = self.conva(output)
        for i in range(self.recurrence):
            output = self.cca(output)
        output = self.convb(output)
        output = self.bottleneck(torch.cat([self.fc(x), output], 1))
        output_mean = torch.mean(output,dim=2).squeeze(1)
        # output_crossEntropy = self.fc2(F.relu(self.fc(output_mean)))
        # print("ouput size: ",output_mean.size())
        # exit()
        return output_mean

class CrissCrossAttention2(nn.Module):
    """ Criss-Cross Attention Module"""
    def __init__(self,in_dim):
        super(CrissCrossAttention2,self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim, kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim, kernel_size= 1)
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

class RCCAModule2(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes):
        super(RCCAModule2, self).__init__()
        inter_channels = (in_channels // 4) + 1
        self.conva = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   InPlaceABNSync(inter_channels))
        self.cca = CrissCrossAttention2(inter_channels)
        self.convb = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                   InPlaceABNSync(inter_channels))

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels+inter_channels, out_channels, kernel_size=3, padding=1, dilation=1, bias=False),
            InPlaceABNSync(out_channels),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
            )

    def forward(self, x, recurrence=2):
        output = self.conva(x)
        for i in range(recurrence):
            output = self.cca(output)
        output = self.convb(output)

        output = self.bottleneck(torch.cat([x, output], 1))
        return output

__all__ = ["CrissCrossAttention", "ca_weight", "ca_map"]

if __name__ == '__main__':
    train_file_paths = ['/home/langruimin/BLSTM_pytorch/data/fcv/fcv_train_feats.h5']
    label_paths = "/home/langruimin/BLSTM_pytorch/data/fcv/fcv_train_labels.mat"
    videoset = VideoDataset(train_file_paths,label_paths)
    trainloader = torch.utils.data.DataLoader(videoset, batch_size=1, shuffle=True,
                                              num_workers=4, pin_memory=True)
    model = RCCAModule(1,1,239)
    model = nn.DataParallel(model).cuda()
    for i, (data, index, _, _) in enumerate(trainloader):
        data = data.to(torch.float32)
        data = data.unsqueeze(1)
        data = data.cuda()
        print("data size: ", data.size())
        output = model(data)
        print(output.size())
