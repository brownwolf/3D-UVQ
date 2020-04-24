import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Parameter


def cos_diff(A, B):
    # [N, d] · [d, k] -> [N, k]
    distance = torch.mm(A, B.transpose(1, 0))
    # [k]
    Cm_square = torch.sum(B * B, 1).unsqueeze(0)
    # [N]
    Xm_square = torch.sum(A * A, 1).unsqueeze(1)

    # [N, k], l2 mod for all X and C
    mod = torch.sqrt(torch.mul(Xm_square, Cm_square))

    # [N, k] distances, larger distance means more similar
    return distance / mod

def l2_diff_with_norm(A, B):

    # norm to [-1, 1], smaller is closer
    diff = torch.sum(torch.pow(torch.unsqueeze(A,1)-B,2),2)
    # diff = tf.reduce_sum(tf.square(tf.expand_dims(A, 1) - B), 2)

    # mini = tf.reduce_min(diff, 1, keepdims=True)

    # maxi = tf.reduce_max(diff, 1, keepdims=True)
    maxi, _ = torch.max(diff, 1, keepdim=True)
    # [N, 1]

    # rangi = maxi - mini

    # norm = ((diff - mini) / rangi - 0.5) * 2

    norm = diff / maxi

    # norm = (tf.nn.l2_normalize(diff, axis=1) - 0.5) * 2

    return norm


class Quantization(nn.Module):
    def __init__(self, subLevel, subCenters, dim):
        super(Quantization, self).__init__()
        self._stackLevel = subLevel  # 4
        self._subCeters = subCenters  # 256
        self._dim = dim  # input feature size
        self.Codebook = Parameter(torch.Tensor(self._stackLevel, self._subCeters, self._dim))
        nn.init.xavier_normal(self.Codebook)
        # self.fc = nn.Linear(1 * 4096, 1024, bias=True)

    def forward(self, input):
        # input = self.fc(input)
        self.X = input
        residual = self.X
        D = residual.size()[-1]
        N = residual.size()[0]
        self.SoftDistortion = 0.
        self.HardDistortion = 0.
        self.JointCenter = 0.
        self.HardCode = []
        self.QSoft = torch.zeros([N, D], dtype=torch.float32).cuda()
        self.QHard = torch.zeros([N, D], dtype=torch.float32).cuda()

        for level in range(self._stackLevel):
            # [K, d]
            codes = self.Codebook[level]

            # distance = +1 * cos_diff(residual, codes)
            distance = -1 * l2_diff_with_norm(residual, codes)

            # [N, K] dot [K, D]
            soft = torch.mm(F.softmax(distance, 1), codes)

            code = torch.argmax(distance, dim=1)
            code_save = code.unsqueeze(1)
            # print("code size: ", code_save.size())

            self.HardCode.append(code_save)
            hard = codes[code]

            residual = residual - hard

            self.QSoft = self.QSoft + soft
            self.QHard = self.QHard + hard

            self.SoftDistortion += torch.mean(torch.sum((self.X - self.QSoft).pow(2), 1))
            self.HardDistortion += torch.mean(torch.sum((self.X - self.QHard).pow(2), 1))
        self.error = torch.mean(torch.sum((self.X - self.QHard).pow(2), 1))
        self.JointCenter = torch.mean((self.QSoft - self.QHard).pow(2))

        self.HardCode = torch.cat(self.HardCode, 1)
        # print("HardCode size: ", self.HardCode.size())
        # input()

        return self.QHard, self.QSoft, self.SoftDistortion, self.HardDistortion, self.JointCenter, self.error, self.HardCode


