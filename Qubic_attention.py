import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
import torch

class QubicAttention(nn.Module):
    """ Criss-Cross Attention Module"""
    def __init__(self,in_dim):
        super(QubicAttention,self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.query_conv_col = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.key_conv_col = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.value_conv_col = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.conv_s = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self,x):
        proj_query = self.query_conv(x)
        proj_key = self.key_conv(x)
        proj_value = self.value_conv(x)
        proj_query_col = self.query_conv_col(x)
        proj_key_col = self.key_conv_col(x)
        proj_value_col = self.value_conv_col(x)
        short = self.conv_s(x)

        # proj_value shape n * 1 * 25 * 1024  reshape to n*25*1024*1
        proj_query = proj_query.permute(0,2,3,1)
        proj_key = proj_key.permute(0,2,3,1)
        proj_value = proj_value.permute(0,2,3,1)

        ## row attention
        proj_query_row = proj_query.view((proj_query.shape[0]*proj_query.shape[1]),proj_query.shape[2],-1) # {n*25}*1024*1
        proj_key_row = proj_key.view((proj_key.shape[0] * proj_key.shape[1]), proj_key.shape[2], -1)  # {n*25}*1024*1
        proj_value_row = proj_value.view((proj_value.shape[0] * proj_value.shape[1]), proj_value.shape[2], -1)  # {n*25}*1024*1

        proj_query_row = proj_query_row.permute(0,2,1)
        energy_row = proj_key_row @ proj_query_row  # {n*25}*1024*1024
        attention_row = F.softmax(energy_row,-1)
        out_row = attention_row @ proj_value_row  #  {n*25}*1024*1

        out_row = out_row.view(x.shape[0],x.shape[2],x.shape[3],x.shape[1]) #n*25*1024*1
        out_row = out_row.permute(0,3,1,2)

        ## column attention
        proj_query_col = proj_query_col.permute(0, 2, 3, 1)
        proj_key_col = proj_key_col.permute(0, 2, 3, 1)
        proj_value_col = proj_value_col.permute(0, 2, 3, 1)

        proj_query_col = proj_query_col.view((proj_query_col.shape[0] * proj_query_col.shape[2]), proj_query_col.shape[1],-1)  # {n*1024}*25*1
        proj_key_col = proj_key_col.view((proj_key_col.shape[0] * proj_key_col.shape[2]), proj_key_col.shape[1], -1)
        proj_value_col = proj_value_col.view((proj_value_col.shape[0] * proj_value_col.shape[2]), proj_value_col.shape[1],-1)

        proj_query_col = proj_query_col.permute(0, 2, 1)
        energy_col = proj_key_col @ proj_query_col  # {n*1024}*25*25
        attention_col = F.softmax(energy_col, -1)
        out_col = attention_col @ proj_value_col

        out_col = out_col.view(x.shape[0],x.shape[3],x.shape[2],x.shape[1]) # n*1024*25*1
        out_col = out_col.permute(0,3,2,1)

        out = out_col + out_row + short
        # out = self.gamma*out + x

        return out

class RCCAModule(nn.Module):
    def __init__(self, in_channels, recurrence =2):
        super(RCCAModule, self).__init__()
        self.recurrence = recurrence
        self.cca = QubicAttention(in_channels)
        self.fc = nn.Linear(1 * 4096, 1024, bias=True)
        # self.conva = nn.Sequential(nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
        #                            nn.BatchNorm2d(in_channels))
        # self.convb = nn.Sequential(nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
        #                            nn.BatchNorm2d(in_channels))
        #
        # self.bottleneck = nn.Sequential(
        #     nn.Conv2d(in_channels + in_channels, in_channels, kernel_size=3, padding=1, dilation=1, bias=False),
        #     nn.BatchNorm2d(in_channels),
        #     nn.Dropout2d(0.1),
        # )
    def forward(self, x):
        # output = x
        output = F.relu(self.fc(x))
        # output = self.conva(output)
        for i in range(self.recurrence):
            output = self.cca(output)
        # output = self.convb(output)
        # output = self.bottleneck(torch.cat([self.fc(x), output], 1))
        output_mean = torch.mean(output,dim=2).squeeze(1)
        return output_mean

class QubicAttention_3d(nn.Module):
    """ Criss-Cross Attention Module"""
    def __init__(self,in_dim):
        super(QubicAttention_3d,self).__init__()
        self.q_conv_t = nn.Conv3d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.k_conv_t = nn.Conv3d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.v_cov_t = nn.Conv3d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)

        self.q_conv_h = nn.Conv3d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.k_conv_h = nn.Conv3d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.v_conv_h = nn.Conv3d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        self.q_conv_v = nn.Conv3d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.k_conv_v = nn.Conv3d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.v_conv_v = nn.Conv3d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        self.conv_s = nn.Conv3d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self,x):
        q_t = self.q_conv_t(x)
        k_t = self.k_conv_t(x)
        v_t = self.v_cov_t(x)

        q_h = self.q_conv_h(x)
        k_h = self.k_conv_h(x)
        v_h = self.v_conv_h(x)

        q_v = self.q_conv_v(x)
        k_v = self.k_conv_v(x)
        v_v = self.v_conv_v(x)

        short = self.conv_s(x)

        #  n*25*2048*7*7  reshape to n*25*7*7*2048
        #  n*2048*25*7*7  reshape to n*25*7*7*2048
        q_t = q_t.permute(0, 2, 3, 4, 1)
        k_t = k_t.permute(0, 2, 3, 4, 1)
        v_t = v_t.permute(0, 2, 3, 4, 1)
        q_h = q_h.permute(0, 2, 3, 4, 1)
        k_h = k_h.permute(0, 2, 3, 4, 1)
        v_h = v_h.permute(0, 2, 3, 4, 1)
        q_v = q_v.permute(0, 2, 3, 4, 1)
        k_v = k_v.permute(0, 2, 3, 4, 1)
        v_v = v_v.permute(0, 2, 3, 4, 1)

        ## temporal attention
        # {n*7*7}*25*2048
        #  x shape:: n*2048*25*7*7
        q_t = q_t.contiguous().view((x.shape[0] * x.shape[3] * x.shape[4]), x.shape[2],-1)
        k_t = k_t.contiguous().view((x.shape[0] * x.shape[3] * x.shape[4]), x.shape[2], -1)
        v_t = v_t.contiguous().view((x.shape[0] * x.shape[3] * x.shape[4]), x.shape[2], -1)

        q_t = q_t.permute(0,2,1)  #{n*7*7}*2048*25
        attention_t = F.softmax(k_t@q_t,-1) #{n*7*7}*25*25
        out_t = attention_t@v_t   #{n*7*7}*25*2048

        out_t = out_t.contiguous().view(x.shape[0],x.shape[3],x.shape[4],x.shape[2],x.shape[1]) # n*7*7*25*2048
        out_t = out_t.permute(0,4,3,1,2)  # n*25*2048*7*7

        ## horizontal attention
        #  x shape:: n*2048*25*7*7   q_h shape:: n*25*7*7*2048
        # target shape:: {n*25*7}*7*2048
        q_h = q_h.contiguous().view((x.shape[0] * x.shape[2] * x.shape[4]), x.shape[3], -1)
        k_h = k_h.contiguous().view((x.shape[0] * x.shape[2] * x.shape[4]), x.shape[3], -1)
        v_h = v_h.contiguous().view((x.shape[0] * x.shape[2] * x.shape[4]), x.shape[3], -1)

        q_h = q_h.permute(0, 2, 1)  # {n*25*7}*2048*7
        attention_h = F.softmax(k_h @ q_h, -1)  # {n*25*7}*7*7
        out_h = attention_h @ v_h  # {n*25*7}*7*2048

        out_h = out_h.contiguous().view(x.shape[0], x.shape[2], x.shape[4], x.shape[3], x.shape[1])  # n*25*7*7*2048
        out_h = out_h.permute(0, 4, 1, 3, 2)  # n*25*2048*7*7

        ## vertical attention
        #  x shape:: n*2048*25*7*7   q shape:: n*25*7*7*2048
        # target shape:: {n*25*7}*7*2048
        q_v = q_v.contiguous().view((x.shape[0] * x.shape[2] * x.shape[3]), x.shape[4], -1)
        k_v = k_v.contiguous().view((x.shape[0] * x.shape[2] * x.shape[3]), x.shape[4], -1)
        v_v = v_v.contiguous().view((x.shape[0] * x.shape[2] * x.shape[3]), x.shape[4], -1)

        q_v = q_v.permute(0, 2, 1)  # {n*25*7}*2048*7
        attention_v = F.softmax(k_v @ q_v, -1)  # {n*25*7}*7*7
        out_v = attention_v @ v_v  # {n*25*7}*7*2048

        out_v = out_v.contiguous().view(x.shape[0], x.shape[2], x.shape[3], x.shape[4], x.shape[1])  # n*25*7*7*2048
        out_v = out_v.permute(0, 4, 1, 2, 3)  # n*25*2048*7*7

        out = out_t + out_h + out_v + short  # n*25*2048*7*7
        return out

class RCCAModule_3d(nn.Module):
    def __init__(self, in_channels, recurrence =2):
        super(RCCAModule_3d, self).__init__()
        self.recurrence = recurrence
        self.cca = QubicAttention_3d(in_channels)

    def forward(self, x):
        output = x
        for i in range(self.recurrence):
            output = self.cca(output)
        output_mean = torch.mean(output,dim=[2,3,-1])
        return output_mean