import torch
import torch.nn as nn
import torchsparse
import torchsparse.nn as spnn
from torchsparse.tensor import PointTensor
from torchsparse.utils import *
from tools.torchsparse_utils import *
import math
import torch.nn.functional as F


class BasicConvolutionBlock(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1, dilation=1):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(inc,
                        outc,
                        kernel_size=ks,
                        dilation=dilation,
                        stride=stride), spnn.BatchNorm(outc),
            spnn.ReLU(True))

    def forward(self, x):
        out = self.net(x)
        return out


class BasicDeconvolutionBlock(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(inc,
                        outc,
                        kernel_size=ks,
                        stride=stride,
                        transposed=True), spnn.BatchNorm(outc),
            spnn.ReLU(True))

    def forward(self, x):
        return self.net(x)


class ResidualBlock(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1, dilation=1):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(inc,
                        outc,
                        kernel_size=ks,
                        dilation=dilation,
                        stride=stride), spnn.BatchNorm(outc),
            spnn.ReLU(True),
            spnn.Conv3d(outc,
                        outc,
                        kernel_size=ks,
                        dilation=dilation,
                        stride=1), spnn.BatchNorm(outc))

        self.downsample = nn.Sequential() if (inc == outc and stride == 1) else \
            nn.Sequential(
                spnn.Conv3d(inc, outc, kernel_size=1, dilation=1, stride=stride),
                spnn.BatchNorm(outc)
            )

        self.relu = spnn.ReLU(True)
    def forward(self, x):
        out = self.relu(self.net(x) + self.downsample(x))
        return out

class AttentionLayer(nn.Module):
    def __init__(self, inc=64, outc=64):
        super(AttentionLayer, self).__init__()
        self.hidden_dim = inc
        self.wv = spnn.Conv3d(in_channels=inc, out_channels=outc, kernel_size=3, stride=1, dilation=1)
        self.wk = spnn.Conv3d(in_channels=inc, out_channels=outc, kernel_size=3, stride=1, dilation=1)
        self.wq = spnn.Conv3d(in_channels=inc, out_channels=outc, kernel_size=3, stride=1, dilation=1)

    def forward(self, hidden, query):

        q = self.wq(query).F
        k = self.wk(hidden).F
        v = self.wv(hidden).F

        k_t = k.T
        score = q@k_t
        score = score/math.sqrt(self.hidden_dim)
        score = F.softmax(score, dim = -1)

        weight = score @ v
        hidden.F = weight
        return hidden


class AttentionBlock2(nn.Module):
    def __init__(self, in_channels, pres=1, vres=1):
        super(AttentionBlock2, self).__init__()
        cs = [8, 16, 32, 24, 24] #8,16,32,24,24
        # cs = [int(rate * x) for x in cs]
        self.pres = pres
        self.vres = vres

        self.stem = nn.Sequential(
            spnn.Conv3d(in_channels, cs[0], kernel_size=3, stride=1), #24,8
            spnn.BatchNorm(cs[0]), spnn.ReLU(True)
        )

        self.downsample1 = nn.Sequential(
            BasicConvolutionBlock(cs[0], cs[0], ks=2, stride=2, dilation=1), #(8,8)
            ResidualBlock(cs[0], cs[1], ks=3, stride=1, dilation=1), #(8,16)
            ResidualBlock(cs[1], cs[1], ks=3, stride=1, dilation=1), #(16,16)
        )

        self.downsample2 = nn.Sequential(
            BasicConvolutionBlock(cs[1], cs[1], ks=2, stride=2, dilation=1), #(16,16)
            ResidualBlock(cs[1], cs[2], ks=3, stride=1, dilation=1), #(16,32)
            ResidualBlock(cs[2], cs[2], ks=3, stride=1, dilation=1), #(32,32)
        )

        self.upsample1 = nn.ModuleList([
            BasicDeconvolutionBlock(cs[2], cs[3], ks=2, stride=2), #32,24
            nn.Sequential(
                ResidualBlock(cs[3] + cs[1], cs[3], ks=3, stride=1,dilation=1), #24+16, 24
                ResidualBlock(cs[3], cs[3], ks=3, stride=1, dilation=1), #24, 24
            )
        ])

        self.upsample2 = nn.ModuleList([
            BasicDeconvolutionBlock(cs[3], cs[4], ks=2, stride=2), #24,24
            nn.Sequential(
                ResidualBlock(cs[4] + cs[0], cs[4], ks=3, stride=1,dilation=1), #24+8, 24
                ResidualBlock(cs[4], cs[4], ks=3, stride=1, dilation=1), #24, 24
            )
        ])

        self.point_transforms = nn.ModuleList([
            nn.Sequential(
                nn.Linear(cs[0], cs[2]),
                nn.BatchNorm1d(cs[2]),
                nn.ReLU(True),
            ),
            nn.Sequential(
                nn.Linear(cs[2], cs[4]),
                nn.BatchNorm1d(cs[4]),
                nn.ReLU(True),
            )
        ])

        self.attention = AttentionLayer(inc=32, outc=32)
        self.weight_initialization()


    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, hidden, query):
        '''

        :param h: PintTensor
        :param x: PintTensor
        :return: h.F: Tensor (N, C)
        '''
        hidden_v0 = initial_voxelize(hidden, self.pres, self.vres) #4096,24->3696,24
        query_v0 = initial_voxelize(query, self.pres, self.vres) #4096,24->3696,24

        hidden_vinit = self.stem(hidden_v0) #3696,24 -> 3696,8
        query_vinit = self.stem(query_v0) #3696,24 -> 3696,8

        hidden_pinit = voxel_to_point(hidden_vinit, hidden, nearest=False) #3696,8 4096,24 -> 4096,8
        query_pinit = voxel_to_point(query_vinit, query, nearest=False) #3696,8 4096,24 -> 4096,8

        hidden_l0 = point_to_voxel(hidden_vinit, hidden_pinit) #3696,8 4096,8 -> 3696,8
        query_l0 = point_to_voxel(query_vinit, query_pinit)  #3696,8 4096,8 -> 3696,8

        hidden_l1 = self.downsample1(hidden_l0) #3696,8 ->552,16
        query_l1 = self.downsample1(query_l0)  #3696,8 ->552,16

        hidden_l2 = self.downsample2(hidden_l1) #3696,16 ->552,32
        query_l2 = self.downsample2(query_l1)  #3696,16 ->552,32

        query_atten = self.attention(hidden_l2, query_l2) #552,32 ->552,32
        print(query_atten.F.shape)

        query_pbottom = voxel_to_point(query_atten, query_pinit) #552,32 ,4096,8-> 4096,32
        query_pbottom.F = query_pbottom.F + self.point_transforms[0](query_pinit.F) #4096,32 4096,32 -> 4096,32

        query_vbottom = point_to_voxel(query_atten, query_pbottom) #552,32  4096,32 ->552,32
        # if self.dropout:
        #     y3.F = self.dropout(y3.F)
        query_r1 = self.upsample1[0](query_vbottom) #552,32 -> 3696,24
        query_r1 = torchsparse.cat([query_r1, query_l1]) #3696,24, 3696,16-> 3696,40
        query_r1 = self.upsample1[1](query_r1) #3696,40 -> 3696,24

        query_r0 = self.upsample2[0](query_r1) #3696,24
        query_r0 = torchsparse.cat([query_r0, query_vinit]) #3696,24
        query_r0 = self.upsample2[1](query_r0) #3696,24
        query = voxel_to_point(query_r0, query_pbottom) #3696,24 4096,24->4096,24
        query.F = query.F + self.point_transforms[1](query_pbottom.F) #4096,24

        return query.F


class AttentionBlock1(nn.Module):
    def __init__(self, in_channels, pres=1, vres=1):
        super(AttentionBlock1, self).__init__()

        cs = [16, 32, 64, 48, 48]
        # cs = [int(rate * x) for x in cs] # 16,32,64,48,48
        self.pres = pres
        self.vres = vres

        self.stem = nn.Sequential(
            spnn.Conv3d(in_channels, cs[0], kernel_size=3, stride=1),  #48,16
            spnn.BatchNorm(cs[0]), spnn.ReLU(True)
        )

        self.downsample = nn.Sequential(
            BasicConvolutionBlock(cs[0], cs[0], ks=2, stride=2, dilation=1), #(16,16)
            ResidualBlock(cs[0], cs[1], ks=3, stride=1, dilation=1), #(16,32)
            ResidualBlock(cs[1], cs[1], ks=3, stride=1, dilation=1), #(32,32)
        )

        self.upsample = nn.ModuleList([
            BasicDeconvolutionBlock(cs[1], cs[0], ks=2, stride=2), #32,16
            nn.Sequential(
                ResidualBlock(cs[0] + cs[0], cs[4], ks=3, stride=1,dilation=1), #16+16, 48
                ResidualBlock(cs[4], cs[4], ks=3, stride=1, dilation=1), #48, 48
            )
        ])

        self.point_transforms = nn.ModuleList([
            nn.Sequential(
                nn.Linear(cs[0], cs[1]),
                nn.BatchNorm1d(cs[1]),
                nn.ReLU(True),
            ),
            nn.Sequential(
                nn.Linear(cs[1], cs[3]),
                nn.BatchNorm1d(cs[3]),
                nn.ReLU(True),
            )
        ])

        self.attention = AttentionLayer(inc=32, outc=32)
        self.weight_initialization()


    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, hidden, query):
        '''

        :param h: PintTensor
        :param x: PintTensor
        :return: h.F: Tensor (N, C)
        '''
        hidden_v0 = initial_voxelize(hidden, self.pres, self.vres) #4096,48->3696,48
        query_v0 = initial_voxelize(query, self.pres, self.vres) #4096,48->3696,48

        hidden_vinit = self.stem(hidden_v0) #3696,48 -> 3696,16
        query_vinit = self.stem(query_v0) #3696,48 -> 3696,16

        hidden_pinit = voxel_to_point(hidden_vinit, hidden, nearest=False) #3696,16 4096,48 -> 4096,16
        query_pinit = voxel_to_point(query_vinit, query, nearest=False) #3696,16 4096,48 -> 4096,16

        hidden_l0 = point_to_voxel(hidden_vinit, hidden_pinit) #3696,16 4096,16 -> 3696,16
        query_l0 = point_to_voxel(query_vinit, query_pinit)  #3696,16 4096,16 -> 3696,16

        hidden_l1 = self.downsample(hidden_l0) #3696,16 ->552,32
        query_l1 = self.downsample(query_l0)  #3696,16 ->552,32

        query_atten = self.attention(hidden_l1, query_l1) 
        print(query_atten.F.shape)

        query_pbottom = voxel_to_point(query_atten, query_pinit) #552,32 ,4096,16-> 4096,32
        query_pbottom.F = query_pbottom.F + self.point_transforms[0](query_pinit.F) #4096,32 4096,32 -> 4096,32

        query_vbottom = point_to_voxel(query_atten, query_pbottom) #552,32  4096,32 ->552,32
        # if self.dropout:
        #     y3.F = self.dropout(y3.F)
        query_r1 = self.upsample[0](query_vbottom) #552,32 -> 3696,16
        query_r1 = torchsparse.cat([query_r1, query_vinit]) #3696,16, 3696,16-> 3696,32
        query_r1 = self.upsample[1](query_r1) #3696,32 -> 3696,48

        query = voxel_to_point(query_r1, query_pbottom) #3696,48 4096,32->4096,48
        query.F = query.F + self.point_transforms[1](query_pbottom.F) #4096,48

        return query.F

class AttentionBlock0(nn.Module):
    def __init__(self, in_channels, pres=1, vres=1):
        super(AttentionBlock0, self).__init__()
        # self.convz = SConv3d(hidden_dim + input_dim, hidden_dim, pres, vres, 3)
        # self.convr = SConv3d(hidden_dim + input_dim, hidden_dim, pres, vres, 3)
        # self.convq = SConv3d(hidden_dim + input_dim, hidden_dim, pres, vres, 3)
        cs = [32, 64, 128, 96, 96]
        self.pres = pres
        self.vres = vres

        self.stem = nn.Sequential(
            spnn.Conv3d(in_channels, cs[0], kernel_size=3, stride=1),  #96,32
            spnn.BatchNorm(cs[0]), spnn.ReLU(True)
        )

        self.downsample = nn.Sequential(
            BasicConvolutionBlock(cs[0], cs[0], ks=2, stride=2, dilation=1), #(32,32)
            ResidualBlock(cs[0], cs[1], ks=3, stride=1, dilation=1), #(32,64)
            ResidualBlock(cs[1], cs[1], ks=3, stride=1, dilation=1), #(64,64)
        )

        self.upsample = nn.ModuleList([
            BasicDeconvolutionBlock(cs[1], cs[0], ks=2, stride=2), #64,32
            nn.Sequential(
                ResidualBlock(cs[0] + cs[0], cs[4], ks=3, stride=1,dilation=1), #32+32, 96
                ResidualBlock(cs[4], cs[4], ks=3, stride=1, dilation=1), #96, 96
            )
        ])

        self.point_transforms = nn.ModuleList([
            nn.Sequential(
                nn.Linear(cs[0], cs[1]), #32 64
                nn.BatchNorm1d(cs[1]),
                nn.ReLU(True),
            ),
            nn.Sequential(
                nn.Linear(cs[1], cs[3]), #64 96
                nn.BatchNorm1d(cs[3]),
                nn.ReLU(True),
            )
        ])

        self.attention = AttentionLayer(inc=64, outc=64)
        self.weight_initialization()


    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, hidden, query):
        '''

        :param h: PintTensor
        :param x: PintTensor
        :return: h.F: Tensor (N, C)
        '''


        hidden_v0 = initial_voxelize(hidden, self.pres, self.vres) #4096,96->3696,96
        query_v0 = initial_voxelize(query, self.pres, self.vres) #4096,96->3696,96

        hidden_vinit = self.stem(hidden_v0) #3696,96 -> 3696,32
        query_vinit = self.stem(query_v0) #3696,96 -> 3696,32

        hidden_pinit = voxel_to_point(hidden_vinit, hidden, nearest=False) #3696,32 4096,96 -> 4096,32
        query_pinit = voxel_to_point(query_vinit, query, nearest=False) #3696,32 4096,96 -> 4096,32

        hidden_l0 = point_to_voxel(hidden_vinit, hidden_pinit) #3696,32 4096,32 -> 3696,32
        query_l0 = point_to_voxel(query_vinit, query_pinit)  #3696,32 4096,32 -> 3696,32

        hidden_l1 = self.downsample(hidden_l0) #3696,32 ->552,64
        query_l1 = self.downsample(query_l0)  #3696,32 ->552,64

        query_atten = self.attention(hidden_l1, query_l1) 
        print(query_atten.F.shape)

        query_pbottom = voxel_to_point(query_atten, query_pinit) #552,64 ,4096,32-> 4096,64
        query_pbottom.F = query_pbottom.F + self.point_transforms[0](query_pinit.F) #4096,64 4096,32 -> 4096,64

        query_vbottom = point_to_voxel(query_atten, query_pbottom) #552,64  4096,64 ->552,64
        # if self.dropout:
        #     y3.F = self.dropout(y3.F)
        query_r1 = self.upsample[0](query_vbottom) #552,64 -> 3696,32
        query_r1 = torchsparse.cat([query_r1, query_vinit]) #3696,32, 3696,32-> 3696,64
        query_r1 = self.upsample[1](query_r1) #3696,64 -> 3696,96

        query = voxel_to_point(query_r1, query_pbottom) #3696,96 4096,64->4096,96
        query.F = query.F + self.point_transforms[1](query_pbottom.F) #4096,96

        return query.F