import torch
import torch.nn as nn
import torchsparse
import torchsparse.nn as spnn
from torchsparse.tensor import PointTensor
from torchsparse.utils import *
from tools.torchsparse_utils import *
import math
import torch.nn.functional as F

__all__ = ['SPVCNN', 'SConv3d', 'ConvGRU']


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


class SPVCNN(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.dropout = kwargs['dropout']

        cr = kwargs.get('cr', 1.0)
        cs = [32, 64, 128, 96, 96]
        cs = [int(cr * x) for x in cs]

        if 'pres' in kwargs and 'vres' in kwargs:
            self.pres = kwargs['pres']
            self.vres = kwargs['vres']

        self.stem = nn.Sequential(
            spnn.Conv3d(kwargs['in_channels'], cs[0], kernel_size=3, stride=1),
            spnn.BatchNorm(cs[0]),
            spnn.ReLU(True)
        )

        self.stage1 = nn.Sequential(
            BasicConvolutionBlock(cs[0], cs[0], ks=2, stride=2, dilation=1),
            ResidualBlock(cs[0], cs[1], ks=3, stride=1, dilation=1),
            ResidualBlock(cs[1], cs[1], ks=3, stride=1, dilation=1),
        )

        self.stage2 = nn.Sequential(
            BasicConvolutionBlock(cs[1], cs[1], ks=2, stride=2, dilation=1),
            ResidualBlock(cs[1], cs[2], ks=3, stride=1, dilation=1),
            ResidualBlock(cs[2], cs[2], ks=3, stride=1, dilation=1),
        )

        self.up1 = nn.ModuleList([
            BasicDeconvolutionBlock(cs[2], cs[3], ks=2, stride=2), #128,96
            nn.Sequential(
                ResidualBlock(cs[3] + cs[1], cs[3], ks=3, stride=1,   #96+64,96
                              dilation=1),
                ResidualBlock(cs[3], cs[3], ks=3, stride=1, dilation=1), #96,96
            )
        ])

        self.up2 = nn.ModuleList([
            BasicDeconvolutionBlock(cs[3], cs[4], ks=2, stride=2),
            nn.Sequential(
                ResidualBlock(cs[4] + cs[0], cs[4], ks=3, stride=1,
                              dilation=1),
                ResidualBlock(cs[4], cs[4], ks=3, stride=1, dilation=1),
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

        self.weight_initialization()

        if self.dropout:
            self.dropout = nn.Dropout(0.3, True)

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, z):
        # x: SparseTensor z: PointTensor
        x0 = initial_voxelize(z, self.pres, self.vres) #4096,81->3696,81

        x0 = self.stem(x0) #3696,32
        z0 = voxel_to_point(x0, z, nearest=False)
        z0.F = z0.F #4096,32

        x1 = point_to_voxel(x0, z0) #3696,32
        x1 = self.stage1(x1) #552,64
        # x1 = self.wv(x1)
        x2 = self.stage2(x1) #76,128
        z1 = voxel_to_point(x2, z0)
        z1.F = z1.F + self.point_transforms[0](z0.F)

        y3 = point_to_voxel(x2, z1) #76,128
        if self.dropout:
            y3.F = self.dropout(y3.F)
        y3 = self.up1[0](y3) #552,96
        y3 = torchsparse.cat([y3, x1]) #552,160
        y3 = self.up1[1](y3) #552,96

        y4 = self.up2[0](y3) #3696,96
        y4 = torchsparse.cat([y4, x0]) #3696,128
        y4 = self.up2[1](y4) #3696,96
        z3 = voxel_to_point(y4, z1)
        z3.F = z3.F + self.point_transforms[1](z1.F) #4096,96

        return z3.F


class SConv3d(nn.Module):
    def __init__(self, inc, outc, pres, vres, ks=3, stride=1, dilation=1):
        super().__init__()
        self.net = spnn.Conv3d(inc,
                               outc,
                               kernel_size=ks,
                               dilation=dilation,
                               stride=stride)
        self.point_transforms = nn.Sequential(
            nn.Linear(inc, outc),
        )
        self.pres = pres
        self.vres = vres

    def forward(self, z):
        x = initial_voxelize(z, self.pres, self.vres)
        x = self.net(x)
        out = voxel_to_point(x, z, nearest=False)
        out.F = out.F + self.point_transforms(z.F)
        return out


class ConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192 + 128, pres=1, vres=1):
        super(ConvGRU, self).__init__()
        self.convz = SConv3d(hidden_dim + input_dim, hidden_dim, pres, vres, 3)
        self.convr = SConv3d(hidden_dim + input_dim, hidden_dim, pres, vres, 3)
        self.convq = SConv3d(hidden_dim + input_dim, hidden_dim, pres, vres, 3)

    def forward(self, h, x):
        '''

        :param h: PintTensor
        :param x: PintTensor
        :return: h.F: Tensor (N, C)
        '''
        print('GRU!')
        hx = PointTensor(torch.cat([h.F, x.F], dim=1), h.C)

        z = torch.sigmoid(self.convz(hx).F)
        r = torch.sigmoid(self.convr(hx).F)
        x.F = torch.cat([r * h.F, x.F], dim=1)
        q = torch.tanh(self.convq(x).F)

        h.F = (1 - z) * h.F + z * q
        return h.F

# class Attention(nn.Module):
#     def __init__(self, hidden_dim=128, input_dim=192 + 128, pres=1, vres=1):
#         super(Attention, self).__init__()
#         # self.convz = SConv3d(hidden_dim + input_dim, hidden_dim, pres, vres, 3)
#         # self.convr = SConv3d(hidden_dim + input_dim, hidden_dim, pres, vres, 3)
#         # self.convq = SConv3d(hidden_dim + input_dim, hidden_dim, pres, vres, 3)
#         self.hidden_dim = hidden_dim
#         self.wv = SConv3d(input_dim, hidden_dim, pres, vres, 3)
#         self.wk = SConv3d(input_dim, hidden_dim, pres, vres, 3)
#         self.wq = SConv3d(input_dim, hidden_dim, pres, vres, 3)

#     def forward(self, hidden, query):
#         '''

#         :param h: PintTensor
#         :param x: PintTensor
#         :return: h.F: Tensor (N, C)
#         '''
#         q = self.wq(query).F
#         k = self.wk(hidden).F
#         v = self.wv(hidden).F
#         k_t = k.T
#         score = (q@k_t)/math.sqrt(self.hidden_dim)
#         score = torch.softmax(score)
#         weight = score @ v
#         return weight

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


# class AttentionBlock2(nn.Module):
#     def __init__(self, in_channels, rate, pres=1, vres=1):
#         super(AttentionBlock2, self).__init__()
#         # self.convz = SConv3d(hidden_dim + input_dim, hidden_dim, pres, vres, 3)
#         # self.convr = SConv3d(hidden_dim + input_dim, hidden_dim, pres, vres, 3)
#         # self.convq = SConv3d(hidden_dim + input_dim, hidden_dim, pres, vres, 3)
#         cs = [32, 64, 128, 96, 96] #8,16,32,24,24
#         cs = [int(rate * x) for x in cs]
#         self.pres = pres
#         self.vres = vres

#         self.stem = nn.Sequential(
#             spnn.Conv3d(in_channels, cs[0], kernel_size=3, stride=1), #24,8
#             spnn.BatchNorm(cs[0]), spnn.ReLU(True)
#         )

#         self.downsample1 = nn.Sequential(
#             BasicConvolutionBlock(cs[0], cs[0], ks=2, stride=2, dilation=1), #(8,8)
#             ResidualBlock(cs[0], cs[1], ks=3, stride=1, dilation=1), #(8,16)
#             ResidualBlock(cs[1], cs[1], ks=3, stride=1, dilation=1), #(16,16)
#         )

#         self.downsample2 = nn.Sequential(
#             BasicConvolutionBlock(cs[1], cs[1], ks=2, stride=2, dilation=1), #(16,16)
#             ResidualBlock(cs[1], cs[3], ks=3, stride=1, dilation=1), #(16,24)
#             ResidualBlock(cs[3], cs[3], ks=3, stride=1, dilation=1), #(24,24)
#         )

#         self.upsample1 = nn.ModuleList([
#             BasicDeconvolutionBlock(cs[2], cs[3], ks=2, stride=2), #32,24
#             nn.Sequential(
#                 ResidualBlock(cs[3] + cs[1], cs[3], ks=3, stride=1,dilation=1), #24+16, 24
#                 ResidualBlock(cs[3], cs[3], ks=3, stride=1, dilation=1), #24, 24
#             )
#         ])

#         self.upsample2 = nn.ModuleList([
#             BasicDeconvolutionBlock(cs[3], cs[4], ks=2, stride=2), #24,24
#             nn.Sequential(
#                 ResidualBlock(cs[4] + cs[0], cs[4], ks=3, stride=1,dilation=1), #24+8, 24
#                 ResidualBlock(cs[4], cs[4], ks=3, stride=1, dilation=1), #24, 24
#             )
#         ])

#         self.point_transforms = nn.ModuleList([
#             nn.Sequential(
#                 nn.Linear(cs[0], cs[3]),
#                 nn.BatchNorm1d(cs[3]),
#                 nn.ReLU(True),
#             ),
#             nn.Sequential(
#                 nn.Linear(cs[2], cs[4]),
#                 nn.BatchNorm1d(cs[4]),
#                 nn.ReLU(True),
#             )
#         ])

#         self.attention = AttentionLayer(inc=24, outc=24)
#         self.weight_initialization()


#     def weight_initialization(self):
#         for m in self.modules():
#             if isinstance(m, nn.BatchNorm1d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)

#     def forward(self, hidden, query):
#         '''

#         :param h: PintTensor
#         :param x: PintTensor
#         :return: h.F: Tensor (N, C)
#         '''
#         hidden_v0 = initial_voxelize(hidden, self.pres, self.vres) #4096,24->3696,24
#         query_v0 = initial_voxelize(query, self.pres, self.vres) #4096,24->3696,24

#         hidden_vinit = self.stem(hidden_v0) #3696,24 -> 3696,8
#         query_vinit = self.stem(query_v0) #3696,24 -> 3696,8

#         hidden_pinit = voxel_to_point(hidden_vinit, hidden, nearest=False) #3696,8 4096,24 -> 4096,8
#         query_pinit = voxel_to_point(query_vinit, query, nearest=False) #3696,8 4096,24 -> 4096,8

#         hidden_l0 = point_to_voxel(hidden_vinit, hidden_pinit) #3696,8 4096,8 -> 3696,8
#         query_l0 = point_to_voxel(query_vinit, query_pinit)  #3696,8 4096,8 -> 3696,8

#         hidden_l1 = self.downsample1(hidden_l0) #3696,8 ->552,16
#         query_l1 = self.downsample1(query_l0)  #3696,8 ->552,16

#         hidden_l2 = self.downsample2(hidden_l1) #3696,16 ->552,24
#         query_l2 = self.downsample2(query_l1)  #3696,16 ->552,24

#         query_atten = self.attention(hidden_l2, query_l2) #552,24 ->552,24
#         print(query_atten.F.shape)

#         hidden_pbottom = voxel_to_point(query_atten, hidden_pinit) #552,24 ,4096,8-> 4096,24
#         hidden_pbottom.F = hidden_pbottom.F + self.point_transforms[0](query_pinit.F) #4096,24 4096,24 -> 4096,24
#         print(hidden_pbottom.F.shape)

#         query_vbottom = point_to_voxel(query_atten, query_pbottom) #552,32  4096,32 ->552,32
#         # if self.dropout:
#         #     y3.F = self.dropout(y3.F)
#         query_r1 = self.upsample1[0](query_vbottom) #552,32 -> 3696,24
#         query_r1 = torchsparse.cat([query_r1, query_l1]) #3696,24, 3696,16-> 3696,40
#         query_r1 = self.upsample1[1](query_r1) #3696,40 -> 3696,24

#         query_r0 = self.upsample2[0](query_r1) #3696,24
#         query_r0 = torchsparse.cat([query_r0, query_vinit]) #3696,24
#         query_r0 = self.upsample2[1](query_r0) #3696,24
#         query = voxel_to_point(query_r0, query_pbottom) #3696,24 4096,24->4096,24
#         query.F = query.F + self.point_transforms[1](query_pbottom.F) #4096,24

#         return hidden_pbottom.F

# class AttentionBlock2(nn.Module):
#     def __init__(self, in_channels, rate, pres=1, vres=1):
#         super(AttentionBlock2, self).__init__()
#         # self.convz = SConv3d(hidden_dim + input_dim, hidden_dim, pres, vres, 3)
#         # self.convr = SConv3d(hidden_dim + input_dim, hidden_dim, pres, vres, 3)
#         # self.convq = SConv3d(hidden_dim + input_dim, hidden_dim, pres, vres, 3)
#         cs = [32, 64, 128, 96, 96] #8,16,32,24,24
#         cs = [int(rate * x) for x in cs]
#         self.pres = pres
#         self.vres = vres

#         self.stem = nn.Sequential(
#             spnn.Conv3d(in_channels, cs[0], kernel_size=3, stride=1), #24,8
#             spnn.BatchNorm(cs[0]), spnn.ReLU(True)
#         )

#         self.downsample1 = nn.Sequential(
#             BasicConvolutionBlock(cs[0], cs[0], ks=2, stride=2, dilation=1), #(8,8)
#             ResidualBlock(cs[0], cs[1], ks=3, stride=1, dilation=1), #(8,16)
#             ResidualBlock(cs[1], cs[1], ks=3, stride=1, dilation=1), #(16,16)
#         )

#         self.downsample2 = nn.Sequential(
#             BasicConvolutionBlock(cs[1], cs[1], ks=2, stride=2, dilation=1), #(16,16)
#             ResidualBlock(cs[1], cs[2], ks=3, stride=1, dilation=1), #(16,32)
#             ResidualBlock(cs[2], cs[2], ks=3, stride=1, dilation=1), #(32,32)
#         )

#         self.upsample1 = nn.ModuleList([
#             BasicDeconvolutionBlock(cs[2], cs[3], ks=2, stride=2), #32,24
#             nn.Sequential(
#                 ResidualBlock(cs[3] + cs[1], cs[3], ks=3, stride=1,dilation=1), #24+16, 24
#                 ResidualBlock(cs[3], cs[3], ks=3, stride=1, dilation=1), #24, 24
#             )
#         ])

#         self.upsample2 = nn.ModuleList([
#             BasicDeconvolutionBlock(cs[3], cs[4], ks=2, stride=2), #24,24
#             nn.Sequential(
#                 ResidualBlock(cs[4] + cs[0], cs[4], ks=3, stride=1,dilation=1), #24+8, 24
#                 ResidualBlock(cs[4], cs[4], ks=3, stride=1, dilation=1), #24, 24
#             )
#         ])

#         self.point_transforms = nn.ModuleList([
#             nn.Sequential(
#                 nn.Linear(cs[0], cs[2]),
#                 nn.BatchNorm1d(cs[2]),
#                 nn.ReLU(True),
#             ),
#             nn.Sequential(
#                 nn.Linear(cs[2], cs[4]),
#                 nn.BatchNorm1d(cs[4]),
#                 nn.ReLU(True),
#             )
#         ])

#         self.attention = AttentionLayer(inc=32, outc=32)
#         self.weight_initialization()


#     def weight_initialization(self):
#         for m in self.modules():
#             if isinstance(m, nn.BatchNorm1d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)

#     def forward(self, hidden, query):
#         '''

#         :param h: PintTensor
#         :param x: PintTensor
#         :return: h.F: Tensor (N, C)
#         '''
#         hidden_v0 = initial_voxelize(hidden, self.pres, self.vres) #4096,24->3696,24
#         query_v0 = initial_voxelize(query, self.pres, self.vres) #4096,24->3696,24

#         hidden_vinit = self.stem(hidden_v0) #3696,24 -> 3696,8
#         query_vinit = self.stem(query_v0) #3696,24 -> 3696,8

#         hidden_pinit = voxel_to_point(hidden_vinit, hidden, nearest=False) #3696,8 4096,24 -> 4096,8
#         query_pinit = voxel_to_point(query_vinit, query, nearest=False) #3696,8 4096,24 -> 4096,8

#         hidden_l0 = point_to_voxel(hidden_vinit, hidden_pinit) #3696,8 4096,8 -> 3696,8
#         query_l0 = point_to_voxel(query_vinit, query_pinit)  #3696,8 4096,8 -> 3696,8

#         hidden_l1 = self.downsample1(hidden_l0) #3696,8 ->552,16
#         query_l1 = self.downsample1(query_l0)  #3696,8 ->552,16

#         hidden_l2 = self.downsample2(hidden_l1) #3696,16 ->552,32
#         query_l2 = self.downsample2(query_l1)  #3696,16 ->552,32

#         query_atten = self.attention(hidden_l2, query_l2) #552,32 ->552,32
#         print(query_atten.F.shape)

#         query_pbottom = voxel_to_point(query_atten, query_pinit) #552,32 ,4096,8-> 4096,32
#         query_pbottom.F = query_pbottom.F + self.point_transforms[0](query_pinit.F) #4096,32 4096,32 -> 4096,32

#         query_vbottom = point_to_voxel(query_atten, query_pbottom) #552,32  4096,32 ->552,32
#         # if self.dropout:
#         #     y3.F = self.dropout(y3.F)
#         query_r1 = self.upsample1[0](query_vbottom) #552,32 -> 3696,24
#         query_r1 = torchsparse.cat([query_r1, query_l1]) #3696,24, 3696,16-> 3696,40
#         query_r1 = self.upsample1[1](query_r1) #3696,40 -> 3696,24

#         query_r0 = self.upsample2[0](query_r1) #3696,24
#         query_r0 = torchsparse.cat([query_r0, query_vinit]) #3696,24
#         query_r0 = self.upsample2[1](query_r0) #3696,24
#         query = voxel_to_point(query_r0, query_pbottom) #3696,24 4096,24->4096,24
#         query.F = query.F + self.point_transforms[1](query_pbottom.F) #4096,24

#         return query.F

# class AttentionBlock1(nn.Module):
#     def __init__(self, in_channels, rate, pres=1, vres=1):
#         super(AttentionBlock1, self).__init__()

#         cs = [32, 64, 128, 96, 96]
#         cs = [int(rate * x) for x in cs] # 16,32,64,48,48
#         self.pres = pres
#         self.vres = vres

#         self.stem = nn.Sequential(
#             spnn.Conv3d(in_channels, cs[0], kernel_size=3, stride=1),  #48,16
#             spnn.BatchNorm(cs[0]), spnn.ReLU(True)
#         )

#         self.downsample = nn.Sequential(
#             BasicConvolutionBlock(cs[0], cs[0], ks=2, stride=2, dilation=1), #(16,16)
#             ResidualBlock(cs[0], cs[3], ks=3, stride=1, dilation=1), #(16,48)
#             ResidualBlock(cs[3], cs[3], ks=3, stride=1, dilation=1), #(48,48)
#         )

#         # self.upsample = nn.ModuleList([
#         #     BasicDeconvolutionBlock(cs[1], cs[0], ks=2, stride=2), #32,16
#         #     nn.Sequential(
#         #         ResidualBlock(cs[0] + cs[0], cs[4], ks=3, stride=1,dilation=1), #16+16, 48
#         #         ResidualBlock(cs[4], cs[4], ks=3, stride=1, dilation=1), #48, 48
#         #     )
#         # ])

#         self.point_transforms = nn.ModuleList([
#             nn.Sequential(
#                 nn.Linear(cs[0], cs[3]),
#                 nn.BatchNorm1d(cs[3]),
#                 nn.ReLU(True),
#             ),
#             nn.Sequential(
#                 nn.Linear(cs[1], cs[3]),
#                 nn.BatchNorm1d(cs[3]),
#                 nn.ReLU(True),
#             )
#         ])

#         self.attention = AttentionLayer(inc=48, outc=48)
#         self.weight_initialization()


#     def weight_initialization(self):
#         for m in self.modules():
#             if isinstance(m, nn.BatchNorm1d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)

#     def forward(self, hidden, query):
#         '''

#         :param h: PintTensor
#         :param x: PintTensor
#         :return: h.F: Tensor (N, C)
#         '''
#         hidden_v0 = initial_voxelize(hidden, self.pres, self.vres) #4096,48->3696,48
#         query_v0 = initial_voxelize(query, self.pres, self.vres) #4096,48->3696,48

#         hidden_vinit = self.stem(hidden_v0) #3696,48 -> 3696,16
#         query_vinit = self.stem(query_v0) #3696,48 -> 3696,16

#         hidden_pinit = voxel_to_point(hidden_vinit, hidden, nearest=False) #3696,16 4096,48 -> 4096,16
#         query_pinit = voxel_to_point(query_vinit, query, nearest=False) #3696,16 4096,48 -> 4096,16

#         hidden_l0 = point_to_voxel(hidden_vinit, hidden_pinit) #3696,16 4096,16 -> 3696,16
#         query_l0 = point_to_voxel(query_vinit, query_pinit)  #3696,16 4096,16 -> 3696,16

#         hidden_l1 = self.downsample(hidden_l0) #3696,16 ->552,48
#         query_l1 = self.downsample(query_l0)  #3696,16 ->552,48

#         hidden_atten = self.attention(hidden_l1, query_l1) 
#         print(hidden_atten.F.shape)

#         hidden_pbottom = voxel_to_point(hidden_atten, hidden_pinit) #552,48 ,4096,16-> 4096,48
#         hidden_pbottom.F = hidden_pbottom.F + self.point_transforms[0](hidden_pinit.F) #4096,48 4096,48 -> 4096,48
#         print(hidden_pbottom.F.shape)

#         # query_vbottom = point_to_voxel(query_atten, query_pbottom) #552,32  4096,32 ->552,32
#         # # if self.dropout:
#         # #     y3.F = self.dropout(y3.F)
#         # query_r1 = self.upsample[0](query_vbottom) #552,32 -> 3696,16
#         # query_r1 = torchsparse.cat([query_r1, query_vinit]) #3696,16, 3696,16-> 3696,32
#         # query_r1 = self.upsample[1](query_r1) #3696,32 -> 3696,48

#         # query = voxel_to_point(query_r1, query_pbottom) #3696,48 4096,32->4096,48
#         # query.F = query.F + self.point_transforms[1](query_pbottom.F) #4096,48

#         return hidden_pbottom.F

class AttentionBlock1(nn.Module):
    def __init__(self, in_channels, rate, pres=1, vres=1):
        super(AttentionBlock1, self).__init__()

        cs = [32, 64, 128, 96, 96]
        cs = [int(rate * x) for x in cs] # 16,32,64,48,48
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

# class AttentionBlock0(nn.Module):
#     def __init__(self, in_channels, rate, pres=1, vres=1):
#         super(AttentionBlock0, self).__init__()
#         # self.convz = SConv3d(hidden_dim + input_dim, hidden_dim, pres, vres, 3)
#         # self.convr = SConv3d(hidden_dim + input_dim, hidden_dim, pres, vres, 3)
#         # self.convq = SConv3d(hidden_dim + input_dim, hidden_dim, pres, vres, 3)
#         cs = [32, 64, 128, 96, 96]
#         cs = [int(rate * x) for x in cs]
#         self.pres = pres
#         self.vres = vres

#         self.stem = nn.Sequential(
#             spnn.Conv3d(in_channels, cs[0], kernel_size=3, stride=1),  #96,32
#             spnn.BatchNorm(cs[0]), spnn.ReLU(True)
#         )

#         self.downsample = nn.Sequential(
#             BasicConvolutionBlock(cs[0], cs[0], ks=2, stride=2, dilation=1), #(32,32)
#             ResidualBlock(cs[0], cs[1], ks=3, stride=1, dilation=1), #(32,64)
#             ResidualBlock(cs[1], cs[1], ks=3, stride=1, dilation=1), #(64,64)
#         )

#         self.upsample = nn.ModuleList([
#             BasicDeconvolutionBlock(cs[1], cs[0], ks=2, stride=2), #64,32
#             nn.Sequential(
#                 ResidualBlock(cs[0] + cs[0], cs[4], ks=3, stride=1,dilation=1), #32+32, 96
#                 ResidualBlock(cs[4], cs[4], ks=3, stride=1, dilation=1), #96, 96
#             )
#         ])

#         self.point_transforms = nn.ModuleList([
#             nn.Sequential(
#                 nn.Linear(cs[0], cs[1]), #32 64
#                 nn.BatchNorm1d(cs[1]),
#                 nn.ReLU(True),
#             ),
#             nn.Sequential(
#                 nn.Linear(cs[1], cs[3]), #64 96
#                 nn.BatchNorm1d(cs[3]),
#                 nn.ReLU(True),
#             )
#         ])

#         self.attention = AttentionLayer(inc=64, outc=64)
#         self.weight_initialization()


#     def weight_initialization(self):
#         for m in self.modules():
#             if isinstance(m, nn.BatchNorm1d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)

#     def forward(self, hidden, query):
#         '''

#         :param h: PintTensor
#         :param x: PintTensor
#         :return: h.F: Tensor (N, C)
#         '''


        # hidden_v0 = initial_voxelize(hidden, self.pres, self.vres) #4096,96->3696,96
        # query_v0 = initial_voxelize(query, self.pres, self.vres) #4096,96->3696,96

        # hidden_vinit = self.stem(hidden_v0) #3696,96 -> 3696,32
        # query_vinit = self.stem(query_v0) #3696,96 -> 3696,32

        # hidden_pinit = voxel_to_point(hidden_vinit, hidden, nearest=False) #3696,32 4096,96 -> 4096,32
        # query_pinit = voxel_to_point(query_vinit, query, nearest=False) #3696,32 4096,96 -> 4096,32

        # hidden_l0 = point_to_voxel(hidden_vinit, hidden_pinit) #3696,32 4096,32 -> 3696,32
        # query_l0 = point_to_voxel(query_vinit, query_pinit)  #3696,32 4096,32 -> 3696,32

        # hidden_l1 = self.downsample(hidden_l0) #3696,32 ->552,64
        # query_l1 = self.downsample(query_l0)  #3696,32 ->552,64

        # query_atten = self.attention(hidden_l1, query_l1) 
        # print(query_atten.F.shape)

        # query_pbottom = voxel_to_point(query_atten, query_pinit) #552,64 ,4096,32-> 4096,64
        # query_pbottom.F = query_pbottom.F + self.point_transforms[0](query_pinit.F) #4096,64 4096,32 -> 4096,64

        # query_vbottom = point_to_voxel(query_atten, query_pbottom) #552,64  4096,64 ->552,64
        # # if self.dropout:
        # #     y3.F = self.dropout(y3.F)
        # query_r1 = self.upsample[0](query_vbottom) #552,64 -> 3696,32
        # query_r1 = torchsparse.cat([query_r1, query_vinit]) #3696,32, 3696,32-> 3696,64
        # query_r1 = self.upsample[1](query_r1) #3696,64 -> 3696,96

        # query = voxel_to_point(query_r1, query_pbottom) #3696,96 4096,64->4096,96
        # query.F = query.F + self.point_transforms[1](query_pbottom.F) #4096,96

        # return query.F

class AttentionBlock0(nn.Module):
    def __init__(self, in_channels, rate, pres=1, vres=1):
        super(AttentionBlock0, self).__init__()
        # self.convz = SConv3d(hidden_dim + input_dim, hidden_dim, pres, vres, 3)
        # self.convr = SConv3d(hidden_dim + input_dim, hidden_dim, pres, vres, 3)
        # self.convq = SConv3d(hidden_dim + input_dim, hidden_dim, pres, vres, 3)
        cs = [96, 96, 128, 96, 96]
        cs = [int(rate * x) for x in cs]
        self.pres = pres
        self.vres = vres

        self.stem = nn.Sequential(
            spnn.Conv3d(in_channels, cs[0], kernel_size=3, stride=1),  #96,96
            spnn.BatchNorm(cs[0]), spnn.ReLU(True)
        )

        # self.downsample = nn.Sequential(
        #     BasicConvolutionBlock(cs[0], cs[0], ks=2, stride=2, dilation=1), #(32,32)
        #     ResidualBlock(cs[0], cs[1], ks=3, stride=1, dilation=1), #(32,64)
        #     ResidualBlock(cs[1], cs[1], ks=3, stride=1, dilation=1), #(64,64)
        # )

        # self.upsample = nn.ModuleList([
        #     BasicDeconvolutionBlock(cs[1], cs[0], ks=2, stride=2), #64,32
        #     nn.Sequential(
        #         ResidualBlock(cs[0] + cs[0], cs[4], ks=3, stride=1,dilation=1), #32+32, 96
        #         ResidualBlock(cs[4], cs[4], ks=3, stride=1, dilation=1), #96, 96
        #     )
        # ])

        self.point_transforms = nn.ModuleList([
            nn.Sequential(
                nn.Linear(cs[0], cs[1]), #96 96
                nn.BatchNorm1d(cs[1]),
                nn.ReLU(True),
            )
        ])

        self.attention = AttentionLayer(inc=96, outc=96)
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

        hidden_vinit = self.stem(hidden_v0) #3696,96 -> 3696,96
        query_vinit = self.stem(query_v0) #3696,96 -> 3696,96

        query_pinit = voxel_to_point(query_vinit, query, nearest=False) #3696,96 4096,96 -> 4096,96
        hidden_pinit = voxel_to_point(hidden_vinit, hidden, nearest=False)

        query_atten = self.attention(hidden_vinit, query_vinit) 
        print(query_atten.F.shape)


        # hidden_l0 = point_to_voxel(hidden_vinit, hidden_pinit) #3696,32 4096,32 -> 3696,32
        # query_l0 = point_to_voxel(query_vinit, query_pinit)  #3696,32 4096,32 -> 3696,32

        # hidden_l1 = self.downsample(hidden_l0) #3696,32 ->552,64
        # query_l1 = self.downsample(query_l0)  #3696,32 ->552,64

        # query_atten = self.attention(hidden_l1, query_l1) 
        # print(query_atten.F.shape)

        query_pbottom = voxel_to_point(query_atten, query_pinit) #3096,96 ,4096,96-> 4096,96
        query_pbottom.F = query_pbottom.F + self.point_transforms[0](query_pinit.F) #4096,96 4096,96 -> 4096,96
        print(query_pbottom.F.shape)
        # query_vbottom = point_to_voxel(query_atten, query_pbottom) #552,64  4096,64 ->552,64
        # if self.dropout:
        #     y3.F = self.dropout(y3.F)
        # query_r1 = self.upsample[0](query_vbottom) #552,64 -> 3696,32
        # query_r1 = torchsparse.cat([query_r1, query_vinit]) #3696,32, 3696,32-> 3696,64
        # query_r1 = self.upsample[1](query_r1) #3696,64 -> 3696,96

        # query = voxel_to_point(query_r1, query_pbottom) #3696,96 4096,64->4096,96
        # query.F = query.F + self.point_transforms[1](query_pbottom.F) #4096,96

        return query_pbottom.F

class AttenSeq2Seq(nn.Module):
    def __init__(self, hidden_dim=96, input_dim=96, pres=1, vres=1):
        super(AttenSeq2Seq, self).__init__()
        self.attention = SConv3d(input_dim + hidden_dim, hidden_dim, pres, vres, 3)
        self.v = SConv3d(hidden_dim, 1, pres, vres, 3)
        self.wh = SConv3d(input_dim, hidden_dim, pres, vres, 3)
        self.wq = SConv3d(input_dim, hidden_dim, pres, vres, 3)
        self.update = SConv3d(input_dim + hidden_dim, hidden_dim, pres, vres, 3)

    def forward(self, hidden, query):
        '''

        :param h: PintTensor
        :param x: PintTensor
        :return: h.F: Tensor (N, C)
        
        '''
        print('attention seq2seq!')
        query = self.wq(query)
        hidden = self.wh(hidden)
        hx = PointTensor(torch.cat([hidden.F,query.F], dim=1), query.C) #4096,192
        energy = torch.tanh(self.attention(hx).F) #4096,96
        energy = PointTensor(energy, query.C)  #4096,96
        attention = self.v(energy).F  #4096,1
        # attention = F.softmax(energy,dim=0)
        context = attention * hidden.F #4096,96
        query.F = torch.cat([query.F, context], dim=1) #4096,192
        output = torch.tanh(self.update(query).F) #4096,96
        return output

class LinearLayer(nn.Module):
    def __init__(self, hidden_dim=96, input_dim=96, pres=1, vres=1):
        super(LinearLayer, self).__init__()
        self.pres = pres
        self.vres = vres
        # self.attention = SConv3d(input_dim + hidden_dim, hidden_dim, pres, vres, 3)
        # self.wh = SConv3d(input_dim, hidden_dim, pres, vres, 3)
        # self.wq = SConv3d(input_dim, hidden_dim, pres, vres, 3)
        self.attention = nn.Sequential(
            spnn.Conv3d(input_dim + hidden_dim, hidden_dim, kernel_size=3, stride=1),  #48,16
            spnn.BatchNorm(hidden_dim), spnn.ReLU(True)
        )
        self.point_transforms = nn.Sequential(
            nn.Linear(hidden_dim + input_dim, input_dim),)
        
        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, hidden, query):
        print('linear layer!')
        # query = self.wq(query)
        # hidden = self.wh(hidden)
        hx = PointTensor(torch.cat([hidden.F,query.F], dim=1), query.C) #4096,192
        x = initial_voxelize(hx, self.pres, self.vres)
        x = self.attention(x)
        out = voxel_to_point(x, hx, nearest=False)
        out.F = out.F + self.point_transforms(hx.F)
        # print(out.F.shape)
        return out.F


if __name__ == '__main__':
    n_vox = 262144
    n_feat = [25,41,81]
    net = []
    for i in range(3):
        net.append(SPVCNN(num_classes=1, in_channels=n_feat[i],
                               pres=1,
                               cr=1,
                               vres=0.08 * 2 ** 2,
                               dropout=False))
    for i in range(3):
        final_input = torch.rand(n_vox,n_feat[i])
        world_c = torch.rand(n_vox,4)
        world_c[:,3]=1
        point_feat = PointTensor(final_input, world_c)  # 262144,channel,  262144,4
        output = net[i](point_feat)
        print(output.shape)
