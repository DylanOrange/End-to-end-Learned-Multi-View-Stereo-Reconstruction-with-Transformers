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


class AttenSeq2Seq(nn.Module):
    def __init__(self, hidden_dim=96, input_dim=96):
        super(AttenSeq2Seq, self).__init__()
        self.attention = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1)
        self.wh =  nn.Linear(input_dim, hidden_dim)
        self.wq =  nn.Linear(input_dim, hidden_dim)
        self.update =  nn.Linear(input_dim + hidden_dim, hidden_dim)

    def forward(self, hidden, query):
        '''

        :param h: PintTensor
        :param x: PintTensor
        :return: h.F: Tensor (N, C)
        
        '''
        print('attenlinear!')
        query = self.wq(query.F)
        hidden = self.wh(hidden.F)
        hx = torch.cat([hidden,query], dim=1)#4096,192
        energy = torch.tanh(self.attention(hx)) #4096,96
        attention = self.v(energy)  #4096,1
        # attention.masked_fill_(attn_mask, -1e9)
        attention = F.softmax(energy,dim=0)
        context = attention * hidden #4096,96
        query = torch.cat([query, context], dim=1) #4096,192
        output = torch.tanh(self.update(query)) #4096,96
        return output

class MultiHeadAttention(nn.Module):
    def __init__(self, inc,n_heads, hidden_dim, pres = 1, vres = 1):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.hidden_dim = hidden_dim

        self.attention = AttenSeq2Seq(hidden_dim = hidden_dim, input_dim = inc)
        self.W_Q = nn.Linear(inc, hidden_dim * n_heads, bias=False)
        self.W_H = nn.Linear(inc, hidden_dim * n_heads, bias=False)
        self.FC = nn.Linear(n_heads * hidden_dim, hidden_dim, bias=False)
        self.LN = nn.LayerNorm(self.hidden_dim)

    def forward(self, input_Q, input_H):
        '''
        input_Q: [, len_q, inc]
        input_K: [, len_k, inc]
        input_V: [batch_size, len_v(=len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        '''
        query = self.W_Q(input_Q.F).view(-1, self.n_heads, self.hidden_dim).transpose(0,1)  # Q: [, n_heads, len_q, hidden_dim]
        hidden = self.W_H(input_H.F).view(-1, self.n_heads, self.hidden_dim).transpose(0,1)  # V: [, n_heads, len_v(=len_k), hidden_dim]

        context = self.attention(hidden,query) #[, n_heads, len_v(=len_k), hidden_dim]
        context = context.transpose(1, 2).reshape(-1, self.n_heads * self.hidden_dim) # context: [batch_size, len_q, n_heads * d_v]
        output = self.FC(context) # [batch_size, len_q, d_model]
        output = self.LN(output + input_Q.F)
        return output

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, hidden_dim):
        super(PoswiseFeedForwardNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=False)
        )
    def forward(self, inputs):
        '''
        inputs: [batch_size, seq_len, d_model]
        '''
        residual = inputs
        output = self.fc(inputs)
        # return nn.LayerNorm(self.hidden_dim).cuda()(output + residual) # [batch_size, seq_len, d_model]
        return nn.BatchNorm1d(self.hidden_dim).cuda()(output + residual)


class EncoderLayer(nn.Module):
    def __init__(self, inc, n_heads, hidden_dim):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(inc,n_heads, hidden_dim)
        self.pos_ffn = PoswiseFeedForwardNet(hidden_dim)
        self.weight_initialization()
    
    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, hidden, query):
        '''
        enc_inputs: [batch_size, src_len, d_model]
        enc_self_attn_mask: [batch_size, src_len, src_len]
        '''
        # enc_outputs: [batch_size, src_len, d_model], attn: [batch_size, n_heads, src_len, src_len]
        enc_outputs = self.enc_self_attn(query, hidden) # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs) # enc_outputs: [batch_size, src_len, d_model]
        return enc_outputs


class LinearLayer(nn.Module):
    def __init__(self, hidden_dim=96, input_dim=96, pres=1, vres=1):
        super(LinearLayer, self).__init__()
        self.pres = pres
        self.vres = vres
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
