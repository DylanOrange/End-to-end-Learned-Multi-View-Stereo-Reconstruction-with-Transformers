from turtle import shape
import torch
import torch.nn as nn
import torchsparse
import torchsparse.nn as spnn
from torchsparse.tensor import PointTensor
from torchsparse.utils import *
from tools.torchsparse_utils import *
import math
import torch.nn.functional as F

def get_dists(points1, points2):
    '''
    Calculate dists between two group points
    :param cur_point: shape=(B, M, C)
    :param points: shape=(B, N, C)
    :return:
    '''
    B, M, C = points1.shape
    _, N, _ = points2.shape
    dists = torch.sum(torch.pow(points1, 2), dim=-1).view(B, M, 1) + \
            torch.sum(torch.pow(points2, 2), dim=-1).view(B, 1, N)
    dists -= 2 * torch.matmul(points1, points2.permute(0, 2, 1))
    dists = torch.where(dists < 0, torch.ones_like(dists) * 1e-7, dists) # Very Important for dist = 0.
    return torch.sqrt(dists).float()

def fps(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def three_nn(xyz1, xyz2):
    '''

    :param xyz1: shape=(B, N1, 3)
    :param xyz2: shape=(B, N2, 3)
    :return: dists: shape=(B, N1, 3), inds: shape=(B, N1, 3)
    '''
    dists = get_dists(xyz1, xyz2)
    dists, inds = torch.sort(dists, dim=-1)
    dists, inds = dists[:, :, :3], inds[:, :, :3]
    return dists, inds


def three_interpolate(xyz1, xyz2, points2):
    '''

    :param xyz1: shape=(B, N1, 3)
    :param xyz2: shape=(B, N2, 3)
    :param points2: shape=(B, N2, C2)
    :return: interpolated_points: shape=(B, N1, C2)
    '''
    _, _, C2 = points2.shape
    dists, inds = three_nn(xyz1, xyz2)
    inversed_dists = 1.0 / (dists + 1e-8)
    weight = inversed_dists / torch.sum(inversed_dists, dim=-1, keepdim=True) # shape=(B, N1, 3)
    weight = torch.unsqueeze(weight, -1).repeat(1, 1, 1, C2)
    interpolated_points = gather_points(points2, inds)  # shape=(B, N1, 3, C2)
    interpolated_points = torch.sum(weight * interpolated_points, dim=2)
    return interpolated_points

def gather_points(points, inds):
    '''
    :param points: shape=(B, N, C)
    :param inds: shape=(B, M) or shape=(B, M, K)
    :return: sampling points: shape=(B, M, C) or shape=(B, M, K, C)
    '''
    device = points.device
    B, N, C = points.shape
    inds_shape = list(inds.shape)
    inds_shape[1:] = [1] * len(inds_shape[1:])
    repeat_shape = list(inds.shape)
    repeat_shape[0] = 1
    batchlists = torch.arange(0, B, dtype=torch.long).to(device).reshape(inds_shape).repeat(repeat_shape)
    return points[batchlists, inds, :]

class AttentionLayer(nn.Module):
    def __init__(self, inc=64, outc=64):
        super(AttentionLayer, self).__init__()
        self.hidden_dim = inc
        self.wv = nn.Linear(in_features=inc, out_features=outc)
        self.wq = nn.Linear(in_features=inc, out_features=outc)
        self.wk = nn.Linear(in_features=inc, out_features=outc)

    def forward(self, hidden, query):

        q = self.wq(query)
        k = self.wk(hidden)
        v = self.wv(hidden)

        k_t = k.T
        score = q@k_t
        score = score/math.sqrt(self.hidden_dim)
        score = F.softmax(score, dim = -1)

        weight = score @ v
        return weight

class AttentionBlock(nn.Module):
    def __init__(self, in_channels, sample = True, number = None):
        super(AttentionBlock, self).__init__()
        self.attention = AttentionLayer(inc=in_channels, outc=in_channels)
        self.weight_initialization()
        self.sample = sample
        self.number = number

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
        if self.sample == True:
            query_fB  = query.F.unsqueeze(0) #1,4096,96
            hidden_fB = hidden.F.unsqueeze(0) #1,4096,96

            B,N,C = query_fB.shape
            print(query_fB.shape)

            sampled_query_feat = F.interpolate(query_fB.permute(0,2,1).contiguous(),size=self.number, mode='nearest') #1,C,N
            sampled_hidden_feat = F.interpolate(hidden_fB.permute(0,2,1).contiguous(), size=self.number, mode='nearest')#1,C,N
            print(sampled_query_feat.shape)

            query_attention = self.attention(sampled_hidden_feat.squeeze(0).permute(1,0).contiguous(), sampled_query_feat.squeeze(0).permute(1,0).contiguous())# N,C
            query_attention = query_attention.unsqueeze(0).permute(0,2,1).contiguous()
            print(query_attention.shape)

            query =  F.interpolate(query_attention, size=N, mode='linear') #1,C,N
            query = query.squeeze(0).permute(1,0).contiguous()
            print(query.shape)

        else:
             query = self.attention(hidden.F, query.F)

        return query