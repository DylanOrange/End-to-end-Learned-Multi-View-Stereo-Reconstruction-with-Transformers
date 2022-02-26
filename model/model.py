import imp
import sys
from turtle import write_docstringdict

sys.path.append("..")
import torch
import torch.nn as nn
import torchsparse
import torchsparse.nn as spnn
from torchsparse.tensor import PointTensor
from torchsparse.utils import *
from model.backbone import MnasMulti
from model.modules import SPVCNN
from model.fusion import Fusion
import sys
# from utils.back_project import backproject_features
from tools.generate_grid import generate_grid
# from model.enet import ENet
from tools.torchsparse_utils import *
import torch.nn.functional as F
from tools.back_project import back_project
from tools.utils import tocuda
from tools.occ2ply import writeocc
from tools.occ2ply_coordinates import writeocc_coordinates


class ReconNet(nn.Module):
    def __init__(self, fusion):
        super(ReconNet, self).__init__()
        self.feature_extraction = MnasMulti(alpha=1.0)  # 24,40,80
        self.N_VOX = [64, 64, 64]
        self.fusion = fusion

        self.n_scales = 2
        self.pixel_mean = torch.Tensor([125.78, 112.22, 97.13]).view(1, -1, 1, 1)
        self.pixel_std = torch.Tensor([59.64, 57.92, 55.41]).view(1, -1, 1, 1)
        self.sp_convs = nn.ModuleList()
        self.occ_preds = nn.ModuleList()
        in_channels = [81, 138, 74]
        channels = [96, 48, 24]
        if self.fusion == True:
            self.AttentionFusion = Fusion([64,64,64], 0.08, direct_substitute=False)
        self.Visualize = Fusion([64,64,64], 0.08, direct_substitute=True)
        for i in range(3):
            self.sp_convs.append(SPVCNN(num_classes=1, in_channels=in_channels[i],
                                        pres=1,
                                        cr=1 / 2 ** i,
                                        vres=0.08 * 2 ** (self.n_scales - i),
                                        dropout=False))
            self.occ_preds.append(nn.Linear(channels[i], 1))

    def normalizer(self, x):
        """ Normalizes the RGB images to the input range"""
        return (x - self.pixel_mean.type_as(x)) / self.pixel_std.type_as(x)

    def upsample(self, pre_feat, pre_coords, interval, num=8):
        '''

        :param pre_feat: (Tensor), features from last level, (N, C)
        :param pre_coords: (Tensor), coordinates from last level, (N, 4) (4 : Batch ind, x, y, z)
        :param interval: interval of voxels, interval = scale ** 2
        :param num: 1 -> 8
        :return: up_feat : (Tensor), upsampled features, (N*8, C)
        :return: up_coords: (N*8, 4), upsampled coordinates, (4 : Batch ind, x, y, z)
        '''
        with torch.no_grad():
            pos_list = [1, 2, 3, [1, 2], [1, 3], [2, 3], [1, 2, 3]]
            n, c = pre_feat.shape
            up_feat = pre_feat.unsqueeze(1).expand(-1, num, -1).contiguous()
            up_coords = pre_coords.unsqueeze(1).repeat(1, num, 1).contiguous()
            for i in range(num - 1):
                up_coords[:, i + 1, pos_list[i]] += interval

            up_feat = up_feat.view(-1, c)
            up_coords = up_coords.view(-1, 4)

        return up_feat, up_coords

    def get_target(self, coords, inputs, scale):
        '''
        Won't be used when 'fusion_on' flag is turned on
        :param coords: (Tensor), coordinates of voxels, (N, 4) (4 : Batch ind, x, y, z)
        :param inputs: (List), inputs['tsdf_list' / 'occ_list']: ground truth volume list, [(B, DIM_X, DIM_Y, DIM_Z)]
        :param scale:
        :return: tsdf_target: (Tensor), tsdf ground truth for each predicted voxels, (N,)
        :return: occ_target: (Tensor), occupancy ground truth for each predicted voxels, (N,)
        '''
        with torch.no_grad():
            # tsdf_target = inputs['tsdf_list'][scale]
            occ_target = inputs['occ_list'][scale]
            # print(f"the shape of occ_target is {occ_target.shape}")
            # print(f"the shape of occ_target is {occ_target.shape}")
            coords_down = coords.detach().clone().long()
            # print(f"the shape of coords_down is {coords_down.shape}")
            # 2 ** scale == interval
            # coords_down[:, 1:] = torch.div(coords[:, 1:],2,rounding_mode='floor') ** scale  # down resolution
            coords_down[:, 1:] = (coords[:, 1:] // 2 ** scale)
            occ_target = occ_target[coords_down[:, 0], coords_down[:, 1], coords_down[:, 2], coords_down[:, 3]]
            return occ_target

    # def get_target(self, input_info, scale):
    #     '''
    #     Won't be used when 'fusion_on' flag is turned on
    #     :param coords: (Tensor), coordinates of voxels, (N, 4) (4 : Batch ind, x, y, z)
    #     :param inputs: (List), inputs['tsdf_list' / 'occ_list']: ground truth volume list, [(B, DIM_X, DIM_Y, DIM_Z)]
    #     :param scale:
    #     :return: tsdf_target: (Tensor), tsdf ground truth for each predicted voxels, (N,)
    #     :return: occ_target: (Tensor), occupancy ground truth for each predicted voxels, (N,)
    #     '''
    #     n_vox = int(64 / (2 ** scale))
    #     with torch.no_grad():
    #         occ_target = input_info['occ_list'][scale]
    #         xv, yv, zv = torch.meshgrid(
    #             torch.arange(0, n_vox),
    #             torch.arange(0, n_vox),
    #             torch.arange(0, n_vox),
    #         )
    #         vox_coords = torch.stack([xv.flatten(), yv.flatten(), zv.flatten()], dim=1)
    #     occ_target = occ_target[0, vox_coords[:, 0], vox_coords[:, 1], vox_coords[:, 2]]
    #     return occ_target

    def forward(self, input_info,save_mesh, save_mesh_epoch, reset):
        input_info = tocuda(input_info)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # device = 'cpu'
        images = input_info['imgs']
        bn, view, _, _, _ = images.shape
        images = torch.unbind(images, 1)  # len 10, 1,3,480,640
        # print(images.shape) (N,10,3,480,640)
        # vol_origin = input_info['vol_origin_partial'].to(device)
        # print(vol_origin.shape) (N,3)
        voxel_size = torch.tensor(0.08).float().cuda()
        # print(voxel_size.shape)
        # intr = input_info['intrinsics'].to(device)
        # print(intr.shape)
        # cam_poses = input_info['extrinsics'].to(device)
        # occ_list = input_info['gt_occ'].to(device)
        # print(cam_poses.shape)

        # features = self.feature_extraction(images)[0].to(device)  # (N,24,120,160)
        normalize_images = []
        for img in images:
            normalize_image = self.normalizer(img)
            normalize_images.append(normalize_image)

        features = [self.feature_extraction(normalize_image) for normalize_image in normalize_images]
        # print(f"the shape of features00 is {features[0][0].shape}")      shape of features[0][0]:1,24,120,160
        # features = torch.cat((features), 0)
        # a, c, w, h = features.shape
        # features = features.reshape(bn, view, c, w, h)
        # RIGHT NOW, WE ONLY CONSIDER BATCH_SIZE = 1 !!!!!
        loss = []
        pre_feat = None
        pre_coords = None
        outputs = {}
        for i in range(3):
            # print(f"layer {i} start!")
            interval = 2 ** (self.n_scales - i)  # 4 2 1
            scale = self.n_scales - i  # 2 1 0
            if i == 0:

                # ----generate new coords----

                coords = generate_grid(self.N_VOX, interval)[0].cuda()  # (3,N_vox)
                up_coords = []
                up_coords.append(
                    torch.cat([torch.ones(1, coords.shape[-1]).to(coords.device) * 0, coords]))  # [(4,N),(4,N)...]
                up_coords = torch.cat(up_coords, dim=1).permute(1, 0).contiguous()  # (N*b,4) --> (N,0,x,y,z)
                # print(up_coords[0,0])
            else:

                # ----upsample coords----

                up_feat, up_coords = self.upsample(pre_feat, pre_coords, interval)  # (N*8,c) (N*8,4)

            # ----back project ----

            feats = torch.stack([feat[scale] for feat in features])
            # print(f"the shape of features is {feats.shape}")
            KRcam = input_info['proj_matrices'][:, :, scale].permute(1, 0, 2, 3).contiguous()
            # print(f"the shape of KRcam is {KRcam.shape}")
            volume, count = back_project(up_coords, input_info['vol_origin_partial'], voxel_size, feats,
                                         KRcam)

            grid_mask = count > 1

            # ----concate feature from last stage
            if i != 0:
                # print(f"the shape of volume is {volume.shape}")
                # print(f"the shape of up_feat is {up_feat.shape}")
                feat = torch.cat([volume, up_feat], dim=1)
            else:
                feat = volume

            # transform = input_info['world_to_aligned_camera'][0, :3, :].permute(1, 0).contiguous().cuda()  # 4*3
            r_coords = up_coords.detach().clone().float()
            coords_batch = r_coords[:, 1:] * voxel_size + input_info['vol_origin_partial'][0].cuda().float() #(N,3)
            coords_batch = torch.cat((coords_batch, torch.ones_like(coords_batch[:, :1])), dim=1)  #(N,4)
            coords_batch = coords_batch @ input_info['world_to_aligned_camera'][0,:3, :].permute(1, 0).contiguous()
            r_coords = torch.cat([coords_batch, torch.zeros(coords_batch.shape[0], 1).to(device)], dim=1)

            # ----sparse conv 3d backbone----

            point_feat = PointTensor(feat, r_coords)
            feat = self.sp_convs[i](point_feat)
            if self.fusion == True:
                up_coords, feat, occ_target = self.AttentionFusion(up_coords, feat, input_info, reset, scale = i)
                grid_mask = torch.ones_like(feat[:, 0]).bool()

            # point_feat = PointTensor(final_input, world_c)  # 262144,channel,  262144,4
            # output = self.sp_convs(point_feat)
            occ = self.occ_preds[i](feat)
            if self.fusion == False:
                occ_target = self.get_target(up_coords, input_info, scale).cuda()
            occ_loss = self.compute_loss(occ, occ_target, mask=grid_mask, pos_weight=1.5)
            loss.append(occ_loss)
            occupancy = occ.squeeze(1) > 0

            # print(f"the shape of occ is {occ.shape}")
            occupancy[grid_mask == False] = False
            num = int(occupancy.sum().data.cpu())
            # print(f"the shape of occupancy is {occupancy.shape}")
            pre_coords = up_coords[occupancy]
            batch_ind = torch.nonzero(pre_coords).squeeze(1)
            if len(batch_ind) == 0:
                print('no valid points: scale {}'.format(i))
                return loss, outputs
            pre_feat = feat[occupancy]
            pre_occ = occ[occupancy]

            pre_feat = torch.cat([pre_feat, pre_occ], dim=1)
            # output = output.permute(1,0)
            if i == 2:
                # print(f"the shape of coords is {pre_coords.shape}, the shape of occ is {pre_occ.shape}")
                # occ_target= input_info['occ_list'][0]
                # occ_target = torch.nonzero(occ_target)
                # outputs['coords'] = occ_target
                # outputs['occ'] = torch.ones(occ_target.shape[0],1).cuda()
                outputs['coords']  = pre_coords
                outputs['occ'] = pre_occ

            # visualize training result
            # epoch = input_info['epoch']
            # fragment_id = input_info['fragment']
            # save_path = '/mnt/hdd/praktikum/ldy/result/fusion/test'
            # gt_mask = occ_target.cpu().numpy()
            # writeocc(gt_mask, save_path, 'train_gt_{}_{}.ply'.format(epoch, fragment_id))
            # coords = outputs['coords'].cpu().detach().numpy()
            # # pred_mask = pred_mask > 0.5
            # writeocc_coordinates(coords, save_path,'train_result_{}_{}.ply'.format(epoch, fragment_id))

        if 'coords' in outputs.keys() and save_mesh_epoch:
            print('visualizing')               
            outputs = self.Visualize(outputs['coords'], outputs['occ'], input_info, reset, self.n_scales, outputs, save_mesh)

        return loss, outputs

    @staticmethod
    def compute_loss(occ, occ_target, mask=None, pos_weight=1.0):
        '''

               :param occ: (Tensor), predicted occupancy, (N, 1)
               :param occ_target: (Tensor), ground truth occupancy, (N, 1)
               :param mask: (Tensor), mask voxels which cannot be seen by all views
               :param pos_weight: (float)
               :return: loss: (Tensor)
               '''
        # compute occupancy/tsdf loss
        occ = occ.view(-1)
        occ_target = occ_target.view(-1)
        if mask is not None:
            mask = mask.view(-1)
            occ = occ[mask]
            occ_target = occ_target[mask]

        n_all = occ_target.shape[0]
        n_p = occ_target.sum()
        if n_p == 0:
            # logger.warning('target: no valid voxel when computing loss')
            print('target: no valid voxel when computing loss')
            return torch.Tensor([0.0]).cuda()[0]
        w_for_1 = (n_all - n_p).float() / n_p
        w_for_1 *= pos_weight

        # compute occ bce loss
        occ_loss = F.binary_cross_entropy_with_logits(occ, occ_target.float(), pos_weight=w_for_1)
        return occ_loss


if __name__ == '__main__':
    input_info = {'imgs': torch.rand(1, 10, 3, 480, 640),
                  'intrinsics': torch.rand(1, 10, 3, 3),
                  'extrinsics': torch.rand(1, 10, 4, 4),
                  'vol_origin': torch.rand(1, 3),
                  'voxel_size': torch.rand(1, )}
    # # color_image:(N,10,3,480,640)
    # # vol_bonds: (N,3,2)
    # # voxel_size:(N,)
    # # intr:(N,3,3)
    # # cam_poses:(N,10,4,4)
    testnet = ReconNet().to('cuda')
    x = testnet(input_info)
    print(x.shape)
