from numpy import number
import torch
import torch.nn as nn
from torchsparse.tensor import PointTensor
from tools.utils import sparse_to_dense_channel, sparse_to_dense_torch
from .modules import ConvGRU, AttentionBlock1, AttentionBlock0, AttentionBlock2, LinearLayer, AttenSeq2Seq
from .attention_pointnet import AttentionBlock


class Fusion(nn.Module):
    """
    Two functionalities of this class:
    1. GRU Fusion module as in the paper. Update hidden state features with ConvGRU.
    2. Substitute TSDF in the global volume when direct_substitute = True.
    """

    def __init__(self, voxel_dim,voxel_size, direct_substitute=False):
        super(Fusion, self).__init__()
        # self.cfg = cfg
        # replace tsdf in global tsdf volume by direct substitute corresponding voxels
        self.direct_substitude = direct_substitute
        self.voxel_dim = voxel_dim
        self.voxel_size = voxel_size

        if direct_substitute: #test
            # tsdf
            self.ch_in = [1, 1, 1]
            self.feat_init = 0
        else:
            self.ch_in = [96,48,24] #train
            self.feat_init = 0

        self.n_scales = 2
        self.scene_name = [None, None, None]
        self.global_origin = [None, None, None]
        self.global_volume = [None, None, None]
        self.target_occ_volume = [None, None, None]

        if direct_substitute:
            self.fusion_nets = None
        else:
            # self.fusion_nets = nn.ModuleList()
            # for i, ch in enumerate(self.ch_in):
            #     self.fusion_nets.append(ConvGRU(hidden_dim=ch,
            #                                     input_dim=ch,
            #                                     pres=1,
            #                                     vres=0.08 * 2 ** (self.n_scales - i)))

            self.fusion_nets = nn.ModuleList((AttentionBlock0(in_channels=96, rate = 1/2**0,pres=1,vres=0.08 * 2 ** (self.n_scales - 0)),
            AttentionBlock1(in_channels=48, rate = 1/2**1,pres=1,vres=0.08 * 2 ** (self.n_scales - 1)),
            AttentionBlock2(in_channels=24, rate = 1/2**2,pres=1,vres=0.08 * 2 ** (self.n_scales - 2))))
            # self.fusion_nets = nn.ModuleList((LinearLayer(hidden_dim = 96, input_dim=96, pres=1,vres=0.08 * 2 ** (self.n_scales - 0)),
            # LinearLayer(hidden_dim=48, input_dim = 48, pres=1,vres=0.08 * 2 ** (self.n_scales - 1)),
            # LinearLayer(hidden_dim=24, input_dim = 24, pres=1,vres=0.08 * 2 ** (self.n_scales - 2))))
            # self.fusion_nets = nn.ModuleList((AttenSeq2Seq(hidden_dim=96,  input_dim=96,pres=1,vres=0.08 * 2 ** (self.n_scales - 0)),
            # AttenSeq2Seq(hidden_dim=48,  input_dim=48,pres=1,vres=0.08 * 2 ** (self.n_scales - 1)),
            # AttenSeq2Seq(hidden_dim=24,  input_dim=24,pres=1,vres=0.08 * 2 ** (self.n_scales - 2))))
            # self.fusion_nets = nn.ModuleList((AttentionBlock(in_channels=96, sample=False),
            # AttentionBlock(in_channels=48, sample=True, number=5000),
            # AttentionBlock(in_channels=24, sample=True, number=10000)))



    def reset(self, i):
        self.global_volume[i] = PointTensor(torch.Tensor([]), torch.Tensor([]).view(0, 3).long()).cuda()
        self.target_occ_volume[i] = PointTensor(torch.Tensor([]), torch.Tensor([]).view(0, 3).long()).cuda()

    def convert2dense(self, current_coords, current_values, coords_target_global, occ_target, relative_origin,
                      scale):
        '''
        1. convert sparse feature to dense feature;
        2. combine current feature coordinates and previous coordinates within FBV from global hidden state to get
        new feature coordinates (updated_coords);
        3. fuse ground truth tsdf.

        :param current_coords: (Tensor), current coordinates, (N, 3)
        :param current_values: (Tensor), current features/tsdf, (N, C)
        :param coords_target_global: (Tensor), ground truth coordinates, (N', 3)
        :param tsdf_target: (Tensor), tsdf ground truth, (N',)
        :param relative_origin: (Tensor), origin in global volume, (3,)
        :param scale:
        :return: updated_coords: (Tensor), coordinates after combination, (N', 3)
        :return: current_volume: (Tensor), current dense feature/tsdf volume, (DIM_X, DIM_Y, DIM_Z, C)
        :return: global_volume: (Tensor), global dense feature/tsdf volume, (DIM_X, DIM_Y, DIM_Z, C)
        :return: target_volume: (Tensor), dense target tsdf volume, (DIM_X, DIM_Y, DIM_Z, 1)
        :return: valid: mask: 1 represent in current FBV (N,)
        :return: valid_target: gt mask: 1 represent in current FBV (N,)
        '''
        # previous frame
        global_coords = self.global_volume[scale].C
        global_value = self.global_volume[scale].F
        global_occ_target = self.target_occ_volume[scale].F
        global_coords_target = self.target_occ_volume[scale].C

        dim = (torch.Tensor([64,64,64]).cuda() // 2 ** (3 - scale - 1)).int()
        dim_list = dim.data.cpu().numpy().tolist()
        # dim = (torch.Tensor(self.voxel_dim).cuda()).int()
        # dim_list = dim.data.cpu()

        # mask voxels that are out of the FBV
        global_coords = global_coords - relative_origin
        valid = ((global_coords < dim) & (global_coords >= 0)).all(dim=-1)
        # if self.direct_substitude:
        # valid_volume = sparse_to_dense_torch(current_coords, 1, dim_list, 0, global_value.device)
        # value = valid_volume[global_coords[valid][:, 0], global_coords[valid][:, 1], global_coords[valid][:, 2]]
        # all_true = valid[valid]
        # all_true[value == 0] = False
        # valid[valid] = all_true
        global_volume = sparse_to_dense_channel(global_coords[valid], global_value[valid], dim_list, self.ch_in[scale],
                                                self.feat_init, global_value.device)

        current_volume = sparse_to_dense_channel(current_coords, current_values, dim_list, self.ch_in[scale],
                                                 self.feat_init, global_value.device)

        #     # change the structure of sparsity, combine current coordinates and previous coordinates from global volume
        if self.direct_substitude:
            updated_coords = torch.nonzero((global_volume > 0).any(-1) | (current_volume > 0).any(-1))
            # print('updated {}'.format(updated_coords.shape))
            # print('current {}'.format(current_coords.shape))
            # updated_coords = current_coords
        else:
            updated_coords = torch.nonzero((global_volume != 0).any(-1) | (current_volume != 0).any(-1))

        # fuse ground truth
        if occ_target is not None:
            # mask voxels that are out of the FBV
            global_coords_target = global_coords_target - relative_origin
            valid_target = ((global_coords_target < dim) & (global_coords_target >= 0)).all(dim=-1)
            # combine current tsdf and global tsdf
            coords_target = torch.cat([global_coords_target[valid_target], coords_target_global])[:, :3]
            occ_target = torch.cat([global_occ_target[valid_target], occ_target.unsqueeze(-1)])
            # sparse to dense
            target_volume = sparse_to_dense_channel(coords_target, occ_target, dim_list, 1, 0,
                                                    occ_target.device)
        else:
            target_volume = valid_target = None

        return updated_coords, current_volume, global_volume, target_volume, valid, valid_target

    def update_map(self, value, coords, target_volume, valid, valid_target,
                   relative_origin, scale):
        '''
        Replace Hidden state/tsdf in global Hidden state/tsdf volume by direct substitute corresponding voxels
        :param value: (Tensor) fused feature (N, C)
        :param coords: (Tensor) updated coords (N, 3)
        :param target_volume: (Tensor) tsdf volume (DIM_X, DIM_Y, DIM_Z, 1)
        :param valid: (Tensor) mask: 1 represent in current FBV (N,)
        :param valid_target: (Tensor) gt mask: 1 represent in current FBV (N,)
        :param relative_origin: (Tensor), origin in global volume, (3,)
        :param scale:
        :return:
        '''
        # pred
        self.global_volume[scale].F = torch.cat(
            [self.global_volume[scale].F[valid == False], value])
        coords = coords + relative_origin
        self.global_volume[scale].C = torch.cat([self.global_volume[scale].C[valid == False], coords])

        # target
        if target_volume is not None:
            target_volume = target_volume.squeeze()
            self.target_occ_volume[scale].F = torch.cat(
                [self.target_occ_volume[scale].F[valid_target == False],
                 target_volume[target_volume>0].unsqueeze(-1)])
            target_coords = torch.nonzero(target_volume>0) + relative_origin

            self.target_occ_volume[scale].C = torch.cat(
                [self.target_occ_volume[scale].C[valid_target == False], target_coords])

    def save_mesh(self, scale, outputs, scene):
        if outputs is None:
            outputs = dict()
        if "scene_name" not in outputs:
            outputs['origin'] = []
            outputs['scene_occ'] = []
            outputs['scene_name'] = []
        # only keep the newest result
        if scene in outputs['scene_name']:
            # delete old
            idx = outputs['scene_name'].index(scene)
            del outputs['origin'][idx]
            del outputs['scene_occ'][idx]
            del outputs['scene_name'][idx]

        # scene name
        outputs['scene_name'].append(scene)

        fuse_coords = self.global_volume[scale].C
        occ = self.global_volume[scale].F.squeeze(-1)
        max_c = torch.max(fuse_coords, dim=0)[0][:3]
        min_c = torch.min(fuse_coords, dim=0)[0][:3]
        outputs['origin'].append(min_c * self.voxel_size * (2 ** (3 - scale - 1)))

        ind_coords = fuse_coords - min_c
        dim_list = (max_c - min_c + 1).int().data.cpu().numpy().tolist()
        occ_volume = sparse_to_dense_torch(ind_coords, occ, dim_list, 0, occ.device)
        outputs['scene_occ'].append(occ_volume)

        return outputs

    def forward(self, coords, values_in, inputs, reset, scale=2, outputs=None, save_mesh=False):
        '''
        :param coords: (Tensor), coordinates of voxels, (N, 4) (4 : Batch ind, x, y, z)
        :param values_in: (Tensor), features/tsdf, (N, C)
        :param inputs: dict: meta data from dataloader
        :param scale:
        :param outputs:
        :param save_mesh: a bool to indicate whether or not to save the reconstructed mesh of current sample
        if direct_substitude:
        :return: outputs: dict: {
            'origin':                  (List), origin of the predicted partial volume,
                                    [3]
            'scene_tsdf':              (List), predicted tsdf volume,
                                    [(nx, ny, nz)]
            'target':                  (List), ground truth tsdf volume,
                                    [(nx', ny', nz')]
            'scene_name':                  (List), name of each scene in 'scene_tsdf',
                                    [string]
        }
        else:
        :return: updated_coords_all: (Tensor), updated coordinates, (N', 4) (4 : Batch ind, x, y, z)
        :return: values_all: (Tensor), features after gru fusion, (N', C)
        :return: tsdf_target_all: (Tensor), tsdf ground truth, (N', 1)
        :return: occ_target_all: (Tensor), occupancy ground truth, (N', 1)
        '''
        if self.global_volume[scale] is not None:
            # delete computational graph to save memory
            self.global_volume[scale] = self.global_volume[scale].detach()
        # if reset:
        #     self.reset(scale)

        batch_size = len(inputs['fragment'])
        interval = 2 ** (3 - scale - 1)

        occ_target_all = None
        values_all = None
        updated_coords_all = None

        # ---incremental fusion----
        for i in range(batch_size):
            scene = inputs['scene'][i]  # scene name
            global_origin = inputs['vol_origin'][i]  # origin of global volume
            origin = inputs['vol_origin_partial'][i]  # origin of part volume

            if scene != self.scene_name[scale] and self.scene_name[scale] is not None and self.direct_substitude:
                outputs = self.save_mesh(scale, outputs, self.scene_name[scale])#save last scene
                self.reset(scale)

            # if this fragment is from new scene, we reinitialize backend map
            if self.scene_name[scale] is None or scene != self.scene_name[scale]:
                self.scene_name[scale] = scene
                self.reset(scale)
                self.global_origin[scale] = global_origin

            # each level has its corresponding voxel size
            voxel_size = self.voxel_size * interval

            # relative origin in global volume
            relative_origin = (origin - self.global_origin[scale]) / voxel_size
            relative_origin = relative_origin.cuda().long()

            batch_ind = torch.nonzero(coords[:, 0] == i).squeeze(1)
            if len(batch_ind) == 0:
                continue
            coords_b = coords[batch_ind, 1:].long() // interval
            values = values_in[batch_ind]

            if 'occ_list' in inputs.keys():
                # get partial gt
                occ_target = inputs['occ_list'][3 - scale - 1][i]
                coords_target = torch.nonzero(occ_target)
                occupancy = (occ_target>0)
                occ_target = occ_target[occupancy]
                # tsdf_target = inputs['tsdf_list'][self.cfg.N_LAYER - scale - 1][i][occ_target]
                # coords_target = torch.nonzero(occ_target)
            else:
                occ_target  = None

            # convert to dense: 1. convert sparse feature to dense feature; 2. combine current feature coordinates and
            # previous feature coordinates within FBV from our backend map to get new feature coordinates (updated_coords)
            updated_coords, current_volume, global_volume, target_volume, valid, valid_target = self.convert2dense(
                coords_b,
                values,
                coords_target,
                occ_target,
                relative_origin,
                scale)

            # dense to sparse: get features using new feature coordinates (updated_coords)
            values = current_volume[updated_coords[:, 0], updated_coords[:, 1], updated_coords[:, 2]]
            global_values = global_volume[updated_coords[:, 0], updated_coords[:, 1], updated_coords[:, 2]]
            # get fused gt
            if target_volume is not None:
                occ_target = target_volume[updated_coords[:, 0], updated_coords[:, 1], updated_coords[:, 2]]
                # occ_target = tsdf_target.abs() < 1
            else:
                occ_target = None

            if not self.direct_substitude:
                # convert to aligned camera coordinate
                r_coords = updated_coords.detach().clone().float()
                r_coords = r_coords.permute(1, 0).contiguous().float() * voxel_size + origin.unsqueeze(-1).float()
                r_coords = torch.cat((r_coords, torch.ones_like(r_coords[:1])), dim=0)
                r_coords = inputs['world_to_aligned_camera'][i, :3, :] @ r_coords
                r_coords = torch.cat([r_coords, torch.zeros(1, r_coords.shape[-1]).to(r_coords.device)])
                r_coords = r_coords.permute(1, 0).contiguous()

                h = PointTensor(global_values, r_coords)
                x = PointTensor(values, r_coords)

                values = self.fusion_nets[scale](h, x)

            # feed back to global volume (direct substitute)
            self.update_map(values, updated_coords, target_volume, valid, valid_target, relative_origin, scale)

            if updated_coords_all is None:
                updated_coords_all = torch.cat([torch.ones_like(updated_coords[:, :1]) * i, updated_coords * interval],
                                               dim=1)
                values_all = values
                occ_target_all = occ_target
            else:
                updated_coords = torch.cat([torch.ones_like(updated_coords[:, :1]) * i, updated_coords * interval],
                                           dim=1)
                updated_coords_all = torch.cat([updated_coords_all, updated_coords])
                values_all = torch.cat([values_all, values])
                if occ_target_all is not None:
                    occ_target_all = torch.cat([occ_target_all, occ_target])

            if self.direct_substitude and save_mesh:
                    outputs = self.save_mesh(scale, outputs, self.scene_name[scale])
                    self.reset(scale)
                    self.scene_name[scale] = None

        if self.direct_substitude:
            return outputs
        else:
            return updated_coords_all, values_all, occ_target_all
