import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from backbone import MnasMulti
from torch.nn.functional import grid_sample
from dataset import ScanNet
from swin_transformer import SwinTransformer

class TripleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TripleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, 1, padding='same'),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, 1, padding='same'),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, 1, padding='same'),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class DilConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DilConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, stride=1, padding=2, dilation=2),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, stride=1, padding=2, dilation=2),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, stride=1, padding=2, dilation=2),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class SurfaceNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[32, 80, 160, 300]):
        super(SurfaceNet, self).__init__()
        self.l1 = TripleConv(in_channels, features[0])
        self.l2 = TripleConv(features[0],features[1])
        self.l3 = TripleConv(features[1],features[2])
        self.l4 = DilConv(features[2],features[3])
        self.l5 = TripleConv(16*4,100)
        self.y = nn.Conv3d(100,out_channels,1,padding=0,stride=1)
        self.pool = nn.MaxPool3d(2, stride=2)  # s -> s/2
        self.upconv1 = nn.ConvTranspose3d(features[0], 16, 1, stride=2, padding=0, output_padding=1) #double size
        self.upconv2 = nn.ConvTranspose3d(features[1], 16, 1, stride=4, padding=0, output_padding=3) #four times size
        self.upconv3 = nn.ConvTranspose3d(features[2], 16, 1, stride=4, padding=0, output_padding=3) #four times size
        self.upconv4 = nn.ConvTranspose3d(features[3], 16, 1, stride=4, padding=0, output_padding=3) #four times size
        self.sigmoid = nn.Sigmoid()
        self.feature_extraction = MnasMulti()
        self.swin_transformer = SwinTransformer()

    def forward(self,input_info):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        images = input_info['imgs'].to(device)
        #print(images.shape)
        vol_origin = input_info['vol_origin'].to(device)
        #print(vol_origin.shape)
        voxel_size = torch.tensor(0.04).float()
        #print(voxel_size.shape)
        intr = input_info['intrinsics'].to(device)
        #print(intr.shape)
        cam_poses = input_info['extrinsics'].to(device)
        #print(cam_poses.shape)
        features = self.feature_extraction(images)[1].to(device)
        B,dim_window,C,H,W = features.shape #N,10,40,60,80
        input_transformer = features.transpose(1,2).reshape(B,C,dim_window*H,W)#N,40,600,80
        output_transformer = self.swin_transformer(input_transformer)#N,15,10,768
        output_transformer = output_transformer.unsqueeze(1)#N,1,15,10,768
        output_transformer = output_transformer.permute(0,1,4,2,3).contiguous()#N,1,768,15,10
        # print('the shape of vol_bonds:{}'.format(vol_bonds.shape))
        # print('the shape of voxel_size:{}'.format(voxel_size.shape))
        # print('the shape of intr:{}'.format(intr.shape))
        # print('the shape of cam-poses:{}'.format(cam_poses.shape))
        final_input = backproject_features(output_transformer,vol_origin,voxel_size,intr,cam_poses).to(device)
        x1 = self.l1(final_input)  #s ->s
        x1 = self.pool(x1) # s->s/2
        s1 = self.upconv1(x1) # s/2 -> s
        #print("shape of x:{},shape of s:{}".format(x1.shape,s1.shape))

        x2 = self.l2(x1)   #s/2 -> s/2
        x2 = self.pool(x2)  #s/2 -> s/4
        s2 = self.upconv2(x2) #s/4 -> s
        #print("shape of x:{},shape of s:{}".format(x2.shape, s2.shape))
        x3 = self.l3(x2)   #s/4 -> s/4
        s3 = self.upconv3(x3) #s/4 -> s
        #print("shape of x:{},shape of s:{}".format(x3.shape, s3.shape))
        x4 = self.l4(x3)   #s/4 -> s/4
        s4 = self.upconv4(x4)  #s/4 ->s
        #print("shape of x:{},shape of s:{}".format(x4.shape, s4.shape))
        s5 = torch.cat([s1,s2,s3,s4],dim=1) #s->s
        #print("shape of s5:{}".format(s5.shape))
        s5 = self.l5(s5)
        output = self.y(s5)
        # output = self.sigmoid(output)
        output = output.squeeze(1)
        return output

def backproject_features(features,vol_origin,voxel_sizes,intrs,cam_poses):
    # features :(N,10,40,60,80)
    # vol_bonds: (N,3,2)
    # voxel_size:(N,)
    # intr:(N,10,3,3)
    # cam_poses:(N,10,4,4)
    # output = (N,30,64,64,64)
    # images = images.float()
    # vol_bonds = vol_bonds.float()
    # voxel_sizes = voxel_sizes.float()
    # intrs = intrs.float()
    # cam_poses = cam_poses.float()
    # output = (N,410,64,64,64)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    bs, nm,c,im_h,im_w = features.shape
    output_c = (c+1)*nm
    output = torch.zeros(bs,output_c, 64,64,64)
    for b in range(bs):
        for n in range(nm):
            xv,yv,zv = torch.meshgrid(
                torch.arange(0,64),
                torch.arange(0,64),
                torch.arange(0,64),
            )
            vox_coords = torch.stack([xv.flatten(), yv.flatten(), zv.flatten()], dim=1).long().to(device)
            # Convert world coordinates to camera coordinates
            world_c = vol_origin[b] + (voxel_sizes * vox_coords)
            world_c = torch.cat([world_c,torch.ones(len(world_c),1,device = device)], dim=1)
            cvc = torch.zeros((c+1),64,64,64).to(device)
            cam_pose = cam_poses[b,n,:].squeeze(0)
            world2cam = torch.inverse(cam_pose)
            cam_c = torch.matmul(world2cam.float(),world_c.float().transpose(1,0)).transpose(1,0).float()
            intr = intrs[b,n]
            #print('the shape of intr is:'.format(intr.shape))
            # convert camera coordinates to pixel coordinates
            fx, fy = intr[0, 0], intr[1, 1]
            cx, cy = intr[0, 2], intr[1, 2]
            pix_z = cam_c[:, 2]
            pix_x = torch.round((cam_c[:, 0] * fx / cam_c[:, 2]) + cx).long().to(device)
            pix_y = torch.round((cam_c[:, 1] * fy / cam_c[:, 2]) + cy).long().to(device)
            pixel_grid = torch.stack([2 * pix_x / (im_w - 1) - 1, 2 * pix_y / (im_h - 1) - 1], dim=-1)

            mask = pixel_grid.abs() <= 1
            mask = (mask.sum(dim=-1) == 2) & (pix_z > 0)

            pixel_grid = pixel_grid.view(1, 1, -1, 2)
            features = grid_sample(features[b,n,:].unsqueeze(0), pixel_grid, padding_mode='zeros', align_corners=True)#1,40,60,80->1,40,1,64*64*64

            features = features.view(1, c, -1)  #10ï¼Œ80ï¼Œ13824
            mask = mask.view(1, -1)
            pix_z = pix_z.view(1, -1)

            # remove nan
            features[mask.unsqueeze(1).expand(-1, c, -1) == False] = 0
            pix_z[mask == False] = 0

            features = features.sum(dim=0)
            mask = mask.sum(dim=0)
            invalid_mask = mask == 0
            mask[invalid_mask] = 1
            in_scope_mask = mask.unsqueeze(0)
            features /= in_scope_mask
            features = features.permute(1, 0).contiguous()

            pix_z = pix_z.sum(dim=0).unsqueeze(1) / in_scope_mask.permute(1, 0).contiguous()
            im_z_mean = pix_z[pix_z > 0].mean()
            im_z_std = torch.norm(pix_z[pix_z > 0] - im_z_mean) + 1e-5
            im_z_norm = (pix_z - im_z_mean) / im_z_std
            im_z_norm[pix_z <= 0] = 0
            features = torch.cat([features, im_z_norm], dim=1) #64*64*64,41

            # Eliminate pixels outside view frustum
            # valid_pix = (pix_x >= 0) & (pix_x < im_w) & (pix_y >= 0) & (pix_y < im_h) & (pix_z > 0)
            rgb_val = torch.zeros(len(pix_x),c+1).to(device)
            rgb_val_valid = features[mask]
            # assign rgb value
            valid_vox_x = vox_coords[mask, 0]
            valid_vox_y = vox_coords[mask, 1]
            valid_vox_z = vox_coords[mask, 2]
            cvc[:,valid_vox_x,valid_vox_y,valid_vox_z] = rgb_val_valid.reshape(c+1,-1)
            output[b,(c+1)*n:(c+1)*n+c+1,:] = cvc
            return output

if __name__ == '__main__':
    input_info = {'color_image': torch.rand(2,10,3,480,640),
                  'intr': torch.rand(2,3,3),
                  'cam_pose': torch.rand(2,10,4,4),
                  'vol_bonds': torch.rand(2,3,2),
                  'voxel_size': torch.rand(2,)}
    # # color_image:(N,10,3,480,640)
    # # vol_bonds: (N,3,2)
    # # voxel_size:(N,)
    # # intr:(N,3,3)
    # # cam_poses:(N,10,4,4)
    testnet = SurfaceNet(410,1)
    x = testnet(input_info)
    print(x.shape)


    # features :(N,10,40,60,80)
    # vol_bonds: (N,3,2)
    # voxel_size:(N,)
    # intr:(N,3,3)
    # cam_poses:(N,10,4,4)
    # # output = (N,30,64,64,64)
    # features = torch.rand(20,10,40,60,80)
    # vol_bonds = torch.rand(20,3,2)
    # voxel_size = torch.rand(20,)
    # intrs = torch.rand(20,3,3)
    # cam_poses = torch.rand(20,10,4,4)
    # output = backproject_features(features,vol_bonds,voxel_size,intrs,cam_poses)
    # print(output.shape)
