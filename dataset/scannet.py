import os
import numpy as np
import pickle
import cv2
from PIL import Image
from torch.utils.data import Dataset
import h5py
import torch

class ScanNetDataset(Dataset):
    def __init__(self, datapath=None, mode=None, transforms=None, length= None):
        super(ScanNetDataset, self).__init__()
        self.datapath = datapath  # ./scannet
        self.mode = mode
        self.transforms = transforms
        self.length = length
        self.fragments = 'fragments'  # all_tsdf_9  wnidow_size = 9
        self.epoch = None
        assert self.mode in ["train", "test",'val']
        self.metas = self.build_list()
        if mode == 'test':
            self.source_path = 'scans_test'
        else:
            self.source_path = 'scans'

        self.tsdf_cashe = {}
        self.max_cashe = 100

    def build_list(self):
        val_path = './tools/split'
        if self.mode == 'val':
            with open(os.path.join(val_path, 'fragments_{}.pkl'.format(self.mode)),
                    'rb') as f:
                metas = pickle.load(f)
        else:
            with open(os.path.join(self.datapath, self.fragments, 'fragments_{}_15_with_partial.pkl'.format(self.mode)),
                    'rb') as f:
                metas = pickle.load(f)
        return metas  

    def __len__(self):
            return int(len(self.metas)) if self.length == None else self.length

    def read_cam_file(self, filepath, index):
        intrinsics = np.loadtxt(os.path.join(filepath, 'intrinsic', 'intrinsic_color.txt'), delimiter=' ')[:3, :3]
        intrinsics = intrinsics.astype(np.float32)
        extrinsics = np.loadtxt(os.path.join(filepath, 'pose', '{}.txt'.format(str(index))))
        return intrinsics, extrinsics

    def read_img(self, filepath):
        img = Image.open(filepath)
        return img

    def read_depth(self, filepath):
        # Read depth image and camera pose
        depth_im = cv2.imread(filepath, -1).astype(
            np.float32)
        depth_im /= 1000.  # depth is saved in 16-bit PNG in millimeters
        depth_im[depth_im > 3.0] = 0
        return depth_im

    def get_target(self, occ):

        # 2 ** scale == interval
        xv,yv,zv = torch.meshgrid(
                torch.arange(0,64),
                torch.arange(0,64),
                torch.arange(0,64),
            )
        vox_coords = torch.stack([xv.flatten(), yv.flatten(), zv.flatten()], dim=1)
        occ_target = occ[vox_coords[:, 0], vox_coords[:, 1], vox_coords[:, 2]]
        #print(occ_target.shape)
        # occ_target = occ_target.reshape(64*64*64,1)

        return occ_target

    def read_scene_volumes(self, data_path, scene):
        if scene not in self.tsdf_cashe.keys():
            if len(self.tsdf_cashe) > self.max_cashe:
                self.tsdf_cashe = {}
            full_tsdf_list = []
            for l in range(3):
                # load full tsdf volume
                full_tsdf = np.load(os.path.join(data_path, scene, 'full_tsdf_layer{}.npz'.format(l)),
                                    allow_pickle=True)
                full_tsdf_list.append(full_tsdf.f.arr_0)
            self.tsdf_cashe[scene] = full_tsdf_list
        return self.tsdf_cashe[scene]

    def __getitem__(self, idx):
        meta = self.metas[idx]
        imgs = []
        depth = []
        extrinsics_list = []
        intrinsics_list = []
        occ=[]

        # begin = time.time()
        tsdf_list = self.read_scene_volumes('./tsdf_gt/all_tsdf_10', meta['scene'])

        for i, vid in enumerate(meta['image_ids']):
            # load images
            imgs.append(
                self.read_img(
                    os.path.join(self.datapath, self.source_path, meta['scene'], 'color', '{}.jpg'.format(vid))))

            depth.append(
                self.read_depth(
                    os.path.join(self.datapath, self.source_path, meta['scene'], 'depth', '{}.png'.format(vid))))

            # load intrinsics and extrinsics
            intrinsics, extrinsics = self.read_cam_file(os.path.join(self.datapath, self.source_path, meta['scene']),vid)
            assert (extrinsics[0][0] != np.inf and extrinsics[0][0] != -np.inf and extrinsics[0][0] != np.nan)
            intrinsics_list.append(intrinsics)
            extrinsics_list.append(extrinsics)

        intrinsics = np.stack(intrinsics_list)
        extrinsics = np.stack(extrinsics_list)
        fragment_id = meta['scene'] + '_' + str(meta['fragment_id'])
        print(fragment_id)


        items = {
            'imgs': imgs,
            'depth':depth,
            'intrinsics': intrinsics,
            'extrinsics': extrinsics,
            'tsdf_list_full': tsdf_list,
            'occ_list': occ,
            'vol_origin': meta['vol_origin'],  # will be transoform to 0,0,0 in pre-processing
            'vol_origin_partial': meta['vol_origin_partial'],  # partial origin
            'scene': meta['scene'],
            'fragment': meta['scene'] + '_' + str(meta['fragment_id']),
            'epoch': [self.epoch],
        }


        if self.transforms is not None:
            items = self.transforms(items)

        return items
        