import os
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
# from tools.fusion import *
import pickle
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
# from utils import *
import torch.optim as optim
# from model_surfacenet import SurfaceNet
from model.model import ReconNet
# from model_old import SurfaceNet
# from model import SurfaceNet
import matplotlib.pyplot as plt
import tensorboard
import random
from torch.utils.tensorboard import SummaryWriter
from tools import transforms, occ2ply, occ2ply_coordinates
from dataset.scannet import ScanNetDataset
import torch.nn.functional as F
import gc
import sys


def parse_args():
    parser = argparse.ArgumentParser(description='Generation of Ground Truth')
    parser.add_argument("--data_path", metavar="DIR", help='path to raw data', default='/mnt/hdd/praktikum/Deep-Learning-in-Visual-Computing/scannet')
    parser.add_argument("--save_path", metavar="DIR", help="file name", default="result/tsdf/2000/0000")
    parser.add_argument("--max_depth", default=3, type=int)
    parser.add_argument("--margin", default=3, type=int)
    parser.add_argument("--voxel_size", default=0.04, type=float)
    parser.add_argument("--window_size", default=10, type=int)
    parser.add_argument("--min_angle", default=15, type=float)
    parser.add_argument("--min_distance", default=0.1, type=float)
    parser.add_argument("--pos_weight", default=1.5, type=float)
    parser.add_argument("--lr", default=3e-4, type=float)
    parser.add_argument("--clip", default=1.0, type=float)
    parser.add_argument("--save", action="store_true", help="Run or not.")
    parser.add_argument("--mode", default="train")
    return parser.parse_args()


args = parse_args()


def save_checkpoint(model, optimizer, epoch):
    checkpoint = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict()
    }
    FILE = 'checkpoint' + '_' + 'tsdf' +'linear' +' 0000'+str(epoch) + '.pth'
    torch.save(checkpoint, os.path.join('checkpoint', 'tsdf', '2000', FILE))


# Hyperparameters
LEARNING_RATE = args.lr
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_EPOCHS = 60
NUM_WORKERS = 0
LOAD_MODEL = False


def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


if __name__ == '__main__':
    gc.collect()
    for clear in range(10):
        torch.cuda.empty_cache()
    torch.manual_seed(0)

    # --- data augmentation ----
    transform_train = []
    transform_train += [transforms.ResizeImage((640, 480)),
                        transforms.ToTensor(),
                        transforms.RandomTransformSpace(
                            [64, 64, 64], 0.04, True, True,
                            0.1, 0.025, max_epoch=NUM_EPOCHS),
                        transforms.IntrinsicsPoseToProjection(10, 4),
                        ]

    transforms_train = transforms.Compose(transform_train)

    transform_val = []
    transform_val += [transforms.ResizeImage((640, 480)),
                      transforms.ToTensor(),
                      transforms.RandomTransformSpace(
                          [64, 64, 64], 0.04, False, False,
                          0, 0, max_epoch=NUM_EPOCHS),
                      transforms.IntrinsicsPoseToProjection(10, 4),
                      ]
    
    transforms_val = transforms.Compose(transform_val)

    # ------------------------------
    loss_weight = [1, 0.8, 0.64]

    dataset_train = ScanNetDataset(datapath=args.data_path, mode='train', transforms=transforms_train, length=1000)
    dataset_val = ScanNetDataset(datapath=args.data_path, mode='val', transforms=transforms_val, length=300)
    dataset_test = ScanNetDataset(datapath=args.data_path, mode='val', transforms=transforms_val, length=300)

    TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())

    writer = SummaryWriter('runs/tsdf'+ TIMESTAMP)

    model = ReconNet(fusion = True).to(DEVICE)

    checkpoint = torch.load('./checkpoint/tsdf/2000/checkpoint_tsdfattenlinear 200049.pth')
    model.load_state_dict(checkpoint['model_state'], strict = False)
    start_epoch= checkpoint['epoch']
    print('load checkpoint from {}!'.format(start_epoch))

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)

    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=1, num_workers=1, shuffle=False)
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=1, num_workers=1, shuffle=False)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, num_workers=1, shuffle=False)

    len_train = len(dataloader_train)
    len_val = len(dataloader_val)
    len_test = len(dataloader_test)

    if args.mode == 'train':
        print('start training!')
        # --- corase-to-fine training--
        for epoch in range(NUM_EPOCHS):
            print('current learning rate is {}'.format(optimizer.state_dict()['param_groups'][0]['lr']))
            train_loss_epoch = 0
            val_loss_epoch = 0
            dataloader_train.dataset.epoch = epoch
            dataloader_val.dataset.epoch = epoch       
            model.train()
            save_mesh = False
            save_mesh_epoch = False
            if (epoch+1)%2 == 0:
                save_mesh_epoch = True
            reset = False
            for i, input_info in enumerate(dataloader_train):
                reset = True if i == 0 else False
                if i == len_train-1 and save_mesh_epoch:
                    save_mesh = True
                loss_dict, outputs = model(input_info,save_mesh, save_mesh_epoch,reset)
                train_loss = 0
                if 'scene_tsdf' in outputs.keys():
                    print("visualizing this scene")
                    num_scene = len(outputs['scene_name'])
                    for index in range(num_scene):
                        occ2ply.save_scene(epoch, outputs, args.save_path,'train', batch_idx=index)
                for index, each in enumerate(loss_dict):
                    train_loss += (each * loss_weight[index])
                train_loss_epoch += train_loss.item()
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()
                writer.add_scalar('training loss iter',train_loss.item(),(epoch) * len_train + i)
                print('epoch:{}/{}, iteration:{}/{}, train loss:{}'.format(epoch, NUM_EPOCHS, i, len_train, train_loss.item()))

            model.eval()
            save_mesh=False
            save_mesh_epoch = False
            if (epoch+1)%60 == 0:
                save_mesh_epoch = True
            for j, input_info in enumerate(dataloader_val):
                reset = True if j == 0 else False
                if j == len_val- 1 and save_mesh_epoch:
                    save_mesh = True
                val_loss = 0
                with torch.no_grad():
                    loss_dict, outputs = model(input_info,save_mesh, save_mesh_epoch, reset)
                    if 'scene_tsdf' in outputs.keys():
                        print("visualizing this scene")
                        num_scene = len(outputs['scene_name'])
                        for index in range(num_scene):
                            occ2ply.save_scene(epoch, outputs, args.save_path,'val', batch_idx=index)
                    for z, each in enumerate(loss_dict):
                        val_loss += (each * loss_weight[z])
                    val_loss_epoch += val_loss.item()
                writer.add_scalar('validation loss iter', val_loss.item(), (epoch) * len_val + j)
                print('epoch:{}/{}, iteration:{}/{}, val loss:{}'.format(epoch, NUM_EPOCHS, j, len_val, val_loss.item()))
            writer.add_scalar('training loss epoch', train_loss_epoch/len_train, epoch)
            writer.add_scalar('val loss epoch', val_loss_epoch/len_val, epoch)
            print('epoch:{}/{}, train_loss:{}, val_loss:{}'.format(epoch, NUM_EPOCHS, train_loss_epoch/len_train,val_loss_epoch/len_val))
            scheduler.step()
            if (epoch+1)%1 == 0:
                save_checkpoint(model, optimizer, epoch)
            for clear in range(10):
                torch.cuda.empty_cache()
    #===test===
    elif args.mode == 'test':
        print('testing!')
        for epoch in range(1):
            test_loss_epoch = 0
            dataloader_test.dataset.epoch = epoch
            model.eval()
            save_mesh = False
            save_mesh_epoch = True
            reset =False
            for k, input_info in enumerate(dataloader_test):
                reset = True if k == 0 else False
                if k == len_test-1 and save_mesh_epoch: #save mesh every five epochs
                    save_mesh = True
                with torch.no_grad():
                    loss_dict, outputs = model(input_info,save_mesh, save_mesh_epoch, reset)
                    test_loss = 0
                    if 'scene_tsdf' in outputs.keys():
                        print("visualizing resulting scenes")
                        num_scene = len(outputs['scene_name'])
                        for index in range(num_scene):
                            occ2ply.save_scene(epoch, outputs, args.save_path,'test', batch_idx=index)
                    for idx, each in enumerate(loss_dict):
                        test_loss += (each * loss_weight[idx])
                    test_loss_epoch += test_loss.item()
                print('iteration:{}/{}, test loss:{}'.format(k, len_test, test_loss.item()))
            print('test_loss_epoch:{}'.format(test_loss_epoch/len_test))
            for clear in range(10):
                torch.cuda.empty_cache()
