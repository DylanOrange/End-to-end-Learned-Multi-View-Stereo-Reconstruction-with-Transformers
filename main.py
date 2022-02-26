import os
import numpy as np
import torch.cuda
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
from torch.cuda.amp import autocast as autocast


def get_target(occ):
    # 2 ** scale == interval
    xv, yv, zv = torch.meshgrid(
        torch.arange(0, 64),
        torch.arange(0, 64),
        torch.arange(0, 64),
    )
    vox_coords = torch.stack([xv.flatten(), yv.flatten(), zv.flatten()], dim=1)
    occ_target = occ[vox_coords[:, 0], vox_coords[:, 1], vox_coords[:, 2]]

    return occ_target


def deactivate_batchnorm(m):
    if isinstance(m, nn.BatchNorm2d):
        m.reset_parameters()
        m.eval()
        with torch.no_grad():
            m.weight.fill_(1.0)
            m.bias.zero_()


# class Logger():
#     def __init__(self, filename="log.txt"):
#         self.terminal = sys.stdout
#         self.log = open(filename, "w")

#     def write(self, message):
#         self.terminal.write(message)
#         self.log.write(message)

#     def flush(self):
#         pass


# sys.stdout = Logger()


def parse_args():
    parser = argparse.ArgumentParser(description='Generation of Ground Truth')
    parser.add_argument("--data_path", metavar="DIR", help='path to raw data', default='/mnt/hdd/praktikum/Deep-Learning-in-Visual-Computing/scannet')
    parser.add_argument("--save_path", metavar="DIR", help="file name", default="result/eval/oldatten")
    parser.add_argument("--max_depth", default=3, type=int)
    parser.add_argument("--margin", default=3, type=int)
    parser.add_argument("--voxel_size", default=0.08, type=float)
    parser.add_argument("--window_size", default=10, type=int)
    parser.add_argument("--min_angle", default=15, type=float)
    parser.add_argument("--min_distance", default=0.1, type=float)
    parser.add_argument("--pos_weight", default=1.5, type=float)
    parser.add_argument("--lr", default=5e-4, type=float)
    parser.add_argument("--train_number", default=10, type=int)
    parser.add_argument("--val_number", default=5, type=int)
    parser.add_argument("--clip", default=1.0, type=float)
    parser.add_argument("--dataset_length", default=20, type=int)
    parser.add_argument("--save", action="store_true", help="Run or not.")
    parser.add_argument("--mode", default="eval")
    return parser.parse_args()


args = parse_args()


def save_checkpoint(model, optimizer, epoch):
    checkpoint = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict()
    }
    FILE = 'checkpoint' + '_' + str(epoch) + '.pth'
    torch.save(checkpoint, os.path.join('checkpoint', 'fusion', '2000', 'atten',FILE))


# Hyperparameters
LEARNING_RATE = args.lr
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 1
NUM_EPOCHS = 40
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
                            [64, 64, 64], 0.08, True, True,
                            0.1, 0.025, max_epoch=NUM_EPOCHS),
                        transforms.IntrinsicsPoseToProjection(10, 4),
                        ]

    transforms_train = transforms.Compose(transform_train)

    transform_val = []
    transform_val += [transforms.ResizeImage((640, 480)),
                      transforms.ToTensor(),
                      transforms.RandomTransformSpace(
                          [64, 64, 64], 0.08, False, False,
                          0, 0, max_epoch=NUM_EPOCHS),
                      transforms.IntrinsicsPoseToProjection(10, 4),
                      ]
    
    transforms_val = transforms.Compose(transform_val)

    # ------------------------------
    loss_weight = [1, 0.8, 0.64]

    # dataset = ScanNetDataset(datapath=args.data_path, mode='train', transforms=transforms_train, length=5)
    # train_size = int(0.7*len(dataset))
    # val_size = len(dataset)-train_size
    # # rest_size = int(int(len(full_dataset)) - train_size - val_size)
    # dataset_train, dataset_val = torch.utils.data.random_split(dataset, [train_size, val_size])
    dataset_train = ScanNetDataset(datapath=args.data_path, mode='train', transforms=transforms_val, length=2000)
    dataset_val = ScanNetDataset(datapath=args.data_path, mode='train', transforms=transforms_val, length=300)
    dataset_test = ScanNetDataset(datapath=args.data_path, mode='train', transforms=transforms_val, length=61)

    TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
    writer = SummaryWriter('runs_neucon/'+TIMESTAMP)

    model = ReconNet(fusion = True).to(DEVICE)

    checkpoint = torch.load('/mnt/hdd/praktikum/ldy/checkpoint/fusion/200/attentionsimply/checkpoint_39.pth')
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
        for epoch in range(start_epoch+1, NUM_EPOCHS):
            print('current learning rate is {}'.format(optimizer.state_dict()['param_groups'][0]['lr']))
            train_loss_epoch = 0
            val_loss_epoch = 0
            # dataloader_train.dataset.dataset.epoch = epoch
            # dataloader_val.dataset.dataset.epoch = epoch
            dataloader_train.dataset.epoch = epoch
            dataloader_val.dataset.epoch = epoch       
            model.train()
            save_mesh = False
            save_mesh_epoch = False
            if (epoch+1)%6 == 0:
                save_mesh_epoch = True
            reset = False
            for i, input_info in enumerate(dataloader_train):
                reset = True if i == 0 else False
                if i == len_train-1 and save_mesh_epoch: #save mesh every five epochs
                    save_mesh = True
                loss_dict, outputs = model(input_info,save_mesh, save_mesh_epoch,reset)
                train_loss = 0
                if 'scene_occ' in outputs.keys():
                    print("visualizing this scene")
                    print(outputs['scene_occ'][0].shape)
                    scene_mesh = outputs['scene_occ'][0].cpu().detach().numpy()
                    occ2ply.writeocc(scene_mesh, args.save_path, 'training_{}_{}.ply'.format(outputs['scene_name'][0], epoch))
                for index, each in enumerate(loss_dict):
                    train_loss += (each * loss_weight[index])
                train_loss_epoch += train_loss.item()
                if ((epoch+1)%5 == 0 and (i+1)%3000 == 0): 
                    print("visualizing this fragment")
                    if 'coords' in outputs.keys():
                        gt = input_info['occ_list'][0].squeeze(0)
                        # gt = get_target(gt)
                        gt_mask = gt.cpu().numpy()
                        print(gt_mask.shape)
                        occ2ply.writeocc(gt_mask, args.save_path, 'train_gt_{}_{}.ply'.format(epoch, i))
                        coords = outputs['coords'].cpu().detach().numpy()
                        # pred_mask = pred_mask > 0.5
                        occ2ply_coordinates.writeocc_coordinates(coords, args.save_path, 'train_result_{}_{}.ply'.format(epoch, i))
                        print('finished!')
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()
                writer.add_scalar('training loss iter',train_loss.item(),(epoch) * len_train + i)
                print('epoch:{}/{}, iteration:{}/{}, train loss:{}'.format(epoch, NUM_EPOCHS, i, len_train, train_loss.item()))
                # print('epoch:{}/{}, iteration:{}/{}, train loss:{}, loss_16:{}, loss_32:{}, loss_64:{}'.format(epoch, NUM_EPOCHS, i, len_train, train_loss.item(),loss_dict[0].item(),
                #                                                                                        loss_dict[1].item() / 0.8, loss_dict[2].item() / 0.64))

            model.eval()
            save_mesh=False
            save_mesh_epoch = False
            if (epoch+1)%6 == 0:
                save_mesh_epoch = True
            for j, input_info in enumerate(dataloader_val):
                reset = True if j == 0 else False
                if j == len_val- 1 and save_mesh_epoch:
                    save_mesh = True
                val_loss = 0
                with torch.no_grad():
                    loss_dict, outputs = model(input_info,save_mesh, save_mesh_epoch, reset)
                    if 'scene_occ' in outputs.keys():
                        print("visualizing this scene")
                        print(outputs['scene_occ'][0].shape)
                        scene_mesh = outputs['scene_occ'][0].cpu().detach().numpy()
                        occ2ply.writeocc(scene_mesh, args.save_path, 'val_{}_{}.ply'.format(outputs['scene_name'][0], epoch))
                    for z, each in enumerate(loss_dict):
                        val_loss += (each * loss_weight[z])
                    if (j == 1000):
                        if 'coords' in outputs.keys():
                            gt = input_info['occ_list'][0].squeeze(0)
                            # gt = get_target(gt)
                            gt_mask = gt.cpu().numpy()
                            occ2ply.writeocc(gt_mask, args.save_path, 'val_gt_{}_{}.ply'.format(epoch, i))
                            coords = outputs['coords'].cpu().detach().numpy()
                            # pred_mask = pred_mask > 0.5
                            occ2ply_coordinates.writeocc_coordinates(coords, args.save_path,'val_result_{}_{}.ply'.format(epoch, i))
                    val_loss_epoch += val_loss.item()
                #             # gt_mask = gt.squeeze(0).cpu().detach().numpy()
                #             # precision_sample, recall_sample, fscore_sample = eval_mesh(pred_mask, gt_mask)
                #             # precision_iter.append(precision_sample)
                #             # recall_iter.append(recall_sample)
                #             # fscore_iter.append(fscore_sample)
                #         # precision = np.mean(precision_iter)
                #         # recall = np.mean(recall_iter)
                #         # fscore = np.mean(fscore_iter)
                writer.add_scalar('validation loss iter', val_loss.item(), (epoch) * len_val + j)
                # print('epoch:{}/{}, iteration:{}/{}, loss:{}, loss_16:{}, loss_32:{}, loss_64:{}'.format(epoch, NUM_EPOCHS, j, len_val, val_loss.item(),loss_dict[0].item(),
                #                                                                                        loss_dict[1].item() / 0.8, loss_dict[2].item() / 0.64))
                print('epoch:{}/{}, iteration:{}/{}, val loss:{}'.format(epoch, NUM_EPOCHS, j, len_val, val_loss.item()))
            writer.add_scalar('training loss epoch', train_loss_epoch/len_train, epoch)
            writer.add_scalar('val loss epoch', val_loss_epoch/len_val, epoch)
            print('epoch:{}/{}, train_loss:{}, val_loss:{}'.format(epoch, NUM_EPOCHS, train_loss_epoch/len_train,val_loss_epoch/len_val))
            #         #         #     # writer.add_scalar('Precision',
            #     #     #         #     #                   precision,
            #     #     #         #     #                   epoch * len(dataloader_train) + i)
            #     #     #         #     # writer.add_scalar('Recall',
            #     #     #         #     #                   recall,
            #     #     #         #     #                   epoch * len(dataloader_train) + i)
            #     #     #         #     # writer.add_scalar('F_Score',
            #     #     #         #     #                   fscore,
            #     #     #         #     #                   epoch * len(dataloader_train) + i)
            #     #     print('epoch:{}/{}, iteration:{}/{}, loss:{}, val_loss:{}'.format((epoch), NUM_EPOCHS, i,
            #     #                                                                       len(dataloader_train),
            #     #                                                                       train_loss.item(), val_loss.item()))
            #     #     model.train()
            scheduler.step()
            if (epoch+1)%2 == 0:
                save_checkpoint(model, optimizer, epoch)
            # scheduler.step()
            for clear in range(10):
                torch.cuda.empty_cache()
    #===test===
    elif args.mode == 'test':
        print('testing!')
        for epoch in range(1):
            test_loss_epoch = 0
            # dataloader_train.dataset.dataset.epoch = epoch
            # dataloader_val.dataset.dataset.epoch = epoch
            dataloader_test.dataset.epoch = epoch
            model.eval()
            save_mesh = False
            save_mesh_epoch = True
            # if (epoch+1)%1 == 0:
            #     save_mesh_epoch = True
            for k, input_info in enumerate(dataloader_test):
                if k == len_test-1 and save_mesh_epoch: #save mesh every five epochs
                    save_mesh = True
                with torch.no_grad():
                    loss_dict, outputs = model(input_info,save_mesh, save_mesh_epoch)
                    if (k+1)%100 == 0: 
                        print("visualizing this fragment")
                        if 'coords' in outputs.keys():
                            gt = input_info['occ_list'][0].squeeze(0)
                            gt_mask = gt.cpu().numpy()
                            print(gt_mask.shape)
                            occ2ply.writeocc(gt_mask, args.save_path, 'test_gt_{}_{}.ply'.format(epoch, k))
                            coords = outputs['coords'].cpu().detach().numpy()
                            occ2ply_coordinates.writeocc_coordinates(coords, args.save_path, 'test_result_{}_{}.ply'.format(epoch, k))
                            print('finished!')
                    test_loss = 0
                    if 'scene_occ' in outputs.keys():
                        print("visualizing resulting scenes")
                        num_scene = len(outputs['scene_name'])
                        for index in range(num_scene):
                            print(outputs['scene_occ'][index].shape)
                            print(outputs['scene_name'][index])
                            scene_mesh = outputs['scene_occ'][index].cpu().detach().numpy()
                            occ2ply.writeocc(scene_mesh, args.save_path, '{}.ply'.format(outputs['scene_name'][index]))
                    for idx, each in enumerate(loss_dict):
                        test_loss += (each * loss_weight[idx])
                    test_loss_epoch += test_loss.item()
                # if (k+1)%5 == 0: 
                #     print("visualizing this fragment")
                #     if 'coords' in outputs.keys():
                #         gt = input_info['occ_list'][0].squeeze(0)
                #         gt_mask = gt.cpu().numpy()
                #         print(gt_mask.shape)
                #         occ2ply.writeocc(gt_mask, args.save_path, 'train_gt_{}_{}.ply'.format(epoch, k))
                #         coords = outputs['coords'].cpu().detach().numpy()
                #         occ2ply_coordinates.writeocc(coords, args.save_path, 'train_result_{}_{}.ply'.format(epoch, k))
                #         print('finished!')
                # writer.add_scalar('test loss iter',test_loss.item(),(epoch) * len_test + k)
                print('iteration:{}/{}, test loss:{}'.format(k, len_test, test_loss.item()))
            print('test_loss_epoch:{}'.format(test_loss_epoch/len_test))
            for clear in range(10):
                torch.cuda.empty_cache()
    else:
        print("evaluating!")
        for epoch in range(1):
            test_loss_epoch = 0
            # dataloader_train.dataset.dataset.epoch = epoch
            # dataloader_val.dataset.dataset.epoch = epoch
            dataloader_test.dataset.epoch = epoch
            model.eval()
            save_mesh = False
            save_mesh_epoch = True
            # if (epoch+1)%1 == 0:
            #     save_mesh_epoch = True
            for k, input_info in enumerate(dataloader_test):
                reset = True if k == 0 else False
                if k == len_test-1 and save_mesh_epoch: #save mesh every five epochs
                    save_mesh = True
                with torch.no_grad():
                    loss_dict, outputs = model(input_info,save_mesh, save_mesh_epoch, reset)
                    test_loss = 0
                    if 'scene_occ' in outputs.keys():
                        print("visualizing resulting scenes")
                        num_scene = len(outputs['scene_name'])
                        for index in range(num_scene):
                            print(outputs['scene_occ'][index].shape)
                            print(outputs['scene_name'][index])
                            scene_mesh = outputs['scene_occ'][index].cpu().detach().numpy()
                            with open(os.path.join(args.save_path, '{}_{}_eval.npy'.format(outputs['scene_name'][index], start_epoch)), 'wb') as f:
                                np.save(f, scene_mesh)
                            occ2ply.writeocc(scene_mesh, args.save_path, '{}_{}_eval.ply'.format(outputs['scene_name'][index], start_epoch))
                    for idx, each in enumerate(loss_dict):
                        test_loss += (each * loss_weight[idx])
                    test_loss_epoch += test_loss.item()
                # writer.add_scalar('test loss iter',test_loss.item(),(epoch) * len_test + k)
                print('iteration:{}/{}, test loss:{}'.format(k, len_test, test_loss.item()))
            print('test_loss_epoch:{}'.format(test_loss_epoch/len_test))
            for clear in range(10):
                torch.cuda.empty_cache()
