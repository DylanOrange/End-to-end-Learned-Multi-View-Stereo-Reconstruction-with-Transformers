import numpy as np
import os
from plyfile import PlyData, PlyElement
from skimage import measure
import trimesh

def write_ply(points, face_data, filename, text=True):
    points = [(points[i,0], points[i,1], points[i,2]) for i in range(points.shape[0])]
    vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4')])

    face = np.empty(len(face_data),dtype=[('vertex_indices', 'i4', (4,))])
    face['vertex_indices'] = face_data

    ply_faces = PlyElement.describe(face, 'face')
    ply_vertexs = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    PlyData([ply_vertexs, ply_faces], text=text).write(filename)

def occ2points(occ, dim):
    points  = []
    for i in range(dim):
        for j in range(dim):
            for k in range(dim):
                if occ[i,j,k] == True:
                    points.append(np.array([i,j,k]))
    return np.array(points)

def generate_faces(points):
    corners = np.zeros((8*len(points),3))
    faces = np.zeros((6*len(points),4))
    for index in range(len(points)):
        corners[index*8]= np.array([points[index,0]-0.5, points[index,1]-0.5, points[index,2]-0.5])#左下后
        corners[index*8+1]= np.array([points[index,0]+0.5, points[index,1]-0.5, points[index,2]-0.5])#右下后
        corners[index*8+2]= np.array([points[index,0]-0.5, points[index,1]+0.5, points[index,2]-0.5])#左下前
        corners[index*8+3]= np.array([points[index,0]+0.5, points[index,1]+0.5, points[index,2]-0.5])#右下前
        corners[index*8+4]= np.array([points[index,0]-0.5, points[index,1]-0.5, points[index,2]+0.5])#左上后
        corners[index*8+5]= np.array([points[index,0]+0.5, points[index,1]-0.5, points[index,2]+0.5])#右上后
        corners[index*8+6]= np.array([points[index,0]-0.5, points[index,1]+0.5, points[index,2]+0.5])#左上前
        corners[index*8+7]= np.array([points[index,0]+0.5, points[index,1]+0.5, points[index,2]+0.5])#右上前
        base = len(points) + 8 * index
        faces[index * 6] = np.array([base + 2, base + 3, base + 1, base + 0])
        faces[index * 6 + 1] = np.array([base + 4, base + 5, base + 7, base + 6])
        faces[index * 6 + 2] = np.array([base + 3, base + 2, base + 6, base + 7])
        faces[index * 6 + 3] = np.array([base + 0, base + 1, base + 5, base + 4])
        faces[index * 6 + 4] = np.array([base + 2, base + 0, base + 4, base + 6])
        faces[index * 6 + 5] = np.array([base + 1, base + 3, base + 7, base + 5])
    return corners, faces

def writeocc(occ_data,save_path,filename):
    # valid_voxel = occ_data.astype(int)
    # valid_voxel = np.squeeze(valid_voxel)
    # valid_voxel = valid_voxel > 0
    # # print(valid_voxel.shape)
    # # for index in range(len(valid_voxel)):
    # #     if valid_voxel[index,0] == 1:
    # #         print(index)
    # #         break
    # # cnt_array = np.where(valid_voxel,0,1)
    # # print(np.sum(cnt_array)) #13000 points
    # xv, yv, zv = np.meshgrid(
    # np.arange(0, 64),
    # np.arange(0, 64),
    # np.arange(0, 64),indexing = 'ij')
    # vox_coords = np.stack([xv.flatten(), yv.flatten(), zv.flatten()], axis=1)
    # # print(vox_coords[60194,0])
    # valid_x = vox_coords[valid_voxel, 0]
    # valid_y = vox_coords[valid_voxel, 1]
    # valid_z = vox_coords[valid_voxel, 2]
    # cnt_array = np.where(valid_x,0,1)
    # #print(np.sum(cnt_array))
    # cnt_array = np.where(valid_y,0,1)
    # #print(np.sum(cnt_array))
    # cnt_array = np.where(valid_z,0,1)
    # #print(np.sum(cnt_array))
    # occ = np.zeros((64,64,64))
    # occ[valid_x,valid_y,valid_z] = 1
    # points = occ2points(occ,64)

    points = np.argwhere(occ_data>0)
    print(points.shape)
    corners, faces = generate_faces(points)
    if points.shape[0] == 0:
        print('the predicted mesh has zero point!')
    else:
        points = np.concatenate((points,corners),axis=0)
        write_ply(points, faces, os.path.join(save_path,filename))

def save_scene(epoch, outputs, save_path, mode, batch_idx=0):
    tsdf_volume = outputs['scene_tsdf'][batch_idx].data.cpu().numpy()
    origin = outputs['origin'][batch_idx].data.cpu().numpy()
    scene_name = outputs['scene_name'][batch_idx]
    voxel_size = 0.04

    if (tsdf_volume == 1).all():
        print('No valid data for scene {}'.format(scene_name))
    else:
        # Marching cubes
        mesh = tsdf2mesh(voxel_size, origin, tsdf_volume)
        # save tsdf volume for atlas evaluation
        # data = {'origin': origin,
        #         'voxel_size': self.cfg.MODEL.VOXEL_SIZE,
        #         'tsdf': tsdf_volume}
        filename = str(mode) + '_' + str(epoch) +'_' + scene_name
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # np.savez_compressed(
        #     os.path.join(save_path, '{}.npz'.format(self.scene_name)),
        #     **data)
        mesh.export(os.path.join(save_path, '{}.ply'.format(filename)))

def tsdf2mesh(voxel_size, origin, tsdf_vol):
    verts, faces, norms, vals = measure.marching_cubes(tsdf_vol, level=0)
    verts = verts * voxel_size + origin  # voxel grid coordinates to world coordinates
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=norms)
    return mesh
