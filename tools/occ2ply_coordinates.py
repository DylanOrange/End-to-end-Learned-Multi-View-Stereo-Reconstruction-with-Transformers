import numpy as np
import os
from plyfile import PlyData, PlyElement

def write_ply(points, face_data, filename, text=True):
    points = [(points[i,0], points[i,1], points[i,2]) for i in range(points.shape[0])]
    vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4')])

    face = np.empty(len(face_data),dtype=[('vertex_indices', 'i4', (4,))])
    face['vertex_indices'] = face_data

    ply_faces = PlyElement.describe(face, 'face')
    ply_vertexs = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    PlyData([ply_vertexs, ply_faces], text=text).write(filename)

def occ2points(coordinates):
    points  = []
    len = coordinates.shape[0]
    for i in range(len):
        points.append(np.array([int(coordinates[i,1]),int(coordinates[i,2]),int(coordinates[i,3])]))
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

def writeocc_coordinates(coordinates,save_path,filename):
    points = occ2points(coordinates)
    #print(points.shape)
    corners, faces = generate_faces(points)
    if points.shape[0] == 0:
        print('the predicted mesh has zero point!')
    else:
        points = np.concatenate((points,corners),axis=0)
        write_ply(points, faces, os.path.join(save_path,filename))

