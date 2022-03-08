
import open3d as o3d
import numpy as np


def occ2points(occ, dim):
    points  = []
    for i in range(dim):
        for j in range(dim):
            for k in range(dim):
                if occ[i,j,k] == True:
                # if occ[i,j,k] > 0.5:
                    points.append(np.array([i,j,k]))
    return np.array(points)

def eval_mesh(pred, gt, threshold=.05, down_sample=.02):
    """ Compute Mesh metrics between prediction and target.

    Opens the Meshs and runs the metrics

    Args:
        file_pred: file path of prediction
        file_trgt: file path of target
        threshold: distance threshold used to compute precision/recal
        down_sample: use voxel_downsample to uniformly sample mesh points

    Returns:
        Dict of mesh metrics
    """

    pcd_pred = o3d.io.read_point_cloud(pred)
    pcd_trgt = o3d.io.read_point_cloud(gt)
    if down_sample:
        pcd_pred = pcd_pred.voxel_down_sample(down_sample)
        pcd_trgt = pcd_trgt.voxel_down_sample(down_sample)
    verts_pred = np.asarray(pcd_pred.points)
    verts_trgt = np.asarray(pcd_trgt.points)

    _, dist1 = nn_correspondance(verts_pred, verts_trgt)
    _, dist2 = nn_correspondance(verts_trgt, verts_pred)
    dist1 = np.array(dist1)
    dist2 = np.array(dist2)

    precision = np.mean((dist1<threshold).astype('float'))
    recal = np.mean((dist2<threshold).astype('float'))
    fscore = 2 * precision * recal / (precision + recal)
    metrics = {'dist1': np.mean(dist1),
               'dist2': np.mean(dist2),
               'prec': precision,
               'recal': recal,
               'fscore': fscore,
               }
    return metrics


def nn_correspondance(verts1, verts2):
    """ for each vertex in verts2 find the nearest vertex in verts1

    Args:
        nx3 np.array's

    Returns:
        ([indices], [distances])

    """

    indices = []
    distances = []
    if len(verts1) == 0 or len(verts2) == 0:
        return indices, distances

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(verts1)
    kdtree = o3d.geometry.KDTreeFlann(pcd)

    for vert in verts2:
        _, inds, dist = kdtree.search_knn_vector_3d(vert, 1)
        indices.append(inds[0])
        distances.append(np.sqrt(dist[0]))

    return indices, distances

def evaluate(pred, gt, threshold=.05, down_sample=.02):
  B, _, _, _ = pred.shape
  precision = []
  recall = []
  fscore = []
  for i in range(B):
    precision_sample, recall_sample, fscore_sample = eval_mesh(pred[i], gt[i], threshold, down_sample)
    precision.append(precision_sample)
    recall.append(recall_sample)
    fscore.append(fscore_sample)
  return np.mean(precision), np.mean(recall), np.mean(fscore)


