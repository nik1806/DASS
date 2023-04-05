import numpy as np
import torch
import os
from lpmp_mc.raw_solvers import mwc_solver
import cv2

def readFloat(name):
    '''
        Adapted from https://github.com/lmb-freiburg/netdef_models
    '''
    f = open(name, 'rb')

    if(f.readline().decode("utf-8"))  != 'float\n':
        raise Exception('float file %s did not contain <float> keyword' % name)

    dim = int(f.readline())

    dims = []
    count = 1
    for i in range(0, dim):
        d = int(f.readline())
        dims.append(d)
        count *= d

    dims = list(reversed(dims))

    data = np.fromfile(f, np.float32, count).reshape(dims)
    if dim > 2:
        data = np.transpose(data, (2, 1, 0))
        data = np.transpose(data, (1, 0, 2))

    return data


def create_adj_index(curr_idx, H, W):
    edge_dist = 1
    neighbors = list()
    if curr_idx[0]+edge_dist < H:
        neighbors.append([curr_idx, (curr_idx[0]+edge_dist, curr_idx[1])]) 

    if curr_idx[1]+edge_dist < W:
        neighbors.append([curr_idx, (curr_idx[0], curr_idx[1]+edge_dist)]) 
    
    return neighbors


def mwc_refiner(pred, img_paths, bound_type='motion'):
    '''
    Refine the segmentation using motion boundaries
    Args:
        pred: logits from segmentation model, shape = (B, C, H, W)
        img_path: complete path to validation  dataset. Here it is Cityscapes
    '''
    # other variables
    motion_dir = "cityscapes_val_mb"
    edge_dir = "cityscapes_val_edge"
    if bound_type == 'motion':
        prefix = "mb_soft[0].fwd.float3"
    else:
        prefix = '.npy'


    # (H, W) = pred.size(2), pred.size(3)
    batch_size, K, H, W = pred.shape
    # print(pred.shape, type(pred), pred)

    # hyperparameters
    gamma = 1    

    ## calculate node cost
    prob = torch.softmax(pred, dim=1) # convert logit to probability
    # reshape prediction
    node_costs = -torch.log(prob)
    node_costs = torch.permute(node_costs, (0, 2, 3, 1)) # convert to (B, H, W, K) K = no. of classes
    node_costs = node_costs.reshape(node_costs.size(0), -1, node_costs.size(3)).cpu().detach().numpy()
    # print("node cost", node_costs.shape, node_costs[0].shape) # remove later

    # preparing for edge_indice and edge_cost matrix
    mat_idx = list(np.ndindex(H, W))

    E = list()
    for idx in mat_idx:
        E += create_adj_index(idx, H, W)
    # each index is numbered as single integer 
    idx_2d = np.arange(0, H*W).reshape(H, W) 

    edge_indices_1d = []
    for i, j in E:
        edge_indices_1d.append(np.array([idx_2d[i], idx_2d[j]]))

    edge_indices_1d = np.array(edge_indices_1d)
    # print("edge_indices", edge_indices_1d.shape)

    refined_labels = []
    # operations on single frames
    for i in range(batch_size):
        node_cost = node_costs[i]
        img_path = img_paths[i]
        
        if bound_type == 'motion': # motion
            float_path = os.path.join(motion_dir, *img_path.split('/')[-2:]) + prefix
            bound_soft = readFloat(float_path) # soft motion boundary
        else: # edge detector
            edge_path = os.path.join(edge_dir, *img_path.split('/')[-2:]) + prefix
            bound_soft = np.load(edge_path)

        bound_soft = cv2.resize(bound_soft.reshape(bound_soft.shape[:2]), (W, H))#.reshape(H, W)
        ## calculate edge cost
        # point_cost = -np.log(bound_soft) * gamma
        # print(point_cost) 

        edge_costs_1d = []
        for i, j in E:
            # edge_costs_1d.append((point_cost[i] + point_cost[j]))
            edge_costs_1d.append(-np.log((bound_soft[i] + bound_soft[j])/2) * gamma)

        edge_costs_1d = np.array(edge_costs_1d)
        # print(edge_costs_1d.reshape(edge_costs_1d.shape[0]).shape)

        ## Use MWC solver
        node_labels_indi, edge_labels, solver_cost = mwc_solver(edge_indices_1d , edge_costs_1d , node_cost)

        # get node labels
        node_labels = np.argmax(node_labels_indi.reshape((H, W, K)), axis=2) # shape = (H, W)
        refined_labels.append(node_labels)

    # print(torch.tensor(refined_labels).shape, refined_labels)
    # print(np.unique(np.array(refined_labels)))
    # exit()

    return torch.tensor(np.array(refined_labels)), bound_soft