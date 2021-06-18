
import os
import os.path as osp
import warnings

import sys
import cv2
import math
import time
import numpy as np
from PIL import Image
from scipy import interpolate
from scipy.spatial.distance import cdist
import torch
import shutil
import yaml
from collections import OrderedDict
import networkx as nx
import matplotlib.pyplot as plt
from torch_scatter import scatter_add
from torch.nn import functional as F



def compute_rounding(graph_obj, edges_out, probs, undirected_edges = False, return_flow_vals = False):
    """
    Determines the proportion of Flow Conservation inequalities that are satisfied.
    For each node, the sum of incoming (resp. outgoing) edge values must be less or equal than 1.

    Args:
        graph_obj: 'Graph' object
        edges_out: BINARIZED output values for edges (1 if active, 0 if not active)
        undirected_edges: determines whether each edge in graph_obj.edge_index appears in both directions (i.e. (i, j)
        and (j, i) are both present (undirected_edges =True), or only (i, j), with  i<j (undirected_edges=False)
        return_flow_vals: determines whether the sum of incoming /outglong flow for each node must be returned

    Returns:
        constr_sat_rate: float between 0 and 1 indicating the proprtion of inequalities that are satisfied

    """
    # Get tensors indicataing which nodes have incoming and outgoing flows (e.g. nodes in first frame have no in. flow)
    edge_ixs = graph_obj.edge_index
    if undirected_edges:
        sorted, _ = edge_ixs.t().sort(dim = 1)
        sorted = sorted.t()
        div_factor = 2. # Each edge is predicted twice, hence, we divide by 2
    else:
        sorted = edge_ixs # Edges (i.e. node pairs) are already sorted
        div_factor = 1.  # Each edge is predicted once, hence, hence we divide by 1.

    flag_rounding_needed = False
    # Compute incoming and outgoing flows for each node
    flow_out = scatter_add(edges_out, sorted[0],dim_size=graph_obj.num_nodes) / div_factor
    flow_in = scatter_add(edges_out, sorted[1], dim_size=graph_obj.num_nodes) / div_factor
    if 5 in flow_out:
        a=1

    nodes_flow_out = np.where(flow_out.cpu().numpy() > 3)
    nodes_flow_in = np.where(flow_in.cpu().numpy() > 3)

    if (len(nodes_flow_out[0]) != 0 or len(nodes_flow_in[0]) != 0):
        flag_rounding_needed = True
        new_predictions = edges_out.clone()
    else:
        new_predictions = []

    while flag_rounding_needed:
        edges_to_remove = []

        for n in nodes_flow_out[0]:
            pos =np.intersect1d(np.where(edge_ixs.cpu().numpy()[0] == n), np.where(new_predictions.cpu().numpy()==1)[0])
            remove_edge = pos[np.argmin(probs[pos].cpu().numpy())]
            edges_to_remove.append(remove_edge)

        for n in nodes_flow_in[0]:
            pos = np.intersect1d(np.where(edge_ixs.cpu().numpy()[1] == n), np.where(new_predictions.cpu().numpy()==1)[0])
            remove_edge = pos[np.argmin(probs[pos].cpu().numpy())]
            edges_to_remove.append(remove_edge)


        if edges_to_remove:
            new_predictions[edges_to_remove] = 0
        else:
            new_predictions = []

        #COMPUTE FLOW AGAIN TO CHECK
        flow_out = scatter_add(new_predictions, sorted[0], dim_size=graph_obj.num_nodes) / div_factor
        flow_in = scatter_add(new_predictions, sorted[1], dim_size=graph_obj.num_nodes) / div_factor

        nodes_flow_out = np.where(flow_out.cpu().numpy() > 3)
        nodes_flow_in = np.where(flow_in.cpu().numpy() > 3)

        if (len(nodes_flow_out[0]) != 0 or len(nodes_flow_in[0]) != 0):
            flag_rounding_needed = True
        else:
            flag_rounding_needed = False





    # Determine how many inequalitites are violated
    # violated_flow_out = (flow_out > 3).sum()
    # violated_flow_in = (flow_in > 3).sum()

    # Compute the final constraint satisfaction rate
    # violated_inequalities = (violated_flow_in + violated_flow_out).float()
    # flow_out_constr, flow_in_constr= sorted[0].unique(), sorted[1].unique()
    # num_constraints = len(flow_out_constr) + len(flow_in_constr)
    # constr_sat_rate = 1 - violated_inequalities / num_constraints
    # if constr_sat_rate.item() < 1:
    #     a=1
    #     # np.asarray([(edge_ixs[0][pos], edge_ixs[1][pos]) for pos, j in enumerate(edges_out) if j.item() == 1])
    # if not return_flow_vals:
    #     return constr_sat_rate.item(), new_predictions
    #
    # else:
    #     return constr_sat_rate.item(), flow_in, flow_out

    return new_predictions

class GreedyProjector:
    """
    Applies the greedy rounding scheme described in https://arxiv.org/pdf/1912.07515.pdf, Appending B.1
    """
    def __init__(self, full_graph):
        self.final_graph = full_graph.graph_obj
        self.num_nodes = full_graph.graph_obj.num_nodes

    def project(self):
        round_preds = (self.final_graph.edge_preds > 0.5).float()

        self.constr_satisf_rate, flow_in, flow_out =compute_constr_satisfaction_rate(graph_obj = self.final_graph,
                                                                                     edges_out = round_preds,
                                                                                     undirected_edges = False,
                                                                                     return_flow_vals = True)
        # Determine the set of constraints that are violated
        nodes_names = torch.arange(self.num_nodes).to(flow_in.device)
        in_type = torch.zeros(self.num_nodes).to(flow_in.device)
        out_type = torch.ones(self.num_nodes).to(flow_in.device)

        flow_in_info = torch.stack((nodes_names.float(), in_type.float())).t()
        flow_out_info = torch.stack((nodes_names.float(), out_type.float())).t()
        all_violated_constr = torch.cat((flow_in_info, flow_out_info))
        mask = torch.cat((flow_in > 1, flow_out > 1))

        # Sort violated constraints by the value of thei maximum pred value among incoming / outgoing edges
        all_violated_constr = all_violated_constr[mask]
        vals, sorted_ix = torch.sort(all_violated_constr[:, 1], descending=True)
        all_violated_constr = all_violated_constr[sorted_ix]

        # Iterate over violated constraints.
        for viol_constr in all_violated_constr:
            node_name, viol_type = viol_constr

            # Determine the set of incoming / outgoing edges
            mask = torch.zeros(self.num_nodes).bool()
            mask[node_name.int()] = True
            if viol_type == 0:  # Flow in violation
                mask = mask[self.final_graph.edge_index[1]]

            else:  # Flow Out violation
                mask = mask[self.final_graph.edge_index[0]]
            flow_edges_ix = torch.where(mask)[0]

            # If the constraint is still violated, set to 1 the edge with highest score, and set the rest to 0
            if round_preds[flow_edges_ix].sum() > 1:
                max_pred_ix = max(flow_edges_ix, key=lambda ix: self.final_graph.edge_preds[ix]*round_preds[ix]) # Multiply for round_preds so that if the edge has been set to 0
                                                                                                                 # it can not be set back to 1
                round_preds[mask] = 0
                round_preds[max_pred_ix] = 1

        # Assert that there are no constraint violations
        assert scatter_add(round_preds, self.final_graph.edge_index[1], dim_size=self.num_nodes).max() <= 1
        assert scatter_add(round_preds, self.final_graph.edge_index[0], dim_size=self.num_nodes).max() <= 1

        # return round_preds, constr_satisf_rate
        self.final_graph.edge_preds = round_preds




def visualize(h, color, edge_labels = None,edge_index =None ,node_label = None,epoch=None, loss=None):
    plt.figure(figsize=(7,7))
    plt.xticks([])
    plt.yticks([])

    pos = nx.spring_layout(h, seed=42)
    if torch.is_tensor(h):
        h = h.detach().cpu().numpy()
        plt.scatter(h[:, 0], h[:, 1], s=20, c=color, cmap="Set2")
        if epoch is not None and loss is not None:
            plt.xlabel(f'Epoch: {epoch}, Loss: {loss.item():.4f}', fontsize=16)
    elif edge_labels is None:

        nx.draw_networkx(h, pos=pos, with_labels=False,
                         node_color=color, cmap="Set3")

    elif edge_index is not None:
        list_active_edges = [(edge_index[0][i], edge_index[1][i]) for i in range(len(edge_labels)) if
                             edge_labels[i] == 1]
        list_nonactive_edges = [(edge_index[0][i], edge_index[1][i]) for i in range(len(edge_labels)) if
                                edge_labels[i] == 0]
        nx.draw_networkx_nodes(h, pos=pos,node_color=color,cmap="Set2")
        if node_label is not None:
            list_node_label = {n: node_label[n] for n in range(len(node_label))}
            nx.draw_networkx_labels(h, pos, labels=list_node_label, font_size=16)

        nx.draw_networkx_edges(h, pos=pos, edgelist=list_active_edges, edge_color='darkred')
        nx.draw_networkx_edges(h, pos=pos, edgelist=list_nonactive_edges, edge_color='lightgray',alpha=0.5)
        # nx.draw_networkx_edges(h, pos=pos, edge_list=edges_ixs_nonactives, edge_color='b')
    # elif node_label is not None:
    #     list_node_label = {n: node_label[n] for n in range(len(node_label))}
    #     nx.draw_networkx_labels(h, pos, labels=list_node_label, font_size=16)

    plt.show()
    a=1


def apply_homography_image_to_world(xi, yi, H_image_to_world):
    # Spatial vector xi, yi, 1
    S = np.array([xi, yi, 1]).reshape(3, 1)
    # Get 3 x 3 matrix and compute inverse
    # H_world_to_image = np.array(H_world_to_image).reshape(3, 3)
    # H_image_to_world = np.linalg.inv(H_world_to_image)

    # H_image_to_world = np.array(H_image_to_world).reshape(3, 3)

    # Dot product
    prj = np.dot(H_image_to_world, S)
    # Get world coordinates
    xw = (prj[0] / prj[2]).item() # latitude
    yw = (prj[1] / prj[2]).item() # longitude
    return xw, yw


def apply_homography_world_to_image(xi, yi, H_world_to_image):
    # Spatial vector xi, yi, 1
    S = np.array([xi, yi, 1]).reshape(3, 1)
    # Get 3 x 3 matrix and compute inverse
    H_world_to_image = np.array(H_world_to_image).reshape(3, 3)

    # Dot product
    prj = np.dot(H_world_to_image, S)
    # Get world coordinates
    xw = (prj[0] / prj[2]).item() # latitude
    yw = (prj[1] / prj[2]).item() # longitude
    return xw, yw

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.std = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def compute_SCC_and_Clusters(G,n_nodes):
    sets = [c for c in sorted(nx.strongly_connected_components(G), key=len, reverse=False)]

    # Add independent nodes to the list of CCs
    for i in range(n_nodes):
        flag = 0
        for s in sets:
            s = list(s)
            if i in s:
                flag = 1
                break
        if flag == 0:
            sets.append({i})

    ID_pred = np.zeros(shape=n_nodes, dtype=int)
    n_components_pred = sets.__len__()
    cluster = 0
    for s in sets:
        for i in list(s):
            ID_pred[i] = cluster
        cluster = cluster + 1
    #
    return ID_pred, n_components_pred

def save_checkpoint(state, is_best, path, filename):
    torch.save(state, path + '/files/' + filename + '_latest.pth.tar')

    if is_best:
        print('Best model updated.')
        shutil.copyfile(path + '/files/' + filename + '_latest.pth.tar',
                        path + '/files/' + filename + '_best.pth.tar')
        shutil.copyfile('config/config.yaml', path + '/files/config.yaml')

        # dict_file = {'TRAIN': {'ACCURACY': str(round(state['best_prec_train'], 2)) + ' %'},
        #              'VALIDATION': {'ACCURACY': str(round(state['best_prec_val'], 2)) + ' %'},
        #              'EPOCH': state['epoch'],
        #              'EPOCH TIME': str(round(state['time_per_epoch'], 2)) + ' Minutes',
        #              'COMMENTS': state['CONFIG']['MODEL']['COMMENTS'],
        #              'MODEL PARAMETERS': str(state['model_parameters']) + ' Millions',
        #              'DATASET': state['CONFIG']['DATASET']['NAME']}
        dict_file = {'TRAIN': {'ACCURACY': str(round(state['best_prec'], 2)) + ' %'},
                     'EPOCH': state['epoch'],
                     'MODEL PARAMETERS': str(state['model_parameters']) + ' Millions',
                     'DATASET': state['CONFIG']['DATASET_TRAIN']['NAME']}

        with open(os.path.join(path, 'Summary Report.yaml'), 'w') as file:
            yaml.safe_dump(dict_file, file)
    #
    # dict_mean_file = {'Last ten epochs avg training accuracy': str(state['10epoch_train_prec']),
    #                   'Last ten epochs avg testing accuracy': str(state['10epoch_test_prec'])
    #                   }
    # with open(os.path.join(path, 'Average precisions.yaml'), 'w') as file:
    #     yaml.safe_dump(dict_mean_file, file)


def load_checkpoint(fpath):
    r"""Loads checkpoint.

    ``UnicodeDecodeError`` can be well handled, which means
    python2-saved files can be read from python3.

    Args:
        fpath (str): path to checkpoint.

    Returns:
        dict


    """
    if fpath is None:
        raise ValueError('File path is None')
    if not osp.exists(fpath):
        raise FileNotFoundError('File is not found at "{}"'.format(fpath))
    map_location = None if torch.cuda.is_available() else 'cpu'

    checkpoint = torch.load(fpath, map_location=map_location)

    return checkpoint

def load_pretrained_weights(model, weight_path):
    r"""Loads pretrianed weights to model.

    Features::
        - Incompatible layers (unmatched in name or size) will be ignored.
        - Can automatically deal with keys containing "module.".

    Args:
        model (nn.Module): network model.
        weight_path (str): path to pretrained weights.

    Examples::

    """
    checkpoint = load_checkpoint(weight_path)
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    model_dict = model.state_dict()
    new_state_dict = OrderedDict()
    matched_layers, discarded_layers = [], []

    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:]  # discard module.

        if k in model_dict and model_dict[k].size() == v.size():
            new_state_dict[k] = v
            matched_layers.append(k)
        else:
            discarded_layers.append(k)

    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict, strict=True)

    if len(matched_layers) == 0:
        warnings.warn(
            'The pretrained weights "{}" cannot be loaded, '
            'please check the key names manually '
            '(** ignored and continue **)'.format(weight_path))
    else:
        print('Successfully loaded pretrained weights from "{}"'.format(weight_path))
        if len(discarded_layers) > 0:
            print('** The following layers are discarded '
                  'due to unmatched keys or layer size: {}'.format(discarded_layers))


    return model

#  FROM TOP DB
def compute_distance_matrix(input1, input2, metric='euclidean'):
    """A wrapper function for computing distance matrix.

    Args:
        input1 (torch.Tensor): 2-D feature matrix.
        input2 (torch.Tensor): 2-D feature matrix.
        metric (str, optional): "euclidean" or "cosine".
            Default is "euclidean".

    Returns:
        torch.Tensor: distance matrix.

    Examples::

    """
    # check input
    assert isinstance(input1, torch.Tensor)
    assert isinstance(input2, torch.Tensor)
    assert input1.dim() == 2, 'Expected 2-D tensor, but got {}-D'.format(input1.dim())
    assert input2.dim() == 2, 'Expected 2-D tensor, but got {}-D'.format(input2.dim())
    assert input1.size(1) == input2.size(1)

    if metric == 'euclidean':
        distmat = euclidean_squared_distance(input1, input2)
    elif metric == 'cosine':
        distmat = cosine_distance(input1, input2)
    else:
        raise ValueError(
            'Unknown distance metric: {}. '
            'Please choose either "euclidean" or "cosine"'.format(metric)
        )

    return distmat


def euclidean_squared_distance(input1, input2):
    """Computes euclidean squared distance.

    Args:
        input1 (torch.Tensor): 2-D feature matrix.
        input2 (torch.Tensor): 2-D feature matrix.

    Returns:
        torch.Tensor: distance matrix.
    """
    m, n = input1.size(0), input2.size(0)
    distmat = torch.pow(input1, 2).sum(dim=1, keepdim=True).expand(m, n) + \
              torch.pow(input2, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat.addmm_(1, -2, input1, input2.t())
    return distmat


def cosine_distance(input1, input2):
    """Computes cosine distance.

    Args:
        input1 (torch.Tensor): 2-D feature matrix.
        input2 (torch.Tensor): 2-D feature matrix.

    Returns:
        torch.Tensor: distance matrix.
    """
    input1_normed = F.normalize(input1, p=2, dim=1)
    input2_normed = F.normalize(input2, p=2, dim=1)
    distmat = 1 - torch.mm(input1_normed, input2_normed.t())
    return distmat


def re_ranking(q_g_dist, q_q_dist, g_g_dist, k1=20, k2=6, lambda_value=0.3):

    # The following naming, e.g. gallery_num, is different from outer scope.
    # Don't care about it.

    original_dist = np.concatenate(
      [np.concatenate([q_q_dist, q_g_dist], axis=1),
       np.concatenate([q_g_dist.T, g_g_dist], axis=1)],
      axis=0)
    original_dist = np.power(original_dist, 2).astype(np.float32)
    original_dist = np.transpose(1. * original_dist/np.max(original_dist,axis = 0))
    V = np.zeros_like(original_dist).astype(np.float32)
    initial_rank = np.argsort(original_dist).astype(np.int32)

    query_num = q_g_dist.shape[0]
    gallery_num = q_g_dist.shape[0] + q_g_dist.shape[1]
    all_num = gallery_num

    for i in range(all_num):
        # k-reciprocal neighbors
        forward_k_neigh_index = initial_rank[i,:k1+1]
        backward_k_neigh_index = initial_rank[forward_k_neigh_index,:k1+1]
        fi = np.where(backward_k_neigh_index==i)[0]
        k_reciprocal_index = forward_k_neigh_index[fi]
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_forward_k_neigh_index = initial_rank[candidate,:int(np.around(k1/2.))+1]
            candidate_backward_k_neigh_index = initial_rank[candidate_forward_k_neigh_index,:int(np.around(k1/2.))+1]
            fi_candidate = np.where(candidate_backward_k_neigh_index == candidate)[0]
            candidate_k_reciprocal_index = candidate_forward_k_neigh_index[fi_candidate]
            if len(np.intersect1d(candidate_k_reciprocal_index,k_reciprocal_index))> 2./3*len(candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index,candidate_k_reciprocal_index)

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        weight = np.exp(-original_dist[i,k_reciprocal_expansion_index])
        V[i,k_reciprocal_expansion_index] = 1.*weight/np.sum(weight)
    original_dist = original_dist[:query_num,]
    if k2 != 1:
        V_qe = np.zeros_like(V,dtype=np.float32)
        for i in range(all_num):
            V_qe[i,:] = np.mean(V[initial_rank[i,:k2],:],axis=0)
        V = V_qe
        del V_qe
    del initial_rank
    invIndex = []
    for i in range(gallery_num):
        invIndex.append(np.where(V[:,i] != 0)[0])

    jaccard_dist = np.zeros_like(original_dist,dtype = np.float32)


    for i in range(query_num):
        temp_min = np.zeros(shape=[1,gallery_num],dtype=np.float32)
        indNonZero = np.where(V[i,:] != 0)[0]
        indImages = []
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0,indImages[j]] = temp_min[0,indImages[j]]+ np.minimum(V[i,indNonZero[j]],V[indImages[j],indNonZero[j]])
        jaccard_dist[i] = 1-temp_min/(2.-temp_min)

    final_dist = jaccard_dist*(1-lambda_value) + original_dist*lambda_value
    del original_dist
    del V
    del jaccard_dist
    final_dist = final_dist[:query_num,query_num:]
    return final_dist
