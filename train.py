
import os
import time
import shutil
import yaml
import datetime

import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pandas as pd
import torch
import argparse
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import torch.optim as optim
import torch.nn.functional as F
from torch import optim as optim_module
# import imgaug as ia
# from imgaug import augmenters as iaa

from PIL import Image
from torch.utils.data import DataLoader,Dataset
from torch_geometric.data import Data, Batch

from datasets import datasets
from models.resnet import resnet50_fc256, load_pretrained_weights
from models.mpn import MOTMPNet

from torch_geometric.utils import to_networkx
import networkx as nx
from skimage.io import imread

import utils
from sklearn.metrics.pairwise import paired_distances
from scipy.sparse.csgraph import connected_components
from sklearn import metrics
import utils


list_cam_colors = list(['royalblue', 'darkorange','green','firebrick'])


def compute_loss_acc(outputs, batch, mode):
    # Define Balancing weight
    positive_vals = batch.edge_labels.sum()

    if positive_vals:
        pos_weight = (batch.edge_labels.shape[0] - positive_vals) / positive_vals

    else:  # If there are no positives labels, avoid dividing by zero
        pos_weight = 0

    # Compute Weighted BCE:
    loss = 0
    precision_class1 = list()
    precision_class0 = list()
    precision_all = list()

    num_steps = len(outputs['classified_edges'])
    for step in range(num_steps):
        preds = outputs['classified_edges'][step].view(-1)
        labels = batch.edge_labels.view(-1)
        if mode == 'train':
            # loss += F.binary_cross_entropy_with_logits(preds,labels,pos_weight=pos_weight)
            loss += F.binary_cross_entropy_with_logits(preds, labels)
        else:
            loss += F.binary_cross_entropy_with_logits(preds, labels)


        with torch.no_grad():
            sig = torch.nn.Sigmoid()
            preds_prob = sig(preds)
            predictions = (preds_prob >= 0.5) * 1

            # Precision class 1
            index_label_1 = np.where(np.asarray(labels.cpu()) == 1)
            sum_successes_1 = np.sum(predictions.cpu().numpy()[index_label_1] == labels.cpu().numpy()[index_label_1])
            if sum_successes_1 == 0:
                precision_class1.append(0)
            else:
                precision_class1.append((sum_successes_1 / len(labels[index_label_1])) * 100.0)

            # Precision class 0
            index_label_0 = np.where(np.asarray(labels.cpu()) == 0)
            sum_successes_0 = np.sum(predictions.cpu().numpy()[index_label_0] == labels.cpu().numpy()[index_label_0])
            if sum_successes_0 == 0:
                precision_class0.append(0)
            else:
                precision_class0.append((sum_successes_0 / len(labels[index_label_0])) * 100.0)

            # Precision
            sum_successes = np.sum(predictions.cpu().numpy() == labels.cpu().numpy())
            if sum_successes == 0:
                precision_all.append(0)
            else:
                precision_all.append((sum_successes / len(labels) )* 100.0)





    return loss, precision_class1, precision_class0, precision_all


def compute_P_R_F(preds, labels):
    index_label_1 = np.where(labels == 1)
    index_label_0 = np.where(labels == 0)

    # TP
    TP = np.sum(preds[index_label_1] == 1)
    # FP
    FP = np.sum(preds[index_label_0] == 1)
    # TN
    TN = np.sum(preds[index_label_0] == 0)
    # FN
    FN = np.sum(preds[index_label_1] == 0)
    # P, R , Fscore
    if (TP + FP) != 0:
        P = TP / (TP + FP)
    else:
        P = 0

    if (TP + FN) != 0:
        R = TP / (TP + FN)
    else:
        R = 0

    if (P + R) != 0:
        F = 2 * (P * R) / (P + R)
    else:
        F = 0

    return TP, FP, TN, FN, P,R, F

def train(CONFIG, train_loader, cnn_model, mpn_model, epoch, optimizer,results_path,train_loss_in_history,train_prec1_in_history,train_prec0_in_history, train_prec_in_history, train_dataset, dataset_dir ):

    train_losses = utils.AverageMeter('losses', ':.4e')
    train_batch_time = utils.AverageMeter('batch_time', ':6.3f')
    train_precision_class1 = utils.AverageMeter('Precision_class1', ':6.2f')
    train_precision_class0 = utils.AverageMeter('Precision_class0', ':6.2f')
    train_precision = utils.AverageMeter('Precision', ':6.2f')
    mpn_model.train()


    for i, data in enumerate(train_loader):
        if i >= 0:

            start_time = time.time()

            ########### Data extraction ###########
            [bboxes, data_df] = data

            len_graphs = [len(item) for item in data_df]
            with torch.no_grad():
                if CONFIG['CNN_MODEL']['arch'] == 'resnet50':
                    node_embeds, reid_embeds = cnn_model(torch.cat(bboxes, dim=0).cuda())
                else:
                    node_embeds = cnn_model(torch.cat(bboxes, dim=0).cuda())
                    reid_embeds= node_embeds


            max_counter = 0
            prev_max_counter = 0
            edge_ixs = []
            node_label = []
            node_id_cam = []

            # create list called batch, each element: a graph
            batch = []

            flag_visualize = False
            for g in range(len(len_graphs)):

                # For visualize, we need a complete structure containing all the graphs info concatenated
                if flag_visualize:
                    data_df[g]['node'] = np.asarray(range(max_counter, max_counter + len(data_df[g].values)))
                    max_counter = max(data_df[g]['node'].values) + 1
                    node_label.append(torch.from_numpy(data_df[g]['id'].values))
                    node_id_cam.append(torch.from_numpy(data_df[g]['id_cam'].values))
                    # Compute nodes connections (edges) without connecting nodes of same caemra
                    for id_cam in np.unique(data_df[g]['id_cam']):
                        ids_in_cam = data_df[g]['node'][data_df[g]['id_cam'] == id_cam].values
                        ids_out_cam = data_df[g]['node'][data_df[g]['id_cam'] != id_cam].values
                        edge_ixs.append(torch.cartesian_prod(torch.from_numpy(ids_in_cam), torch.from_numpy(ids_out_cam)))


                else:  # For computing the batch processing, we need a list of independent graphs
                    # data_df[g].loc[:,'node'] = np.asarray(range(max_counter, max_counter + len(data_df[g])))
                    data_df[g] = data_df[g].assign(node=np.asarray(range(max_counter, max_counter + len(data_df[g]))))

                    max_counter = max(data_df[g]['node'].values) + 1
                    node_embeds_g = torch.stack([node_embeds[i] for i in data_df[g]['node']])
                    edge_ixs_g = []
                    for id_cam in np.unique(data_df[g]['id_cam']):
                        ids_in_cam = data_df[g]['node'][data_df[g]['id_cam'] == id_cam].values
                        ids_out_cam = data_df[g]['node'][data_df[g]['id_cam'] != id_cam].values
                        edge_ixs_g.append(torch.cartesian_prod(torch.from_numpy(ids_in_cam), torch.from_numpy(ids_out_cam)))

                    # Compute edges attributes
                    edge_ixs_g = torch.cat(edge_ixs_g, dim=0).T.cuda()
                    edge_ixs_g_np = edge_ixs_g.cpu().numpy()

                    node_label_g = torch.from_numpy(data_df[g]['id'].values)
                    # node_label_g_np = node_label_g.numpy()

                    # features reid distances between each pair of points
                    emb_dist_g = F.pairwise_distance(reid_embeds[edge_ixs_g[0]], reid_embeds[edge_ixs_g[1]]).view(-1, 1)
                    node_dist_g = F.pairwise_distance(node_embeds[edge_ixs_g[0]], node_embeds[edge_ixs_g[1]]).view(-1, 1)
                    emb_dist_g_cos = F.cosine_similarity(reid_embeds[edge_ixs_g[0]], reid_embeds[edge_ixs_g[1]]).view(-1, 1)
                    node_dist_g_cos = F.cosine_similarity(node_embeds[edge_ixs_g[0]], node_embeds[edge_ixs_g[1]]).view(-1, 1)


                    # coordinates of each pair of points

                    xws_1 = np.expand_dims(np.asarray([data_df[g]['xw'].values[item-prev_max_counter] for item in edge_ixs_g_np[0]]), axis = 1)
                    yws_1 = np.expand_dims(np.asarray([data_df[g]['yw'].values[item-prev_max_counter]for item in edge_ixs_g_np[0]]), axis = 1)
                    xws_2 = np.expand_dims(np.asarray([data_df[g]['xw'].values[item-prev_max_counter] for item in edge_ixs_g_np[1]]), axis = 1)
                    yws_2 = np.expand_dims(np.asarray([data_df[g]['yw'].values[item-prev_max_counter] for item in edge_ixs_g_np[1]]), axis = 1)

                    points1 = np.concatenate((xws_1, yws_1), axis=1)
                    points2 = np.concatenate((xws_2, yws_2), axis=1)

                    spatial_dist_g = torch.unsqueeze((torch.from_numpy(paired_distances(points1, points2))), dim=1).cuda()
                    spatial_dist_x = torch.abs(torch.from_numpy(xws_1 - xws_2)).cuda()
                    spatial_dist_y = torch.abs(torch.from_numpy(yws_1 - yws_2)).cuda()


                    if CONFIG['TRAINING']['ONLY_APPEARANCE']:
                        edge_attr = torch.cat((emb_dist_g, node_dist_g, emb_dist_g_cos, node_dist_g_cos),   dim=1)
                        # edge_attr = torch.cat((emb_dist_g, node_dist_g), dim=1)

                    elif CONFIG['TRAINING']['ONLY_DIST']:
                        edge_attr = torch.cat((spatial_dist_g.type(torch.float32), spatial_dist_x.type(torch.float32), spatial_dist_y.type(torch.float32)), dim=1)

                    else:
                        edge_attr = torch.cat((spatial_dist_g.type(torch.float32), spatial_dist_x.type(torch.float32), spatial_dist_y.type(torch.float32), emb_dist_g), dim=1)
                    # EDGE LABELS

                    edge_labels_g = torch.from_numpy(
                        np.asarray([1 if (data_df[g]['id'].values[data_df[g]['node'].values == edge_ixs_g_np[0][i]] ==
                                          data_df[g]['id'].values[data_df[g]['node'].values == edge_ixs_g_np[1][i]]) else 0
                                    for i in range(edge_ixs_g_np.shape[1])])).type(torch.float).cuda()

                    # bajar rango a 0 de edge_iuxs_g
                    edge_ixs_g = edge_ixs_g - torch.min(edge_ixs_g)
                    data = Data(x=node_embeds_g, edge_index=edge_ixs_g, y=node_label_g, edge_attr=edge_attr,
                                edge_labels=edge_labels_g)

                    batch.append(data)
                    prev_max_counter = max_counter

            if flag_visualize:
                # EDGE CONNECTIONS
                edge_ixs = torch.cat(edge_ixs, dim=0).T
                edge_ixs_np = edge_ixs.numpy()

                # NODE LABEL NUMBER OF NODE
                node_label = torch.cat(node_label, dim=0)
                node_label_np = node_label.numpy()

                # NODE LABEL NUMBER OF CAMERA
                node_id_cam = torch.cat(node_id_cam, dim=0)
                node_id_cam_np = node_id_cam.numpy()

                # EDGE LABELS
                edge_labels = torch.tensor(
                    [1 if (node_label_np[edge_ixs_np[0][i]] == node_label_np[edge_ixs_np[1][i]]) else 0 for i in
                     range(edge_ixs_np.shape[1])])
                edge_labels_np = np.asarray(
                    [1 if (node_label_np[edge_ixs_np[0][i]] == node_label_np[edge_ixs_np[1][i]]) else 0 for i in
                     range(edge_ixs_np.shape[1])])

                # formatted_edge_labels = {(edge_ixs_np[0][i], edge_ixs_np[1][i]): edge_labels_np[i] for i in range(len(edge_labels))}

                # EDGE FEATURES : pairwise feature distance
                emb_dist = F.pairwise_distance(reid_embeds[edge_ixs[0]], reid_embeds[edge_ixs[1]]).view(-1, 1)

                edge_feats = emb_dist

                # GRAPH CREATION
                data = Data(x=node_embeds, edge_index=edge_ixs, y=node_label, edge_feats=edge_feats)
                G = to_networkx(data, to_undirected=True)
                utils.visualize(G, color=data.y, edge_labels = edge_labels_np, edge_index = edge_ixs_np, node_label = node_id_cam_np )

                # PLOT GROUNDPLANE DETECTIONS
                frame = train_dataset.frames_valid[1175]

                # GET each camera view plane

                plt.figure()
                plt.title('Detections in frame ' + str(data_df[0]['frame'].values[0]))
                list_legend = list()

                # CAM VIEW
                for id_cam in np.unique(data_df[0]['id_cam']):
                    # plt.fill(train_dataset.list_corners_x[id_cam], train_dataset.list_corners_y[id_cam], c=list_cam_colors[id_cam], alpha=0.2)

                    # TO SHOW THE FRAME WARPED
                    plt.figure()
                    homog_file = os.path.join(dataset_dir, train_dataset.cameras[id_cam], 'Homography.txt')
                    H = np.asarray(pd.read_csv(homog_file, header=None, sep="\t"))
                    frame_path = os.path.join(dataset_dir, train_dataset.cameras[id_cam], 'img1',
                                              str(frame).zfill(6) + '.jpg')
                    img = imread(frame_path)
                    dst = cv2.warpPerspective(img, H, (500, 500))
                    plt.imshow(dst)
                    plt.fill(train_dataset.list_corners_x[id_cam], train_dataset.list_corners_y[id_cam],
                             c=list_cam_colors[id_cam], alpha=0.2)
                    data_cam = data_df[0][data_df[0]['id_cam'] == id_cam]
                    xw_in_cam = np.asarray(data_cam['xw'])
                    yw_in_cam = np.asarray(data_cam['yw'])
                    plt.scatter(xw_in_cam, yw_in_cam, c=list_cam_colors[id_cam])
                    plt.show()
                    a = 1

                # DETECTIONS POINT
                for id_cam in np.unique(data_df[0]['id_cam']):
                    data_cam = data_df[0][data_df[0]['id_cam'] == id_cam]
                    xw_in_cam = np.asarray(data_cam['xw'])
                    yw_in_cam = np.asarray(data_cam['yw'])
                    plt.scatter(yw_in_cam, xw_in_cam, c=list_cam_colors[id_cam])
                    list_legend.append('camera ' + str(id_cam))

                plt.legend(list_legend)

                # TEXT WITH ID
                for id_cam in np.unique(data_df[0]['id_cam']):
                    data_cam = data_df[0][data_df[0]['id_cam'] == id_cam]
                    xw_in_cam = np.asarray(data_cam['xw'])
                    yw_in_cam = np.asarray(data_cam['yw'])
                    for i in range(len(xw_in_cam)):
                        plt.text(yw_in_cam[i] + 1, xw_in_cam[i] + 1, str(data_cam['id'].values[i]))

                plt.show()
                a = 1
            # TRAINING #

            data_batch = Batch.from_data_list(batch)

            ########### Forward ###########

            outputs = mpn_model(data_batch)

            ########### Loss ###########

            loss, precision1, precision0, precision = compute_loss_acc(outputs, data_batch,mode='train')
            train_losses.update(loss.item(), CONFIG['TRAINING']['BATCH_SIZE']['TRAIN'])
            train_precision_class1.update(np.sum(np.asarray([item for item in precision1])) / len(precision1),CONFIG['TRAINING']['BATCH_SIZE']['TRAIN'] )
            train_precision_class0.update(np.sum(np.asarray([item for item in precision0])) / len(precision0),CONFIG['TRAINING']['BATCH_SIZE']['TRAIN'] )
            train_precision.update(np.sum(np.asarray([item for item in precision])) / len(precision),CONFIG['TRAINING']['BATCH_SIZE']['TRAIN'] )


            # accuracies.append()
            train_loss_in_history.append(train_losses.avg)
            train_prec1_in_history.append(np.sum(np.asarray([item for item in precision1])) / len(precision1))
            train_prec0_in_history.append(np.sum(np.asarray([item for item in precision0])) / len(precision0))
            train_prec_in_history.append(np.sum(np.asarray([item for item in precision])) / len(precision))

            ########### Accuracy ###########

            ########### Optimizer update ###########

            optimizer.zero_grad()
            loss.backward()
            optimizer.step(lambda: float(loss))

            train_batch_time.update(time.time() - start_time)

            if i % 10 == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Batch Time {batch_time.val:.3f} (avg: {batch_time.avg:.3f})\t'
                      'Train Loss {loss.val:.3f} (avg: {loss.avg:.3f})\t'
                      'Train Acc 1 {acc.val:.3f} (avg: {acc.avg:.3f})\t'
                      'Train Acc 0 Act {acc2.val:.3f} (avg: {acc2.avg:.3f})\t'
                      '{et}<{eta}'.format(epoch, i, len(train_loader), batch_time=train_batch_time,  loss=train_losses,
                                          acc = train_precision_class1, acc2 =train_precision_class0, et=str(datetime.timedelta(seconds=int(train_batch_time.sum))),
                                          eta=str(datetime.timedelta(seconds=int(train_batch_time.avg * (len(train_loader) - i))))))




    plt.figure()
    plt.plot(train_loss_in_history, label='Loss')

    plt.ylabel('Loss'), plt.xlabel('Iteration')
    plt.legend(loc='best')
    plt.savefig(results_path + '/images/Training Loss per Iteration.pdf', bbox_inches='tight')
    plt.close()

    plt.figure()
    plt.plot(train_prec1_in_history,'g', label='Precision class 1')
    plt.plot(train_prec0_in_history, 'r', label='Precission class 0')

    plt.ylabel('Precision'), plt.xlabel('Iteration')
    plt.legend(loc= 'best')
    plt.savefig(results_path + '/images/Training Precision per Iteration.pdf', bbox_inches='tight')
    plt.close()

    return train_losses, train_precision_class1, train_precision_class0, train_loss_in_history,train_prec1_in_history,train_prec0_in_history,train_prec_in_history


def validate(CONFIG, val_loader, cnn_model, mpn_model, results_path,epoch,val_loss_in_history,val_prec1_in_history,val_prec0_in_history,val_prec_in_history, val_dataset, dataset_dir):
    val_losses = utils.AverageMeter('losses', ':.4e')
    val_batch_time = utils.AverageMeter('batch_time', ':6.3f')
    val_precision_1 = utils.AverageMeter('Val prec class 1', ':6.2f')
    val_precision_0 = utils.AverageMeter('Val prec class 0', ':6.2f')
    val_precision = utils.AverageMeter('Val prec', ':6.2f')

    mpn_model.eval()
    cnn_model.eval()


    with torch.no_grad():
        for i, data in enumerate(val_loader):
            if i >= 0:

                start_time = time.time()

                ########### Data extraction ###########
                [bboxes, data_df] = data

                len_graphs = [len(item) for item in data_df]
                if CONFIG['CNN_MODEL']['arch'] == 'resnet50':
                    node_embeds, reid_embeds = cnn_model(torch.cat(bboxes, dim=0).cuda())
                else:
                    node_embeds = cnn_model(torch.cat(bboxes, dim=0).cuda())
                    reid_embeds= node_embeds



                max_counter = 0
                prev_max_counter  = 0
                edge_ixs = []
                node_label = []
                node_id_cam = []

                # create list called batch, each element: a graph
                batch = []

                flag_visualize = False
                for g in range(len(len_graphs)):

                    # For visualize, we need a complete structure containing all the graphs info concatenated
                    if flag_visualize:
                        data_df[g]['node'] = np.asarray(range(max_counter, max_counter + len(data_df[g].values)))
                        max_counter = max(data_df[g]['node'].values) + 1
                        node_label.append(torch.from_numpy(data_df[g]['id'].values))
                        node_id_cam.append(torch.from_numpy(data_df[g]['id_cam'].values))
                        # Compute nodes connections (edges) without connecting nodes of same caemra
                        for id_cam in np.unique(data_df[g]['id_cam']):
                            ids_in_cam = data_df[g]['node'][data_df[g]['id_cam'] == id_cam].values
                            ids_out_cam = data_df[g]['node'][data_df[g]['id_cam'] != id_cam].values
                            edge_ixs.append(
                                torch.cartesian_prod(torch.from_numpy(ids_in_cam), torch.from_numpy(ids_out_cam)))


                    else:  # For computing the batch processing, we need a list of independent graphs
                        # data_df[g].loc[:,'node'] = np.asarray(range(max_counter, max_counter + len(data_df[g])))
                        data_df[g] = data_df[g].assign(
                            node=np.asarray(range(max_counter, max_counter + len(data_df[g]))))

                        max_counter = max(data_df[g]['node'].values) + 1
                        node_embeds_g = torch.stack([node_embeds[i] for i in data_df[g]['node']])
                        edge_ixs_g = []
                        for id_cam in np.unique(data_df[g]['id_cam']):
                            ids_in_cam = data_df[g]['node'][data_df[g]['id_cam'] == id_cam].values
                            ids_out_cam = data_df[g]['node'][data_df[g]['id_cam'] != id_cam].values
                            edge_ixs_g.append(
                                torch.cartesian_prod(torch.from_numpy(ids_in_cam), torch.from_numpy(ids_out_cam)))

                        # Compute edges attributes
                        edge_ixs_g = torch.cat(edge_ixs_g, dim=0).T.cuda()
                        edge_ixs_g_np = edge_ixs_g.cpu().numpy()

                        node_label_g = torch.from_numpy(data_df[g]['id'].values)
                        # node_label_g_np = node_label_g.numpy()

                        # features reid distances between each pair of points
                        emb_dist_g = F.pairwise_distance(reid_embeds[edge_ixs_g[0]], reid_embeds[edge_ixs_g[1]]).view(
                            -1, 1)
                        node_dist_g = F.pairwise_distance(node_embeds[edge_ixs_g[0]], node_embeds[edge_ixs_g[1]]).view(
                            -1, 1)

                        emb_dist_g_cos = F.cosine_similarity(reid_embeds[edge_ixs_g[0]],
                                                             reid_embeds[edge_ixs_g[1]]).view(-1, 1)
                        node_dist_g_cos = F.cosine_similarity(node_embeds[edge_ixs_g[0]],
                                                              node_embeds[edge_ixs_g[1]]).view(-1, 1)

                        # coordinates of each pair of points
                        xws_1 = np.expand_dims(np.asarray([data_df[g]['xw'].values[item - prev_max_counter] for item in edge_ixs_g_np[0]]), axis=1)
                        yws_1 = np.expand_dims(np.asarray([data_df[g]['yw'].values[item - prev_max_counter] for item in edge_ixs_g_np[0]]),  axis=1)
                        xws_2 = np.expand_dims(np.asarray([data_df[g]['xw'].values[item - prev_max_counter] for item in edge_ixs_g_np[1]]),  axis=1)
                        yws_2 = np.expand_dims(np.asarray([data_df[g]['yw'].values[item - prev_max_counter] for item in edge_ixs_g_np[1]]),    axis=1)
                        points1 = np.concatenate((xws_1, yws_1), axis=1)
                        points2 = np.concatenate((xws_2, yws_2), axis=1)

                        spatial_dist_g = torch.unsqueeze((torch.from_numpy(paired_distances(points1, points2))),
                                                         dim=1).cuda()
                        spatial_dist_x = torch.abs(torch.from_numpy(xws_1 - xws_2)).cuda()
                        spatial_dist_y = torch.abs(torch.from_numpy(yws_1 - yws_2)).cuda()

                        if CONFIG['TRAINING']['ONLY_APPEARANCE']:
                            edge_attr = torch.cat((emb_dist_g, node_dist_g, emb_dist_g_cos, node_dist_g_cos), dim=1)
                            # edge_attr = torch.cat((emb_dist_g, node_dist_g), dim=1)

                        elif CONFIG['TRAINING']['ONLY_DIST']:
                            edge_attr = torch.cat((spatial_dist_g.type(torch.float32),
                                                   spatial_dist_x.type(torch.float32),
                                                   spatial_dist_y.type(torch.float32)), dim=1)

                        else:
                            edge_attr = torch.cat((spatial_dist_g.type(torch.float32),
                                                   spatial_dist_x.type(torch.float32),
                                                   spatial_dist_y.type(torch.float32), emb_dist_g), dim=1)

                        # EDGE LABELS

                        edge_labels_g = torch.from_numpy(
                            np.asarray(
                                [1 if (data_df[g]['id'].values[data_df[g]['node'].values == edge_ixs_g_np[0][i]] ==
                                       data_df[g]['id'].values[data_df[g]['node'].values == edge_ixs_g_np[1][i]]) else 0
                                 for i in range(edge_ixs_g_np.shape[1])])).type(torch.float).cuda()

                        # bajar rango a 0 de edge_iuxs_g
                        edge_ixs_g = edge_ixs_g - torch.min(edge_ixs_g)
                        data = Data(x=node_embeds_g, edge_index=edge_ixs_g, y=node_label_g, edge_attr=edge_attr,
                                    edge_labels=edge_labels_g)

                        batch.append(data)
                        prev_max_counter = max_counter

                if flag_visualize:
                    # EDGE CONNECTIONS
                    edge_ixs = torch.cat(edge_ixs, dim=0).T
                    edge_ixs_np = edge_ixs.numpy()

                    # NODE LABEL NUMBER OF NODE
                    node_label = torch.cat(node_label, dim=0)
                    node_label_np = node_label.numpy()

                    # NODE LABEL NUMBER OF CAMERA
                    node_id_cam = torch.cat(node_id_cam, dim=0)
                    node_id_cam_np = node_id_cam.numpy()

                    # EDGE LABELS
                    edge_labels = torch.tensor(
                        [1 if (node_label_np[edge_ixs_np[0][i]] == node_label_np[edge_ixs_np[1][i]]) else 0 for i in
                         range(edge_ixs_np.shape[1])])
                    edge_labels_np = np.asarray(
                        [1 if (node_label_np[edge_ixs_np[0][i]] == node_label_np[edge_ixs_np[1][i]]) else 0 for i in
                         range(edge_ixs_np.shape[1])])

                    # formatted_edge_labels = {(edge_ixs_np[0][i], edge_ixs_np[1][i]): edge_labels_np[i] for i in range(len(edge_labels))}

                    # EDGE FEATURES : pairwise feature distance
                    emb_dist = F.pairwise_distance(reid_embeds[edge_ixs[0]], reid_embeds[edge_ixs[1]]).view(-1, 1)

                    edge_feats = emb_dist

                    # GRAPH CREATION
                    data = Data(x=node_embeds, edge_index=edge_ixs, y=node_label, edge_feats=edge_feats)
                    G = to_networkx(data, to_undirected=True)
                    utils.visualize(G, color=data.y, edge_labels=edge_labels_np, edge_index=edge_ixs_np,
                                    node_label=node_id_cam_np)

                    # PLOT GROUNDPLANE DETECTIONS
                    frame = val_dataset.frames_valid[1175]

                    # GET each camera view plane

                    plt.figure()
                    plt.title('Detections in frame ' + str(data_df[0]['frame'].values[0]))
                    list_legend = list()

                    # CAM VIEW
                    for id_cam in np.unique(data_df[0]['id_cam']):
                        # plt.fill(train_dataset.list_corners_x[id_cam], train_dataset.list_corners_y[id_cam], c=list_cam_colors[id_cam], alpha=0.2)

                        # TO SHOW THE FRAME WARPED
                        plt.figure()
                        homog_file = os.path.join(dataset_dir, val_dataset.cameras[id_cam], 'Homography.txt')
                        H = np.asarray(pd.read_csv(homog_file, header=None, sep="\t"))
                        frame_path = os.path.join(dataset_dir, val_dataset.cameras[id_cam], 'img1',
                                                  str(frame).zfill(6) + '.jpg')
                        img = imread(frame_path)
                        dst = cv2.warpPerspective(img, H, (500, 500))
                        plt.imshow(dst)
                        plt.fill(val_dataset.list_corners_x[id_cam], val_dataset.list_corners_y[id_cam],
                                 c=list_cam_colors[id_cam], alpha=0.2)
                        data_cam = data_df[0][data_df[0]['id_cam'] == id_cam]
                        xw_in_cam = np.asarray(data_cam['xw'])
                        yw_in_cam = np.asarray(data_cam['yw'])
                        plt.scatter(xw_in_cam, yw_in_cam, c=list_cam_colors[id_cam])
                        plt.show()
                        a = 1

                    # DETECTIONS POINT
                    for id_cam in np.unique(data_df[0]['id_cam']):
                        data_cam = data_df[0][data_df[0]['id_cam'] == id_cam]
                        xw_in_cam = np.asarray(data_cam['xw'])
                        yw_in_cam = np.asarray(data_cam['yw'])
                        plt.scatter(yw_in_cam, xw_in_cam, c=list_cam_colors[id_cam])
                        list_legend.append('camera ' + str(id_cam))

                    plt.legend(list_legend)

                    # TEXT WITH ID
                    for id_cam in np.unique(data_df[0]['id_cam']):
                        data_cam = data_df[0][data_df[0]['id_cam'] == id_cam]
                        xw_in_cam = np.asarray(data_cam['xw'])
                        yw_in_cam = np.asarray(data_cam['yw'])
                        for i in range(len(xw_in_cam)):
                            plt.text(yw_in_cam[i] + 1, xw_in_cam[i] + 1, str(data_cam['id'].values[i]))

                    plt.show()
                    a = 1
                # TRAINING #

                data_batch = Batch.from_data_list(batch)

                ########### Forward ###########

                outputs = mpn_model(data_batch)

                ########### Loss ###########

                # loss, acc_actives, acc_nonactives = compute_loss_acc(outputs, data_batch, mode = 'validate')
                loss, precision1, precision0, precision = compute_loss_acc(outputs, data_batch, mode='validate')

                val_losses.update(loss.item(), CONFIG['TRAINING']['BATCH_SIZE']['VAL'])
                val_precision_1.update(np.sum(np.asarray([item for item in precision1])) / len(precision1),
                                       CONFIG['TRAINING']['BATCH_SIZE']['VAL'])
                val_precision_0.update(np.sum(np.asarray([item for item in precision0])) / len(precision0),
                                       CONFIG['TRAINING']['BATCH_SIZE']['VAL'])

                val_precision.update(np.sum(np.asarray([item for item in precision])) / len(precision),
                                     CONFIG['TRAINING']['BATCH_SIZE']['VAL'])

                # accuracies.append()
                val_loss_in_history.append(val_losses.avg)
                val_prec1_in_history.append(np.sum(np.asarray([item for item in precision1])) / len(precision1))
                val_prec0_in_history.append(np.sum(np.asarray([item for item in precision0])) / len(precision0))
                val_prec_in_history.append(np.sum(np.asarray([item for item in precision])) / len(precision))

                ########### Accuracy ###########

                ########### Optimizer update ###########


                val_batch_time.update(time.time() - start_time)

                if i % 10 == 0:
                    print('Testing validation batch [{0}/{1}]\t'
                          'Batch Time {batch_time.val:.3f} (avg: {batch_time.avg:.3f})\t'
                          'Val Loss {loss.val:.3f} (avg: {loss.avg:.3f})\t'
                          'Val Precision class 1 {acc.val:.3f} (avg: {acc.avg:.3f})\t'
                          'Val Precision class 0 {acc2.val:.3f} (avg: {acc2.avg:.3f})\t'
                          '{et}<{eta}'.format(i, len(val_loader), batch_time=val_batch_time, loss=val_losses,
                                              acc=val_precision_1, acc2=val_precision_0,
                                              et=str(datetime.timedelta(seconds=int(val_batch_time.sum))),
                                              eta=str(datetime.timedelta(
                                                  seconds=int(val_batch_time.avg * (len(val_loader) - i))))))

    plt.figure()
    plt.plot(val_loss_in_history, label='Loss')

    plt.ylabel('Loss'), plt.xlabel('Iteration')
    plt.legend(loc='best')
    plt.savefig(results_path + '/images/Validation Loss per Iteration.pdf', bbox_inches='tight')
    plt.close()

    plt.figure()
    plt.plot(val_prec1_in_history, 'g', label='Precision class 1')
    plt.plot(val_prec0_in_history, 'r', label='Precision class 0')
    plt.ylabel('Precision'), plt.xlabel('Iteration')
    plt.legend(loc='best')
    plt.savefig(results_path + '/images/Validation Precision per Iteration.pdf', bbox_inches='tight')
    plt.close()

    return val_losses, val_precision_1, val_precision_0,val_loss_in_history,val_prec1_in_history, val_prec0_in_history,val_prec_in_history


def validate_REID(val_loader, cnn_model,CONFIG):
    val_batch_time = utils.AverageMeter('batch_time', ':6.3f')

    cnn_model.eval()
    reid_distances_L2 = []
    reid_distances_cos = []
    edge_label = []

    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            if i >= 0:

                start_time = time.time()

                ########### Data extraction ###########
                [bboxes, data_df] = data

                len_graphs = [len(item) for item in data_df]
                # node_embeds, reid_embeds = cnn_model(torch.cat(bboxes, dim=0).cuda())
                if CONFIG['CNN_MODEL']['arch'] == 'resnet50':
                    node_embeds, reid_embeds = cnn_model(torch.cat(bboxes, dim=0).cuda())
                else:
                    node_embeds = cnn_model(torch.cat(bboxes, dim=0).cuda())
                    reid_embeds= node_embeds
                # features reid distances between each pair of points
                max_counter = 0

                flag_visualize = False
                for g in range(len(len_graphs)):
                    # print('Testing frame ' + str(data_df[g].iloc[0]['frame']))
                    # For visualize, we need a complete structure containing all the graphs info concatenated
                    # data_df[g].loc[:,'node'] = np.asarray(range(max_counter, max_counter + len(data_df[g])))
                    data_df[g] = data_df[g].assign(
                        node=np.asarray(range(max_counter, max_counter + len(data_df[g]))))

                    max_counter = max(data_df[g]['node'].values) + 1
                    edge_ixs_g = []
                    for id_cam in np.unique(data_df[g]['id_cam']):
                        ids_in_cam = data_df[g]['node'][data_df[g]['id_cam'] == id_cam].values
                        ids_out_cam = data_df[g]['node'][data_df[g]['id_cam'] != id_cam].values
                        edge_ixs_g.append(
                            torch.cartesian_prod(torch.from_numpy(ids_in_cam), torch.from_numpy(ids_out_cam)))

                    # Compute edges attributes
                    edge_ixs_g = torch.cat(edge_ixs_g, dim=0).T.cuda()
                    edge_ixs_g_np = edge_ixs_g.cpu().numpy()


                    # features reid distances between each pair of points
                    reid_distances_L2.append(F.pairwise_distance(reid_embeds[edge_ixs_g[0]], reid_embeds[edge_ixs_g[1]]).cpu().numpy())
                    reid_distances_cos.append(cos(reid_embeds[edge_ixs_g[0]], reid_embeds[edge_ixs_g[1]]).cpu().numpy())
                    edge_label.append(np.asarray(
                        [1 if (data_df[g]['id'].values[data_df[g]['node'].values == edge_ixs_g_np[0][i]] ==
                               data_df[g]['id'].values[data_df[g]['node'].values == edge_ixs_g_np[1][i]]) else 0
                         for i in range(edge_ixs_g_np.shape[1])]))




                # COMPUTE PRECISION #



                val_batch_time.update(time.time() - start_time)

                if i % 10 == 0:
                    print('Testing validation batch [{0}/{1}]\t'
                          '{et}<{eta}'.format(i, len(val_loader),
                                              et=str(datetime.timedelta(seconds=int(val_batch_time.sum))),
                                              eta=str(datetime.timedelta(seconds=int(val_batch_time.avg * (len(val_loader) - i))))))

    reid_distances_L2 = np.concatenate(reid_distances_L2)
    reid_distances_cos = np.concatenate(reid_distances_cos)
    edge_label = np.concatenate(edge_label)
    return reid_distances_L2, edge_label, reid_distances_cos


def validate_GNN_cross_camera_association(CONFIG, val_loader, cnn_model, mpn_model):

    val_batch_time = utils.AverageMeter('batch_time', ':6.3f')

    cnn_model.eval()
    mpn_model.eval()

    P_list = []
    R_list = []
    TP_list = []
    FP_list = []
    FN_list = []
    TN_list = []
    F_list = []

    P_r_list = []
    R_r_list = []
    TP_r_list = []
    FP_r_list = []
    FN_r_list = []
    TN_r_list = []
    F_r_list = []


    rand_index = []
    rand_index_rounding = []


    mutual_index = []
    homogeneity = []
    completeness = []
    v_measure = []

    with torch.no_grad():
        for i, data in enumerate(val_loader):
            if i >= 0 :

                start_time = time.time()

                ########### Data extraction ###########
                [bboxes, data_df] = data

                len_graphs = [len(item) for item in data_df]
                if CONFIG['CNN_MODEL']['arch'] == 'resnet50':
                    node_embeds, reid_embeds = cnn_model(torch.cat(bboxes, dim=0).cuda())
                else:
                    node_embeds = cnn_model(torch.cat(bboxes, dim=0).cuda())
                    reid_embeds= node_embeds
                # node_embeds, reid_embeds = cnn_model(torch.cat(bboxes, dim=0).cuda())

                max_counter = 0
                prev_max_counter = 0
                edge_ixs = []
                node_label = []
                node_id_cam = []

                # create list called batch, each element: a graph
                batch = []


                for g in range(len(len_graphs)):


                    data_df[g] = data_df[g].assign(
                        node=np.asarray(range(max_counter, max_counter + len(data_df[g]))))
                    print('Testing frame ' + str(data_df[g].iloc[0]['frame']))
                    if data_df[g].iloc[0]['frame'] == 734:
                        a=1
                    max_counter = max(data_df[g]['node'].values) + 1
                    node_embeds_g = torch.stack([node_embeds[i] for i in data_df[g]['node']])
                    edge_ixs_g = []
                    for id_cam in np.unique(data_df[g]['id_cam']):
                        ids_in_cam = data_df[g]['node'][data_df[g]['id_cam'] == id_cam].values
                        ids_out_cam = data_df[g]['node'][data_df[g]['id_cam'] != id_cam].values
                        edge_ixs_g.append(
                            torch.cartesian_prod(torch.from_numpy(ids_in_cam), torch.from_numpy(ids_out_cam)))

                    # Compute edges attributes
                    edge_ixs_g = torch.cat(edge_ixs_g, dim=0).T.cuda()
                    edge_ixs_g_np = edge_ixs_g.cpu().numpy()

                    node_label_g = torch.from_numpy(data_df[g]['id'].values)
                    # node_label_g_np = node_label_g.numpy()

                    # features reid distances between each pair of points
                    emb_dist_g = F.pairwise_distance(reid_embeds[edge_ixs_g[0]], reid_embeds[edge_ixs_g[1]]).view(
                        -1, 1)
                    node_dist_g = F.pairwise_distance(node_embeds[edge_ixs_g[0]], node_embeds[edge_ixs_g[1]]).view(
                        -1, 1)

                    emb_dist_g_cos = F.cosine_similarity(reid_embeds[edge_ixs_g[0]],
                                                         reid_embeds[edge_ixs_g[1]]).view(-1, 1)
                    node_dist_g_cos = F.cosine_similarity(node_embeds[edge_ixs_g[0]],
                                                          node_embeds[edge_ixs_g[1]]).view(-1, 1)

                    # coordinates of each pair of points
                    xws_1 = np.expand_dims(  np.asarray([data_df[g]['xw'].values[item - prev_max_counter] for item in edge_ixs_g_np[0]]),
                        axis=1)
                    yws_1 = np.expand_dims(  np.asarray([data_df[g]['yw'].values[item - prev_max_counter] for item in edge_ixs_g_np[0]]),
                        axis=1)
                    xws_2 = np.expand_dims( np.asarray([data_df[g]['xw'].values[item - prev_max_counter] for item in edge_ixs_g_np[1]]),
                        axis=1)
                    yws_2 = np.expand_dims(  np.asarray([data_df[g]['yw'].values[item - prev_max_counter] for item in edge_ixs_g_np[1]]),
                        axis=1)

                    points1 = np.concatenate((xws_1, yws_1), axis=1)
                    points2 = np.concatenate((xws_2, yws_2), axis=1)

                    spatial_dist_g = torch.unsqueeze((torch.from_numpy(paired_distances(points1, points2))),
                                                     dim=1).cuda()
                    spatial_dist_x = torch.abs(torch.from_numpy(xws_1 - xws_2)).cuda()
                    spatial_dist_y = torch.abs(torch.from_numpy(yws_1 - yws_2)).cuda()
                    if CONFIG['TRAINING']['ONLY_APPEARANCE']:
                        edge_attr = torch.cat((emb_dist_g, node_dist_g, emb_dist_g_cos, node_dist_g_cos), dim=1)
                    else:
                        edge_attr = torch.cat((spatial_dist_g.type(torch.float32),
                                               spatial_dist_x.type(torch.float32),
                                               spatial_dist_y.type(torch.float32), emb_dist_g), dim=1)
                    # EDGE LABELS

                    edge_labels_g = torch.from_numpy(
                        np.asarray(
                            [1 if (data_df[g]['id'].values[data_df[g]['node'].values == edge_ixs_g_np[0][i]] ==
                                   data_df[g]['id'].values[data_df[g]['node'].values == edge_ixs_g_np[1][i]]) else 0
                             for i in range(edge_ixs_g_np.shape[1])])).type(torch.float).cuda()

                    # bajar rango a 0 de edge_iuxs_g
                    edge_ixs_g = edge_ixs_g - torch.min(edge_ixs_g)
                    data = Data(x=node_embeds_g, edge_index=edge_ixs_g, y=node_label_g, edge_attr=edge_attr,
                                edge_labels=edge_labels_g)
                    # H = to_networkx(data)
                    # utils.visualize(H, data.y, node_label=node_label_g)

                    batch.append(data)
                    prev_max_counter = max_counter



                data_batch = Batch.from_data_list(batch)

                ########### Forward ###########

                outputs = mpn_model(data_batch)
                labels_edges_GT = data_batch.edge_labels.view(-1).cpu().numpy()

                preds = outputs['classified_edges'][-1]

                sig = torch.nn.Sigmoid()
                preds_prob = sig(preds)
                predictions = (preds_prob >= 0.5) * 1

                if i==753:
                    a=1
                # # COMPUTE GREEDY ROUNDING
                if CONFIG['ROUNDING']:
                   new_predictions = utils.compute_rounding(data_batch, predictions.view(-1), preds_prob)



                # CLUSTERING IDENTITIES MEASURES
                edge_list = data_batch.edge_index.cpu().numpy()

                GT_active_edges = [(edge_list[0][pos], edge_list[1][pos]) for pos, p in enumerate(labels_edges_GT) if p == 1]
                G_GT = nx.DiGraph(GT_active_edges)
                ID_GT, n_clusters_GT = utils.compute_SCC_and_Clusters(G_GT,data_batch.num_nodes)
                print('# Clusters GT : ' + str(n_clusters_GT))
                print(ID_GT)

                predicted_active_edges = [(edge_list[0][pos], edge_list[1][pos]) for pos, p in enumerate(predictions) if p == 1]
                G = nx.DiGraph(predicted_active_edges)
                ID_pred, n_clusters_pred = utils.compute_SCC_and_Clusters(G,data_batch.num_nodes)
                print('# Clusters:  ' + str(n_clusters_pred))
                print(ID_pred)

                # label_ID_to_disjoint = np.where(np.bincount(ID_pred) > 4)
                # if len(label_ID_to_disjoint) > 1:
                #     a = 1

                # # H2 is graph directed with removed single direction edges
                # H = nx.difference(G.to_undirected().to_directed(), G)
                # H2 = nx.difference(G, H.to_undirected())  # get the difference graph
                # G_undirected = nx.to_undirected(H2)
                # edges_bridges = [c for c in nx.bridges(G_undirected)]


                if new_predictions != []:
                    new_predicted_active_edges = [(edge_list[0][pos], edge_list[1][pos]) for pos, p in
                                              enumerate(new_predictions) if p == 1]
                    rounding_G = nx.DiGraph(new_predicted_active_edges)
                    rounding_ID_pred,rounding_n_clusters_pred = utils.compute_SCC_and_Clusters(rounding_G, data_batch.num_nodes)

                    # Count how many items in each class, then look which cluster label is the one with more than 4 elements (# cameras)
                    label_ID_to_disjoint = np.where(np.bincount(rounding_ID_pred) > 4)
                    if len(label_ID_to_disjoint) > 1:
                        a=1
                    global_idx_new_predicted_active_edges = [pos for pos, p in enumerate(new_predictions) if p == 1]

                    #COMPROBAR SI HAY DOS
                    for l in label_ID_to_disjoint:
                        flag_need_disjoint = True
                        while flag_need_disjoint:
                            global_idx_new_predicted_active_edges = [pos for pos, p in enumerate(new_predictions) if p == 1]

                            nodes_to_disjoint = np.where(rounding_ID_pred == l)
                            idx_active_edges_to_disjoint = [pos for pos, n in enumerate(new_predicted_active_edges) if np.any(np.in1d(nodes_to_disjoint, n))]
                            global_idx_edges_disjoint = np.asarray(global_idx_new_predicted_active_edges)[np.asarray(idx_active_edges_to_disjoint)]
                            min_prob = np.min(preds_prob[global_idx_edges_disjoint].cpu().numpy())
                            global_idx_min_prob = np.where(preds_prob.cpu().numpy() == min_prob)[0]
                            new_predictions[global_idx_min_prob] = 0

                            # Check if still need to disjoint

                            new_predicted_active_edges = [(edge_list[0][pos], edge_list[1][pos]) for pos, p in
                                                          enumerate(new_predictions) if p == 1]

                            rounding_G = nx.DiGraph(new_predicted_active_edges)
                            rounding_ID_pred, rounding_n_clusters_pred = utils.compute_SCC_and_Clusters(rounding_G,
                                                                                                        data_batch.num_nodes)


                            if np.bincount(rounding_ID_pred)[l] > 4:
                                flag_need_disjoint = True
                            else:
                                flag_need_disjoint = False


                    print('# Clusters with rounding:  ' + str(rounding_n_clusters_pred))
                    print(rounding_ID_pred)


                else:
                    rounding_ID_pred = ID_pred
                    new_predictions = predictions

                rand_index_rounding.append(metrics.adjusted_rand_score(ID_GT, rounding_ID_pred))
                rand_index.append(metrics.adjusted_rand_score(ID_GT, ID_pred))


                TP, FP, TN, FN, P, R, FS = compute_P_R_F(predictions.cpu().numpy(), labels_edges_GT)

                TP_r, FP_r, TN_r, FN_r, P_r, R_r, FS_r = compute_P_R_F(new_predictions.cpu().numpy(), labels_edges_GT)
                # rand_index.append(metrics.adjusted_rand_score(ID_GT, ID_pred))
                mutual_index.append(metrics.adjusted_mutual_info_score(ID_GT, rounding_ID_pred))
                homogeneity.append(metrics.homogeneity_score(ID_GT, rounding_ID_pred))
                completeness.append(metrics.completeness_score(ID_GT, rounding_ID_pred))
                v_measure.append(metrics.v_measure_score(ID_GT, rounding_ID_pred))



                TP_list.append(TP)
                FP_list.append(FP)
                FN_list.append(FN)
                P_list.append(P)
                R_list.append(R)
                F_list.append(FS)
                TN_list.append(TN)

                TP_r_list.append(TP_r)
                FP_r_list.append(FP_r)
                FN_r_list.append(FN_r)
                P_r_list.append(P_r)
                R_r_list.append(R_r)
                F_r_list.append(FS_r)
                TN_r_list.append(TN_r)

                val_batch_time.update(time.time() - start_time)

                if i % 10 == 0:
                    print('Testing validation batch [{0}/{1}]\t'
                          '{et}<{eta}'.format(i, len(val_loader),
                                              et=str(datetime.timedelta(seconds=int(val_batch_time.sum))),
                                              eta=str(datetime.timedelta(seconds=int(val_batch_time.avg * (len(val_loader) - i))))))


    return P_list, R_list, F_list, TP_list, FP_list, FN_list, TN_list, rand_index, rand_index_rounding, P_r_list, R_r_list, F_r_list, TP_r_list, FP_r_list, FN_r_list, TN_r_list,mutual_index, homogeneity, completeness, v_measure


def eval_RANK(val_loader, model,CONFIG):
    # EVAL RANK X
    val_batch_time = utils.AverageMeter('batch_time', ':6.3f')

    model.eval()
    reid_distances = []
    edge_label = []

    P_list = []
    R_list = []
    TP_list = []
    FP_list = []
    FN_list = []
    TN_list = []
    F_list = []
    rand_index = []
    with torch.no_grad():
        for it, data in enumerate(val_loader):
            if it >= 0:
                start_time = time.time()

                ########### Data extraction ###########
                [bboxes, data_df] = data

                len_graphs = [len(item) for item in data_df]
                feat = model(torch.cat(bboxes, dim=0).cuda())

                # feat = F.normalize(feat, p=2, dim=1)

                # features reid distances between all detections in the frame
                # dist_mat2 = F.pairwise_distance(feat, feat).view(-1, 1)
                # dist_mat = utils.compute_distance_matrix(feat, feat, 'euclidean')
                dist_mat = torch.cdist(feat, feat, p=2)
                dist_mat = dist_mat.cpu().numpy()

                rerank = False
                if rerank:
                    distmat_qq = torch.cdist(feat, feat)
                    distmat_gg = torch.cdist(feat, feat)
                    dist_mat = utils.re_ranking(dist_mat, distmat_qq.cpu().numpy(), distmat_gg.cpu().numpy())


                max_counter = 0
                # indices = np.argsort(distmat, axis=1)

                apply_cam_restrictions = True
                for g in range(len(len_graphs)):
                    # print('Testing frame ' + str(data_df[g].iloc[0]['frame']))
                    # For visualize, we need a complete structure containing all the graphs info concatenated
                    # data_df[g].loc[:,'node'] = np.asarray(range(max_counter, max_counter + len(data_df[g])))
                    data_df[g] = data_df[g].assign(
                        node=np.asarray(range(max_counter, max_counter + len(data_df[g]))))

                    max_counter = max(data_df[g]['node'].values) + 1
                    edge_ixs_g = []
                    for id_cam in np.unique(data_df[g]['id_cam']):
                        ids_in_cam = data_df[g]['node'][data_df[g]['id_cam'] == id_cam].values
                        ids_out_cam = data_df[g]['node'][data_df[g]['id_cam'] != id_cam].values
                        edge_ixs_g.append(
                            torch.cartesian_prod(torch.from_numpy(ids_in_cam), torch.from_numpy(ids_out_cam)))

                    # Compute edges attributes
                    edge_ixs_g = torch.cat(edge_ixs_g, dim=0).T.cuda()
                    edge_ixs_g_np = edge_ixs_g.cpu().numpy()


                    # features reid distances between each pair of points
                    # reid_distances.append(F.pairwise_distance(reid_embeds[edge_ixs_g[0]], reid_embeds[edge_ixs_g[1]]).cpu().numpy())

                    edge_label = (np.asarray(
                        [1 if (data_df[g]['id'].values[data_df[g]['node'].values == edge_ixs_g_np[0][i]] ==
                               data_df[g]['id'].values[data_df[g]['node'].values == edge_ixs_g_np[1][i]]) else 0
                         for i in range(edge_ixs_g_np.shape[1])]))
                # COMPUTE PRECISION #


                if apply_cam_restrictions:
                    new_dist_mat = np.ones(dist_mat.shape) * 100
                    list_edges = [(edge_ixs_g_np[0, i], edge_ixs_g_np[1, i]) for i in range(edge_ixs_g_np.shape[1])]
                    for i in list_edges:
                        new_dist_mat[i] = dist_mat[i]

                indices = np.argsort(new_dist_mat, axis=1)
                # Always last colum of indices is just the same node, so we remove the last colum
                indices = indices[:, 0:-1]

                pred_active_edges_in = []
                pred_active_edges_out = []
                for r in range(0,CONFIG['RANK']):
                    if r <= indices.shape[1]-1:
                        temp_in = [(i, indices[i, r]) for i in range(indices.shape[0])]

                        temp_out = [(i[::-1]) for i in temp_in]
                        for i in temp_in:
                            pred_active_edges_in.append(i)
                        for i in temp_out:
                            pred_active_edges_out.append(i)



                pred_active_edges = pred_active_edges_in + pred_active_edges_out


                predictions = np.asarray([1 if i in pred_active_edges else 0 for i in list_edges])
                #HACER CALULO DE LAS MEDIDAS E IR APPEND Y DEVOLVER LAS LISTAS
                val_batch_time.update(time.time() - start_time)


                TP, FP, TN, FN, P, R, FS = compute_P_R_F(predictions, edge_label)
                GT_active_edges = [(edge_ixs_g_np[0][pos], edge_ixs_g_np[1][pos]) for pos, p in enumerate(edge_label)
                                   if p == 1]
                G_GT = nx.DiGraph(GT_active_edges)
                ID_GT, n_clusters_GT = utils.compute_SCC_and_Clusters(G_GT, data_df[0].node.shape[0])
                print('# Clusters GT : ' + str(n_clusters_GT))
                print(ID_GT)

                # predicted_active_edges = [(edge_ixs_g_np[0][pos], edge_ixs_g_np[1][pos]) for pos, p in enumerate(predictions) if
                #                           p == 1]
                G = nx.DiGraph(pred_active_edges)
                ID_pred, n_clusters_pred = utils.compute_SCC_and_Clusters(G, data_df[0].node.shape[0])
                print('# Clusters:  ' + str(n_clusters_pred))
                print(ID_pred)

                rand_index.append(metrics.adjusted_rand_score(ID_GT, ID_pred))
                TP_list.append(TP)
                FP_list.append(FP)
                FN_list.append(FN)
                P_list.append(P)
                R_list.append(R)
                F_list.append(FS)
                TN_list.append(TN)



                if it % 10 == 0:
                    print('Testing validation batch [{0}/{1}]\t'
                          '{et}<{eta}'.format(it, len(val_loader),
                                              et=str(datetime.timedelta(seconds=int(val_batch_time.sum))),
                                              eta=str(datetime.timedelta(
                                                  seconds=int(val_batch_time.avg * (len(val_loader) - it))))))


    return P_list, R_list, F_list, TP_list, FP_list, FN_list, TN_list, rand_index



def validate_REID_with_th(val_loader, cnn_model, th,max_v, dist):

    # RUN THIS ONCE YOU ALREADY HAVE THE OPTIMAL TH
    #  EVALS FRAME BY FRAME
    val_batch_time = utils.AverageMeter('batch_time', ':6.3f')

    cnn_model.eval()
    reid_distances = []
    edge_label = []
    P_list = []
    R_list = []
    TP_list = []
    FP_list = []
    FN_list = []
    TN_list = []
    F_list = []
    rand_index = []
    mutual_index = []
    homogeneity = []
    completeness = []
    v_measure = []

    with torch.no_grad():
        for i, data in enumerate(val_loader):
            if i >= 0:

                start_time = time.time()

                ########### Data extraction ###########
                [bboxes, data_df] = data

                len_graphs = [len(item) for item in data_df]
                node_embeds, reid_embeds = cnn_model(torch.cat(bboxes, dim=0).cuda())

                # features reid distances between each pair of points
                max_counter = 0

                flag_visualize = False
                for g in range(len(len_graphs)):
                    # print('Testing frame ' + str(data_df[g].iloc[0]['frame']))

                    data_df[g] = data_df[g].assign(
                        node=np.asarray(range(max_counter, max_counter + len(data_df[g]))))

                    max_counter = max(data_df[g]['node'].values) + 1
                    edge_ixs_g = []
                    for id_cam in np.unique(data_df[g]['id_cam']):
                        ids_in_cam = data_df[g]['node'][data_df[g]['id_cam'] == id_cam].values
                        ids_out_cam = data_df[g]['node'][data_df[g]['id_cam'] != id_cam].values
                        edge_ixs_g.append(
                            torch.cartesian_prod(torch.from_numpy(ids_in_cam), torch.from_numpy(ids_out_cam)))

                    # Compute edges attributes
                    edge_ixs_g = torch.cat(edge_ixs_g, dim=0).T.cuda()
                    edge_ixs_g_np = edge_ixs_g.cpu().numpy()


                    # features reid distances between each pair of points
                    if dist == 'L2':
                        reid_distances = (
                            F.pairwise_distance(reid_embeds[edge_ixs_g[0]], reid_embeds[edge_ixs_g[1]]).cpu().numpy())
                        reid_distances_norm = reid_distances / max_v
                        predictions = (reid_distances_norm <= th) * 1

                    elif dist == 'cosine':
                        reid_distances = (
                            F.cosine_similarity(reid_embeds[edge_ixs_g[0]], reid_embeds[edge_ixs_g[1]]).cpu().numpy())
                        reid_distances_norm = np.abs(reid_distances)
                        predictions = (reid_distances_norm >= th) * 1

                    edge_label = (np.asarray(
                        [1 if (data_df[g]['id'].values[data_df[g]['node'].values == edge_ixs_g_np[0][i]] ==
                               data_df[g]['id'].values[data_df[g]['node'].values == edge_ixs_g_np[1][i]]) else 0
                         for i in range(edge_ixs_g_np.shape[1])]))



                    TP, FP, TN, FN, P, R, FS = compute_P_R_F(predictions, edge_label)
                    GT_active_edges = [(edge_ixs_g_np[0][pos], edge_ixs_g_np[1][pos]) for pos, p in
                                       enumerate(edge_label)
                                       if p == 1]
                    G_GT = nx.DiGraph(GT_active_edges)
                    ID_GT, n_clusters_GT = utils.compute_SCC_and_Clusters(G_GT, data_df[0].node.shape[0])
                    # print('# Clusters GT : ' + str(n_clusters_GT))
                    # print(ID_GT)
                    predicted_active_edges = [(edge_ixs_g_np[0][pos], edge_ixs_g_np[1][pos]) for pos, p in enumerate(predictions) if          p == 1]
                    G = nx.DiGraph(predicted_active_edges)
                    ID_pred, n_clusters_pred = utils.compute_SCC_and_Clusters(G, data_df[0].node.shape[0])
                    # print('# Clusters:  ' + str(n_clusters_pred))
                    # print(ID_pred)

                    rand_index.append(metrics.adjusted_rand_score(ID_GT, ID_pred))
                    mutual_index.append(metrics.adjusted_mutual_info_score(ID_GT, ID_pred))
                    homogeneity.append(metrics.homogeneity_score(ID_GT, ID_pred))
                    completeness.append(metrics.completeness_score(ID_GT, ID_pred))
                    v_measure.append(metrics.v_measure_score(ID_GT, ID_pred))
                    TP_list.append(TP)
                    FP_list.append(FP)
                    FN_list.append(FN)
                    P_list.append(P)
                    R_list.append(R)
                    F_list.append(FS)
                    TN_list.append(TN)

                # COMPUTE PRECISION #



                val_batch_time.update(time.time() - start_time)

                if i % 10 == 0:
                    print('Testing validation batch [{0}/{1}]\t'
                          '{et}<{eta}'.format(i, len(val_loader),
                                              et=str(datetime.timedelta(seconds=int(val_batch_time.sum))),
                                              eta=str(datetime.timedelta(seconds=int(val_batch_time.avg * (len(val_loader) - i))))))


    return P_list, R_list, F_list, TP_list, FP_list, FN_list, TN_list, rand_index, mutual_index, homogeneity, completeness, v_measure



def geometrical_association(CONFIG, val_loader):

    val_batch_time = utils.AverageMeter('batch_time', ':6.3f')



    P_list = []
    R_list = []
    TP_list = []
    FP_list = []
    FN_list = []
    TN_list = []
    F_list = []

    P_r_list = []
    R_r_list = []
    TP_r_list = []
    FP_r_list = []
    FN_r_list = []
    TN_r_list = []
    F_r_list = []


    rand_index = []
    rand_index_rounding = []
    fowlkes_index = []
    silhouette_index = []


    mutual_index = []
    homogeneity = []
    completeness = []
    v_measure = []

    for i, data in enumerate(val_loader):
        if i >= 0 :

            start_time = time.time()

            ########### Data extraction ###########
            [bboxes, data_df] = data

            len_graphs = [len(item) for item in data_df]

            max_counter = 0
            prev_max_counter = 0
            edge_ixs = []
            node_label = []
            node_id_cam = []

            # create list called batch, each element: a graph
            batch = []

            for g in range(len(len_graphs)):


                data_df[g] = data_df[g].assign(node=np.asarray(range(max_counter, max_counter + len(data_df[g]))))
                print('Testing frame ' + str(data_df[g].iloc[0]['frame']))

                max_counter = max(data_df[g]['node'].values) + 1
                edge_ixs_g = []

                for id_cam in np.unique(data_df[g]['id_cam']):
                    ids_in_cam = data_df[g]['node'][data_df[g]['id_cam'] == id_cam].values
                    ids_out_cam = data_df[g]['node'][data_df[g]['id_cam'] != id_cam].values
                    edge_ixs_g.append(
                        torch.cartesian_prod(torch.from_numpy(ids_in_cam), torch.from_numpy(ids_out_cam)))

                # Compute edges attributes
                edge_ixs_g = torch.cat(edge_ixs_g, dim=0).T.cuda()
                edge_ixs_g_np = edge_ixs_g.cpu().numpy()

                node_label_g = torch.from_numpy(data_df[g]['id'].values)
                # node_label_g_np = node_label_g.numpy()


                # coordinates of each pair of points
                xws_1 = np.expand_dims(  np.asarray([data_df[g]['xw'].values[item - prev_max_counter] for item in edge_ixs_g_np[0]]),
                    axis=1)
                yws_1 = np.expand_dims(  np.asarray([data_df[g]['yw'].values[item - prev_max_counter] for item in edge_ixs_g_np[0]]),
                    axis=1)
                xws_2 = np.expand_dims( np.asarray([data_df[g]['xw'].values[item - prev_max_counter] for item in edge_ixs_g_np[1]]),
                    axis=1)
                yws_2 = np.expand_dims(  np.asarray([data_df[g]['yw'].values[item - prev_max_counter] for item in edge_ixs_g_np[1]]),
                    axis=1)

                points1 = np.concatenate((xws_1, yws_1), axis=1)
                points2 = np.concatenate((xws_2, yws_2), axis=1)

                spatial_dist_g = torch.unsqueeze((torch.from_numpy(paired_distances(points1, points2))),
                                                 dim=1).cuda()
                # EDGE LABELS

                edge_labels_g = torch.from_numpy(
                    np.asarray(
                        [1 if (data_df[g]['id'].values[data_df[g]['node'].values == edge_ixs_g_np[0][i]] ==
                               data_df[g]['id'].values[data_df[g]['node'].values == edge_ixs_g_np[1][i]]) else 0
                         for i in range(edge_ixs_g_np.shape[1])])).type(torch.float).cuda()

                # bajar rango a 0 de edge_iuxs_g
                edge_ixs_g = edge_ixs_g - torch.min(edge_ixs_g)

                data = Data(edge_index=edge_ixs_g, y=node_label_g, edge_attr=spatial_dist_g,
                            edge_labels=edge_labels_g)
                # H = to_networkx(data)
                # utils.visualize(H, data.y, node_label=node_label_g)

                batch.append(data)
                prev_max_counter = max_counter
                # visualize(H, data.y, node_label=node_label_g)


            data_batch = Batch.from_data_list(batch)
            #
            # ########### Forward ###########
            #
            # outputs = mpn_model(data_batch)
            labels_edges_GT = edge_labels_g


            preds = (spatial_dist_g < 30) * 1

            if i == 915:
                a=1
            # # COMPUTE GREEDY ROUNDING
            if CONFIG['ROUNDING']:
               new_predictions = utils.compute_rounding(data_batch, (preds).view(-1), 1-spatial_dist_g)


            # CLUSTERING IDENTITIES MEASURES
            edge_list = data_batch.edge_index.cpu().numpy()

            GT_active_edges = [(edge_list[0][pos], edge_list[1][pos]) for pos, p in enumerate(labels_edges_GT) if p == 1]
            G_GT = nx.DiGraph(GT_active_edges)
            ID_GT, n_clusters_GT = utils.compute_SCC_and_Clusters(G_GT,data_batch.num_nodes)
            print('# Clusters GT : ' + str(n_clusters_GT))
            print(ID_GT)

            predicted_active_edges = [(edge_list[0][pos], edge_list[1][pos]) for pos, p in enumerate(preds) if p == 1]
            G = nx.DiGraph(predicted_active_edges)
            ID_pred, n_clusters_pred = utils.compute_SCC_and_Clusters(G,data_batch.num_nodes)
            print('# Clusters:  ' + str(n_clusters_pred))
            print(ID_pred)

            if new_predictions != []:
                new_predicted_active_edges = [(edge_list[0][pos], edge_list[1][pos]) for pos, p in
                                          enumerate(new_predictions) if p == 1]
                rounding_G = nx.DiGraph(new_predicted_active_edges)
                rounding_ID_pred,rounding_n_clusters_pred = utils.compute_SCC_and_Clusters(rounding_G, data_batch.num_nodes)

                a=1

                print('# Clusters with rounding:  ' + str(rounding_n_clusters_pred))
                print(rounding_ID_pred)


            else:
                rounding_ID_pred = ID_pred
                new_predictions = preds

            rand_index_rounding.append(metrics.adjusted_rand_score(ID_GT, rounding_ID_pred))
            rand_index.append(metrics.adjusted_rand_score(ID_GT, ID_pred))

            fowlkes_index.append(metrics.fowlkes_mallows_score(ID_GT,rounding_ID_pred))
            # silhouette_index.append(metrics.silhouette_score(ID_GT,rounding_ID_pred))

            TP, FP, TN, FN, P, R, FS = compute_P_R_F(preds.cpu().numpy(), labels_edges_GT.cpu().numpy())

            TP_r, FP_r, TN_r, FN_r, P_r, R_r, FS_r = compute_P_R_F(new_predictions.cpu().numpy(),  labels_edges_GT.cpu().numpy())
            # rand_index.append(metrics.adjusted_rand_score(ID_GT, ID_pred))
            mutual_index.append(metrics.adjusted_mutual_info_score(ID_GT, rounding_ID_pred))
            homogeneity.append(metrics.homogeneity_score(ID_GT, rounding_ID_pred))
            completeness.append(metrics.completeness_score(ID_GT, rounding_ID_pred))
            v_measure.append(metrics.v_measure_score(ID_GT, rounding_ID_pred))


            TP_list.append(TP)
            FP_list.append(FP)
            FN_list.append(FN)
            P_list.append(P)
            R_list.append(R)
            F_list.append(FS)
            TN_list.append(TN)

            TP_r_list.append(TP_r)
            FP_r_list.append(FP_r)
            FN_r_list.append(FN_r)
            P_r_list.append(P_r)
            R_r_list.append(R_r)
            F_r_list.append(FS_r)
            TN_r_list.append(TN_r)

            val_batch_time.update(time.time() - start_time)

            if i % 10 == 0:
                print('Testing validation batch [{0}/{1}]\t'
                      '{et}<{eta}'.format(i, len(val_loader),
                                          et=str(datetime.timedelta(seconds=int(val_batch_time.sum))),
                                          eta=str(datetime.timedelta(seconds=int(val_batch_time.avg * (len(val_loader) - i))))))


    return P_list, R_list, F_list, TP_list, FP_list, FN_list, TN_list, rand_index, rand_index_rounding, P_r_list, R_r_list, \
           F_r_list, TP_r_list, FP_r_list, FN_r_list, TN_r_list,mutual_index, homogeneity, completeness, v_measure, \
           fowlkes_index

