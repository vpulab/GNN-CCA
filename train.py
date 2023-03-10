
import os
import time
import shutil
import yaml
import datetime

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
# import matplotlib.pyplot as plt



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

from libs import datasets
from models.resnet import resnet50_fc256, load_pretrained_weights
from models.mpn import MOTMPNet

from torch_geometric.utils import to_networkx
import networkx as nx
from skimage.io import imread

from libs import utils
from sklearn.metrics.pairwise import paired_distances
from scipy.sparse.csgraph import connected_components
from sklearn import metrics


list_cam_colors = list(['royalblue', 'darkorange','green','firebrick'])


def compute_loss_acc(outputs, batch, criterion, criterion_no_reduction,  mode):
    # global num_edges, num_edges1
    # num_edges1 += np.int(positive_vals.cpu())
    # num_edges += np.int(labels.shape[0])

    # Define Balancing weight
    labels = batch.edge_labels.view(-1)

    # Compute Weighted BCE:
    loss = 0
    loss_class1 = 0
    loss_class0 = 0
    precision_class1 = list()
    precision_class0 = list()
    precision_all = list()

    list_pred_prob = list()
    num_steps = len(outputs['classified_edges'])

   # Compute loss of all the steps and sum them

    ## FOR CONSIDERING ONLY LAST 3 STEPS or less
    # step_ini = max(0,num_steps-3)
    # step_end = num_steps

    # comment FOR CONSIDERING ALL STEPS
    step_ini= 0
    step_end = num_steps

    for step in range(step_ini, step_end):
        preds = outputs['classified_edges'][step].view(-1)

        if mode == 'train':

            loss_per_sample = criterion_no_reduction(preds, labels)
            loss += criterion(preds, labels) # before +=

            loss_class1 += torch.mean(loss_per_sample[labels == 1])
            loss_class0 += torch.mean(loss_per_sample[labels == 0])


        else:
            loss_per_sample = F.binary_cross_entropy_with_logits(preds, labels, reduction='none')
            loss_class1 += torch.mean(loss_per_sample[labels == 1])
            loss_class0 += torch.mean(loss_per_sample[labels == 0])

            loss += F.binary_cross_entropy_with_logits(preds, labels, reduction='mean')


        with torch.no_grad():
            sig = torch.nn.Sigmoid()
            preds_prob = sig(preds)
            list_pred_prob.append(preds_prob)


    # Precision is computed only with last step predictions
    with torch.no_grad():
        preds = outputs['classified_edges'][-1].view(-1)
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
   #  end


   # Compute loss and precision only of the last step




    # preds = outputs['classified_edges'][-1].view(-1)
    # #
    # if mode == 'train':
    #
    #     loss_per_sample = criterion_no_reduction(preds, labels)
    #     loss = criterion(preds, labels)
    #
    #     loss_class1 = torch.mean(loss_per_sample[labels == 1])
    #     loss_class0 = torch.mean(loss_per_sample[labels == 0])
    #
    #
    # else:
    #     loss_per_sample = F.binary_cross_entropy_with_logits(preds, labels, reduction='none')
    #     loss_class1 = torch.mean(loss_per_sample[labels == 1])
    #     loss_class0 = torch.mean(loss_per_sample[labels == 0])
    #
    #     loss = F.binary_cross_entropy_with_logits(preds, labels, reduction='mean')
    #
    #
    # with torch.no_grad():
    #     sig = torch.nn.Sigmoid()
    #     preds_prob = sig(preds)
    #     predictions = (preds_prob >= 0.5) * 1
    #     # Precision class 1
    #     index_label_1 = np.where(np.asarray(labels.cpu()) == 1)
    #     sum_successes_1 = np.sum(predictions.cpu().numpy()[index_label_1] == labels.cpu().numpy()[index_label_1])
    #     if sum_successes_1 == 0:
    #         precision_class1.append(0)
    #         # precision_class1 = 0
    #     else:
    #         precision_class1.append((sum_successes_1 / len(labels[index_label_1])) * 100.0)
    #         # precision_class1 = (sum_successes_1 / len(labels[index_label_1])) * 100.0
    #
    #
    #     # Precision class 0
    #     index_label_0 = np.where(np.asarray(labels.cpu()) == 0)
    #     sum_successes_0 = np.sum(predictions.cpu().numpy()[index_label_0] == labels.cpu().numpy()[index_label_0])
    #
    #     if sum_successes_0 == 0:
    #         precision_class0.append(0)
    #         # precision_class0 = (0)
    #
    #     else:
    #         precision_class0.append((sum_successes_0 / len(labels[index_label_0])) * 100.0)
    #         # precision_class0 = (sum_successes_0 / len(labels[index_label_0])) * 100.0
    #
    #
    #     # Precision
    #     sum_successes = np.sum(predictions.cpu().numpy() == labels.cpu().numpy())
    #     if sum_successes == 0:
    #         precision_all.append(0.5)
    #         # precision_all = 0
    #
    #     else:
    #         precision_all.append((sum_successes / len(labels)) * 100.0)
    #         # precision_all = (sum_successes / len(labels)) * 100.0
    #
    #     for step in range(num_steps):
    #         preds = outputs['classified_edges'][step].view(-1)
    #         preds_prob = sig(preds)
    #
    #         list_pred_prob.append(preds_prob)

    #     a=1
            ## end

    return loss, precision_class1, precision_class0, precision_all, loss_class1, loss_class0, list_pred_prob


def train(CONFIG, train_loader, cnn_model, mpn_model, epoch, optimizer,results_path,train_loss_in_history,
          train_prec1_in_history,train_prec0_in_history, train_prec_in_history, train_dataset, dataset_dir , criterion, criterion_no_reduction,list_mean_probs_history):

    train_losses = utils.AverageMeter('losses', ':.4e')
    train_losses1 = utils.AverageMeter('losses', ':.4e')
    train_losses0 = utils.AverageMeter('losses', ':.4e')

    train_batch_time = utils.AverageMeter('batch_time', ':6.3f')
    train_precision_class1 = utils.AverageMeter('Precision_class1', ':6.2f')
    train_precision_class0 = utils.AverageMeter('Precision_class0', ':6.2f')
    train_precision = utils.AverageMeter('Precision', ':6.2f')
    mpn_model.train()
    n_steps = CONFIG['GRAPH_NET_PARAMS']['num_class_steps']
    list_mean_probs = {"0": {}, "1": {}}
    if n_steps >0:
        for i in range(n_steps):
            list_mean_probs["0"]["step" + str(i)] = []
            list_mean_probs["1"]["step" + str(i)] = []

    else:
        n_steps = 1
        for i in range(n_steps):
            list_mean_probs["0"]["step" + str(i)] = []
            list_mean_probs["1"]["step" + str(i)] = []


    for i, data in enumerate(train_loader):

        if i >= 0 :

            start_time = time.time()

            ########### Data extraction ###########
            [bboxes, data_df,max_dist] = data
            # max_dist = [1] * len(max_dist)
            len_graphs = [len(item) for item in data_df]

            with torch.no_grad():
                if CONFIG['CNN_MODEL']['arch'] == 'resnet50':
                    node_embeds, reid_embeds = cnn_model(torch.cat(bboxes, dim=0).cuda())
                else:
                    node_embeds = cnn_model(torch.cat(bboxes, dim=0).cuda())
                    reid_embeds= node_embeds

            #reid embeds needs to be normalize before computing distances

            if CONFIG['CNN_MODEL']['L2norm']:
                reid_embeds = F.normalize(reid_embeds, p= 2,dim=0)
                node_embeds =  F.normalize(node_embeds, p= 2,dim=0)

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
                    # node_dist_g = F.pairwise_distance(node_embeds[edge_ixs_g[0]], node_embeds[edge_ixs_g[1]]).view(-1, 1)
                    emb_dist_g_cos = F.cosine_similarity(reid_embeds[edge_ixs_g[0]], reid_embeds[edge_ixs_g[1]]).view(-1, 1)
                    # node_dist_g_cos = F.cosine_similarity(node_embeds[edge_ixs_g[0]], node_embeds[edge_ixs_g[1]]).view(-1, 1)

                    # coordinates of each pair of points

                    xws_1 = np.expand_dims(np.asarray([data_df[g]['xw'].values[item-prev_max_counter] for item in edge_ixs_g_np[0]]), axis = 1)
                    yws_1 = np.expand_dims(np.asarray([data_df[g]['yw'].values[item-prev_max_counter]for item in edge_ixs_g_np[0]]), axis = 1)
                    xws_2 = np.expand_dims(np.asarray([data_df[g]['xw'].values[item-prev_max_counter] for item in edge_ixs_g_np[1]]), axis = 1)
                    yws_2 = np.expand_dims(np.asarray([data_df[g]['yw'].values[item-prev_max_counter] for item in edge_ixs_g_np[1]]), axis = 1)

                    points1 = np.concatenate((xws_1, yws_1), axis=1)
                    points2 = np.concatenate((xws_2, yws_2), axis=1)

                    #Convert distances to meters
                    spatial_dist_g = torch.unsqueeze((torch.from_numpy(paired_distances(points1, points2))), dim=1).cuda()
                    spatial_dist_g_norm = torch.from_numpy(spatial_dist_g.cpu().numpy() / max_dist[g]).cuda()

                    spatial_dist_manh_g = torch.unsqueeze((torch.from_numpy(paired_distances(points1, points2,metric='manhattan'))),       dim=1).cuda()
                    spatial_dist_manh_g_norm = torch.from_numpy(spatial_dist_manh_g.cpu().numpy() / max_dist[g]).cuda()

                    # spatial_dist_x = torch.abs(torch.from_numpy(xws_1 - xws_2)).cuda()
                    # spatial_dist_x_norm = (spatial_dist_x / max_dist[g])
                    # spatial_dist_y = torch.abs(torch.from_numpy(yws_1 - yws_2)).cuda()
                    # spatial_dist_y_norm = (spatial_dist_y / max_dist[g]).cuda()
                    # spatial_dist_g_l.append(spatial_dist_g.cpu().numpy())
                    # spatial_dist_g_l_norm.append(spatial_dist_g_norm.cpu().numpy())

                    if CONFIG['TRAINING']['ONLY_APPEARANCE']:
                        # edge_attr = torch.cat((emb_dist_g, node_dist_g, emb_dist_g_cos, node_dist_g_cos),   dim=1)
                        edge_attr = torch.cat((emb_dist_g, emb_dist_g_cos),   dim=1)

                    elif CONFIG['TRAINING']['ONLY_DIST']:
                        edge_attr = torch.cat((spatial_dist_g_norm.type(torch.float32),spatial_dist_manh_g_norm.type(torch.float32) ), dim=1)

                    else:
                        # edge_attr = torch.cat((spatial_dist_g_norm.type(torch.float32), spatial_dist_x_norm.type(torch.float32), spatial_dist_y_norm.type(torch.float32), emb_dist_g), dim=1)
                        edge_attr = torch.cat((spatial_dist_g_norm.type(torch.float32), spatial_dist_manh_g_norm.type(torch.float32), emb_dist_g, emb_dist_g_cos), dim=1)

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
                frame = data_df[0]['frame'].values[0]

                # GET each camera view plane


                # # CAM VIEW
                # for id_cam in np.unique(data_df[0]['id_cam']):
                #     # plt.fill(train_dataset.list_corners_x[id_cam], train_dataset.list_corners_y[id_cam], c=list_cam_colors[id_cam], alpha=0.2)
                #
                #     # TO SHOW THE FRAME WARPED
                #     plt.figure()
                #     homog_file = os.path.join(dataset_dir, train_dataset.cameras[id_cam], 'Homography.txt')
                #     H = np.asarray(pd.read_csv(homog_file, header=None, sep="\t"))
                #     frame_path = os.path.join(dataset_dir, train_dataset.cameras[id_cam], 'img1',
                #                               str(frame).zfill(6) + '.jpg')
                #     img = imread(frame_path)
                #     dst = cv2.warpPerspective(img, H, (500, 500))
                #     plt.imshow(dst)
                #     plt.fill(train_dataset.list_corners_x[id_cam], train_dataset.list_corners_y[id_cam],
                #              c=list_cam_colors[id_cam], alpha=0.2)
                #     data_cam = data_df[0][data_df[0]['id_cam'] == id_cam]
                #     xw_in_cam = np.asarray(data_cam['xw'])
                #     yw_in_cam = np.asarray(data_cam['yw'])
                #     plt.scatter(xw_in_cam, yw_in_cam, c=list_cam_colors[id_cam])
                #     plt.show(block=False)
                #     a = 1
                #
                plt.figure()
                plt.title('Detections in frame ' + str(data_df[0]['frame'].values[0]))
                list_legend = list()

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

            loss, precision1, precision0, precision,loss_class1, loss_class0, list_pred_probs = compute_loss_acc(outputs, data_batch, criterion, criterion_no_reduction, mode='train')
            #Fill dictionary with mean probabilities of each class at each step
            nsteps = len(list_pred_probs)
            for s in range(nsteps):
                if sum(sum([data_batch.edge_labels == 0])) == 0:
                    list_mean_probs["0"]["step" + str(s)].append(torch.tensor(0.5).cuda())
                else:
                    list_mean_probs["0"]["step" + str(s)].append(torch.mean(list_pred_probs[s][data_batch.edge_labels == 0]))
                if sum(sum([data_batch.edge_labels == 1])) == 0:
                    list_mean_probs["1"]["step" + str(s)].append(torch.tensor(0.5).cuda())
                else:
                    list_mean_probs["1"]["step" + str(s)].append(torch.mean(list_pred_probs[s][data_batch.edge_labels == 1]))


            train_losses.update(loss.item(), CONFIG['TRAINING']['BATCH_SIZE']['TRAIN'])
            train_losses1.update(loss_class1.item(), CONFIG['TRAINING']['BATCH_SIZE']['TRAIN'])
            train_losses0.update(loss_class0.item(), CONFIG['TRAINING']['BATCH_SIZE']['TRAIN'])


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
    for i in range(nsteps):
        list_mean_probs_history["0"]["step" + str(i)].append(
            np.mean(torch.stack(list_mean_probs["0"]["step" + str(i)]).cpu().numpy()))
        plt.plot(list_mean_probs_history["0"]["step" + str(i)], '--', label="Class 0 Iter" + str(i))
        list_mean_probs_history["1"]["step" + str(i)].append(
            np.mean(torch.stack(list_mean_probs["1"]["step" + str(i)]).cpu().numpy()))
        plt.plot(list_mean_probs_history["1"]["step" + str(i)], '-', label="Class 1 Iter" + str(i))
    plt.legend(loc='best')
    plt.savefig(results_path + '/images/Mean Probability per Class per Epoch Training.pdf', bbox_inches='tight')



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

    return train_losses, train_losses1, train_losses0, train_precision_class1, train_precision_class0, train_loss_in_history,train_prec1_in_history,train_prec0_in_history,train_prec_in_history,list_mean_probs_history


def validate(CONFIG, val_loader, cnn_model, mpn_model, results_path,epoch,val_loss_in_history,val_prec1_in_history,val_prec0_in_history,val_prec_in_history, val_dataset, dataset_dir,list_mean_probs_history_val):
    val_losses = utils.AverageMeter('losses', ':.4e')
    val_losses1 = utils.AverageMeter('losses', ':.4e')
    val_losses0 = utils.AverageMeter('losses', ':.4e')

    val_batch_time = utils.AverageMeter('batch_time', ':6.3f')
    val_precision_1 = utils.AverageMeter('Val prec class 1', ':6.2f')
    val_precision_0 = utils.AverageMeter('Val prec class 0', ':6.2f')
    val_precision = utils.AverageMeter('Val prec', ':6.2f')

    mpn_model.eval()
    cnn_model.eval()

    # global num_edges, num_edges1
    # num_edges = 0
    # num_edges1 = 0
    nsteps = CONFIG['GRAPH_NET_PARAMS']['num_class_steps']
    list_mean_probs = {"0": {}, "1": {}}
    for i in range(nsteps):
        list_mean_probs["0"]["step" + str(i)] = []
        list_mean_probs["1"]["step" + str(i)] = []


    with torch.no_grad():
        for i, data in enumerate(val_loader):
            if i >= 0:

                start_time = time.time()

                ########### Data extraction ###########
                [bboxes, data_df,max_dist] = data

                len_graphs = [len(item) for item in data_df]
                if CONFIG['CNN_MODEL']['arch'] == 'resnet50':
                    node_embeds, reid_embeds = cnn_model(torch.cat(bboxes, dim=0).cuda())
                else:
                    node_embeds = cnn_model(torch.cat(bboxes, dim=0).cuda())
                    reid_embeds= node_embeds

                # reid embeds needs to be normalize before computing distances

                if CONFIG['CNN_MODEL']['L2norm']:
                    reid_embeds = F.normalize(reid_embeds, p=2, dim=0)
                    node_embeds = F.normalize(node_embeds, p=2, dim=0)

                max_counter = 0
                prev_max_counter  = 0
                edge_ixs = []
                node_label = []
                node_id_cam = []

                # create list called batch, each element: a graph
                batch = []
                # basket_dists = []
                # basket_dists_norm = []
                # spatial_dist_g_l = []
                # spatial_dist_g_l_norm = []

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
                        # node_dist_g = F.pairwise_distance(node_embeds[edge_ixs_g[0]], node_embeds[edge_ixs_g[1]]).view(-1, 1)

                        emb_dist_g_cos = F.cosine_similarity(reid_embeds[edge_ixs_g[0]],
                                                             reid_embeds[edge_ixs_g[1]]).view(-1, 1)
                        # node_dist_g_cos = F.cosine_similarity(node_embeds[edge_ixs_g[0]],
                        #                                       node_embeds[edge_ixs_g[1]]).view(-1, 1)

                        # coordinates of each pair of points
                        xws_1 = np.expand_dims(np.asarray([data_df[g]['xw'].values[item - prev_max_counter] for item in edge_ixs_g_np[0]]), axis=1)
                        yws_1 = np.expand_dims(np.asarray([data_df[g]['yw'].values[item - prev_max_counter] for item in edge_ixs_g_np[0]]),  axis=1)
                        xws_2 = np.expand_dims(np.asarray([data_df[g]['xw'].values[item - prev_max_counter] for item in edge_ixs_g_np[1]]),  axis=1)
                        yws_2 = np.expand_dims(np.asarray([data_df[g]['yw'].values[item - prev_max_counter] for item in edge_ixs_g_np[1]]),    axis=1)
                        points1 = np.concatenate((xws_1, yws_1), axis=1)
                        points2 = np.concatenate((xws_2, yws_2), axis=1)

                        # Convert distances to meters
                        spatial_dist_g = torch.unsqueeze((torch.from_numpy(paired_distances(points1, points2))),    dim=1).cuda()
                        spatial_dist_g_norm = torch.from_numpy(spatial_dist_g.cpu().numpy() / max_dist[g]).cuda()
                        spatial_dist_manh_g = torch.unsqueeze((torch.from_numpy(paired_distances(points1, points2, metric='manhattan'))), dim=1).cuda()
                        spatial_dist_manh_g_norm = torch.from_numpy(spatial_dist_manh_g.cpu().numpy() / max_dist[g]).cuda()



                        if CONFIG['TRAINING']['ONLY_APPEARANCE']:
                            # edge_attr = torch.cat((emb_dist_g, node_dist_g, emb_dist_g_cos, node_dist_g_cos),   dim=1)
                            edge_attr = torch.cat((emb_dist_g, emb_dist_g_cos), dim=1)

                        elif CONFIG['TRAINING']['ONLY_DIST']:
                            edge_attr = torch.cat(
                                (spatial_dist_g_norm.type(torch.float32), spatial_dist_manh_g_norm.type(torch.float32)),
                                dim=1)

                        else:
                            # edge_attr = torch.cat((spatial_dist_g_norm.type(torch.float32), spatial_dist_x_norm.type(torch.float32), spatial_dist_y_norm.type(torch.float32), emb_dist_g), dim=1)
                            edge_attr = torch.cat((spatial_dist_g_norm.type(torch.float32),
                                                   spatial_dist_manh_g_norm.type(torch.float32), emb_dist_g,
                                                   emb_dist_g_cos), dim=1)

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

                        # quitar
                        # basket_dists.append([n for pos, n in enumerate(spatial_dist_g.cpu().numpy()) if         edge_labels_g.cpu().numpy()[pos] == 1])
                        # basket_dists_norm.append([n / max_dist[g] for pos, n in enumerate(spatial_dist_g.cpu().numpy()) if
                        #      edge_labels_g.cpu().numpy()[pos] == 1])
                        # spatial_dist_g_l.append(spatial_dist_g.cpu().numpy())
                        # spatial_dist_g_l_norm.append(spatial_dist_g.cpu().numpy() / max_dist[g])



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
                loss, precision1, precision0, precision,loss_class1, loss_class0,list_pred_probs = compute_loss_acc(outputs, data_batch, criterion ='', criterion_no_reduction='', mode='validate')

                # Fill dictionary with mean probabilities of each class at each step
                nsteps = len(list_pred_probs)

                for s in range(nsteps):
                    if sum(sum([data_batch.edge_labels == 0])) == 0:
                        list_mean_probs["0"]["step" + str(s)].append(torch.tensor(0.5).cuda())
                    else:
                        list_mean_probs["0"]["step" + str(s)].append(
                            torch.mean(list_pred_probs[s][data_batch.edge_labels == 0]))
                    if sum(sum([data_batch.edge_labels == 1])) == 0:
                        list_mean_probs["1"]["step" + str(s)].append(torch.tensor(0.5).cuda())
                    else:
                        list_mean_probs["1"]["step" + str(s)].append(
                            torch.mean(list_pred_probs[s][data_batch.edge_labels == 1]))

                val_losses.update(loss.item(), CONFIG['TRAINING']['BATCH_SIZE']['VAL'])
                val_losses1.update(loss_class1.item(), CONFIG['TRAINING']['BATCH_SIZE']['TRAIN'])
                val_losses0.update(loss_class0.item(), CONFIG['TRAINING']['BATCH_SIZE']['TRAIN'])


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
    for i in range(nsteps):
        list_mean_probs_history_val["0"]["step" + str(i)].append(np.mean(torch.stack(list_mean_probs["0"]["step" + str(i)]).cpu().numpy()))
        plt.plot(list_mean_probs_history_val["0"]["step" + str(i)], '--', label="Class 0 Iter" + str(i))
        list_mean_probs_history_val["1"]["step" + str(i)].append( np.mean(torch.stack(list_mean_probs["1"]["step" + str(i)]).cpu().numpy()))
        plt.plot(list_mean_probs_history_val["1"]["step" + str(i)], '-', label="Class 1 Iter" + str(i))
    plt.legend(loc='best')
    plt.savefig(results_path + '/images/Mean Probability per Class per Epoch Validation.pdf', bbox_inches='tight')


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

    return val_losses, val_losses1, val_losses0, val_precision_1, val_precision_0,val_loss_in_history,val_prec1_in_history, val_prec0_in_history,val_prec_in_history,list_mean_probs_history_val




# # CODIGO PINTAS DIST Y DIST NORM (va en el bucle)

    # spatial_dist_g_l = []
    #         spatial_dist_g_l_norm = []
    #         pets_dists = []
    #         pets_dists_norm = []
    #         terrace_dists = []
    #         terrace_dists_norm = []
    #         lab_dists = []
    #         lab_dists_norm = []
    # basket_dists  = []
    #         garden1_dists = []
    #         garden1_dists_norm = []


# spatial_dist_g = torch.unsqueeze((torch.from_numpy(paired_distances(points1, points2))), dim=1).cuda()
#                 spatial_dist_g_l.append(spatial_dist_g.cpu().numpy())
#                 spatial_dist_g_norm.append(spatial_dist_g.cpu().numpy() / max_dist[g])
#                 spatial_dist_x = torch.abs(torch.from_numpy(xws_1 - xws_2)).cuda()
#                 spatial_dist_x_norm = spatial_dist_x / max_dist[g]
#                 spatial_dist_y = torch.abs(torch.from_numpy(yws_1 - yws_2)).cuda()
#                 spatial_dist_y_norm = spatial_dist_y / max_dist[g]
# #
# edge_labels_g = torch.from_numpy(
#     np.asarray([1 if (data_df[g]['id'].values[data_df[g]['node'].values == edge_ixs_g_np[0][i]] ==
#                       data_df[g]['id'].values[data_df[g]['node'].values == edge_ixs_g_np[1][i]]) else 0
#                 for i in range(edge_ixs_g_np.shape[1])])).type(torch.float).cuda()
# if max_dist[g] == 26.56:  # PETS
#     pets_dists.append(
#         [n for pos, n in enumerate(spatial_dist_g.cpu().numpy()) if edge_labels_g.cpu().numpy()[pos] == 1])
#     pets_dists_norm.append([n / max_dist[g] for pos, n in enumerate(spatial_dist_g.cpu().numpy()) if
#                             edge_labels_g.cpu().numpy()[pos] == 1])
# elif max_dist[g] == 50.83:  # Terrace
#     terrace_dists.append([n for pos, n in enumerate(spatial_dist_g.cpu().numpy()) if
#                           edge_labels_g.cpu().numpy()[pos] == 1])
#     terrace_dists_norm.append([n / max_dist[g] for pos, n in enumerate(spatial_dist_g.cpu().numpy()) if
#                                edge_labels_g.cpu().numpy()[pos] == 1])
# elif max_dist[g] == 44.23:  # Laboratory
#     lab_dists.append([n for pos, n in enumerate(spatial_dist_g.cpu().numpy()) if
#                       edge_labels_g.cpu().numpy()[pos] == 1])
#     lab_dists_norm.append([n / max_dist[g] for pos, n in enumerate(spatial_dist_g.cpu().numpy()) if
#                            edge_labels_g.cpu().numpy()[pos] == 1])
# elif max_dist[g] == 85.23:  # Garden1 CAMPUS
#     garden1_dists.append([n for pos, n in enumerate(spatial_dist_g.cpu().numpy()) if
#                           edge_labels_g.cpu().numpy()[pos] == 1])
#     garden1_dists_norm.append([n / max_dist[g] for pos, n in enumerate(spatial_dist_g.cpu().numpy()) if
#                                edge_labels_g.cpu().numpy()[pos] == 1])



# # CODIGO PINTAR DISTSN desopues de forwards
#  dists = np.concatenate(spatial_dist_g_l)
#             dists_norm = np.concatenate(spatial_dist_g_l_norm)
#
#             pets_dists = np.concatenate(pets_dists)
#             pets_dists_norm = np.concatenate(pets_dists_norm)
#             terrace_dists = np.concatenate(terrace_dists)
#             terrace_dists_norm = np.concatenate(terrace_dists_norm)
#             lab_dists = np.concatenate(lab_dists)
#             lab_dists_norm = np.concatenate(lab_dists_norm)
#             garden1_dists = np.concatenate(garden1_dists)
#             garden1_dists_norm = np.concatenate(garden1_dists_norm)
#
#
#             pets_dists_mean = np.mean(pets_dists)
#             pets_dists_norm_mean = np.mean(pets_dists_norm)
#             terrace_dists_mean = np.mean(terrace_dists)
#             terrace_dists_norm_mean = np.mean(terrace_dists_norm)
#             lab_dists_norm_mean = np.mean(lab_dists_norm)
#             lab_dists_mean = np.mean(lab_dists)
#             garden1_dists_mean = np.mean(garden1_dists)
#             garden1_dists_norm_mean = np.mean(garden1_dists_norm)
#
#
#             plt.figure()
#             plt.subplot(2, 1, 1)
#             plt.scatter(np.arange(len(dists)), dists, c=data_batch.edge_labels.cpu().numpy())
#             plt.plot(np.arange(len(dists)), np.ones(len(dists)) * terrace_dists_mean)
#             plt.plot(np.arange(len(dists)), np.ones(len(dists)) * pets_dists_mean)
#             plt.plot(np.arange(len(dists)), np.ones(len(dists)) * lab_dists_mean)
#             plt.plot(np.arange(len(dists)), np.ones(len(dists)) * garden1_dists_mean)
#
#
#             plt.title('Distances. Mean(1) terrace = ' + str(int(terrace_dists_mean)) + ' Mean(1) pets = ' + str(
#                 int(pets_dists_mean)) + 'Mean(1) Lab = ' + str(int(lab_dists_mean)) + 'Mean(1) Garden1 = ' + str(int(garden1_dists_mean)) )
#             plt.show(block=False)
#             plt.subplot(2, 1, 2)
#             plt.scatter(np.arange(len(dists_norm)), dists_norm, c=data_batch.edge_labels.cpu().numpy())
#             plt.plot(np.arange(len(dists)), np.ones(len(dists)) * terrace_dists_norm_mean)
#             plt.plot(np.arange(len(dists)), np.ones(len(dists)) * pets_dists_norm_mean)
#             plt.plot(np.arange(len(dists)), np.ones(len(dists)) * lab_dists_norm_mean)
#             plt.plot(np.arange(len(dists)), np.ones(len(dists)) * garden1_dists_norm_mean)
#
#
#             plt.title(
#                 'Distances in meters. Mean(1) terrace = ' + str((terrace_dists_norm_mean)) + ' Mean(1) pets = ' + str(
#                     pets_dists_norm_mean) + 'Mean(1) Lab = ' + str(lab_dists_norm_mean) + 'Mean(1) Garden1 = ' + str(garden1_dists_norm_mean) )
#
#             plt.show(block=False)


# dists = np.concatenate(spatial_dist_g_l)
# dists_norm = np.concatenate(spatial_dist_g_l_norm)
# basket_dists = np.concatenate(basket_dists)
# basket_dists_norm = np.concatenate(basket_dists_norm)
# basket_dists_mean = np.mean(basket_dists)
# basket_dists_norm_mean = np.mean(basket_dists_norm)
#
# plt.figure()
# plt.subplot(2, 1, 1)
# plt.scatter(np.arange(len(dists)), dists, c=data_batch.edge_labels.cpu().numpy())
# plt.plot(np.arange(len(dists)), np.ones(len(dists)) * basket_dists_mean)
#
#
# plt.title('Distances. Mean(1) Basketball = ' + str(int(basket_dists_mean))  )
# plt.subplot(2, 1, 2)
# plt.scatter(np.arange(len(dists_norm)), dists_norm, c=data_batch.edge_labels.cpu().numpy())
# plt.plot(np.arange(len(dists)), np.ones(len(dists)) * basket_dists_norm_mean)
#
# plt.title(  'Distances in meters. Mean(1) Basketball = ' + str((basket_dists_norm_mean)) )
#
# plt.show(block=False)