
import os
import time
import shutil
import yaml
import datetime


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
import networkx as nx
from sklearn.metrics.pairwise import paired_distances
from sklearn import metrics
import utils

list_cam_colors = list(['royalblue', 'darkorange','green','firebrick'])



def compute_P_R_F(preds, labels):
    index_label_1 = np.where(labels == 1)
    index_label_0 = np.where(labels == 0)
    precision_class1 = list()
    precision_class0 = list()
    ## Add computation of precision of class 1 and class 0
    sum_successes_1 = np.sum(preds[index_label_1] == labels[index_label_1])
    if sum_successes_1 == 0:
        precision_class1.append(0)
    else:
        precision_class1.append((sum_successes_1 / len(labels[index_label_1])) * 100.0)

    # Precision class 0
    sum_successes_0 = np.sum(preds[index_label_0] == labels[index_label_0])
    if sum_successes_0 == 0:
        precision_class0.append(0)
    else:
        precision_class0.append((sum_successes_0 / len(labels[index_label_0])) * 100.0)

    ##

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

    return TP, FP, TN, FN, P,R, F, precision_class0, precision_class1

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
                [bboxes, data_df,max_dist] = data

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


    rand_index = []


    mutual_index = []
    homogeneity = []
    completeness = []
    v_measure = []

    precision_1_list = []
    precision_0_list = []

    tic = time.time()
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            if i >= 0 :

                start_time = time.time()

                ########### Data extraction ###########
                [bboxes, data_df,max_dist] = data

                len_graphs = [len(item) for item in data_df]
                if CONFIG['CNN_MODEL']['arch'] == 'resnet50':
                    node_embeds, reid_embeds = cnn_model(torch.cat(bboxes, dim=0).cuda())
                else:
                    node_embeds = cnn_model(torch.cat(bboxes, dim=0).cuda())
                    reid_embeds= node_embeds


                reid_embeds = F.normalize(reid_embeds, p=2, dim=0)
                node_embeds = F.normalize(node_embeds, p=2, dim=0)

                max_counter = 0
                prev_max_counter = 0


                # create list called batch, each element: a graph
                batch = []

                for g in range(len(len_graphs)):


                    data_df[g] = data_df[g].assign(
                        node=np.asarray(range(max_counter, max_counter + len(data_df[g]))))
                    print('Testing frame ' + str(data_df[g].iloc[0]['frame']))

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

                    # features reid distances between each pair of points
                    emb_dist_g = F.pairwise_distance(reid_embeds[edge_ixs_g[0]], reid_embeds[edge_ixs_g[1]]).view(
                        -1, 1)

                    emb_dist_g_cos = F.cosine_similarity(reid_embeds[edge_ixs_g[0]],
                                                         reid_embeds[edge_ixs_g[1]]).view(-1, 1)

                    # coordinates of each pair of points
                    xws_1 = np.expand_dims(
                        np.asarray([data_df[g]['xw'].values[item - prev_max_counter] for item in edge_ixs_g_np[0]]),
                        axis=1)
                    yws_1 = np.expand_dims(
                        np.asarray([data_df[g]['yw'].values[item - prev_max_counter] for item in edge_ixs_g_np[0]]),
                        axis=1)
                    xws_2 = np.expand_dims(
                        np.asarray([data_df[g]['xw'].values[item - prev_max_counter] for item in edge_ixs_g_np[1]]),
                        axis=1)
                    yws_2 = np.expand_dims(
                        np.asarray([data_df[g]['yw'].values[item - prev_max_counter] for item in edge_ixs_g_np[1]]),
                        axis=1)
                    points1 = np.concatenate((xws_1, yws_1), axis=1)
                    points2 = np.concatenate((xws_2, yws_2), axis=1)

                    # Convert distances to meters
                    spatial_dist_g = torch.unsqueeze((torch.from_numpy(paired_distances(points1, points2))),
                                                     dim=1).cuda()
                    spatial_dist_g_norm = torch.from_numpy(spatial_dist_g.cpu().numpy() / max_dist[g]).cuda()
                    spatial_dist_manh_g = torch.unsqueeze(
                        (torch.from_numpy(paired_distances(points1, points2, metric='manhattan'))), dim=1).cuda()
                    spatial_dist_manh_g_norm = torch.from_numpy(spatial_dist_manh_g.cpu().numpy() / max_dist[g]).cuda()


                    if CONFIG['TRAINING']['ONLY_APPEARANCE']:
                        edge_attr = torch.cat((emb_dist_g, emb_dist_g_cos), dim=1)

                    elif CONFIG['TRAINING']['ONLY_DIST']:
                        edge_attr = torch.cat(
                            (spatial_dist_g_norm.type(torch.float32), spatial_dist_manh_g_norm.type(torch.float32)),
                            dim=1)

                    else:
                        edge_attr = torch.cat((spatial_dist_g_norm.type(torch.float32),
                                               spatial_dist_manh_g_norm.type(torch.float32), emb_dist_g,
                                               emb_dist_g_cos), dim=1)



                     # EDGE LABELS

                    edge_labels_g = torch.from_numpy(
                        np.asarray(
                            [1 if (data_df[g]['id'].values[data_df[g]['node'].values == edge_ixs_g_np[0][i]] ==
                                   data_df[g]['id'].values[data_df[g]['node'].values == edge_ixs_g_np[1][i]]) else 0
                             for i in range(edge_ixs_g_np.shape[1])])).type(torch.float).cuda()


                    edge_ixs_g = edge_ixs_g - torch.min(edge_ixs_g)
                    data = Data(x=node_embeds_g, edge_index=edge_ixs_g, y=node_label_g, edge_attr=edge_attr,
                                edge_labels=edge_labels_g)


                    batch.append(data)
                    prev_max_counter = max_counter



                data_batch = Batch.from_data_list(batch)

                ########### Forward ###########

                outputs = mpn_model(data_batch)
                labels_edges_GT = data_batch.edge_labels.view(-1).cpu().numpy()

                preds = outputs['classified_edges'][-1].view(-1)

                sig = torch.nn.Sigmoid()
                preds_prob = sig(preds)
                predictions = (preds_prob >= 0.5) * 1


                # CLUSTERING IDENTITIES MEASURES
                edge_list = data_batch.edge_index.cpu().numpy()

                GT_active_edges = [(edge_list[0][pos], edge_list[1][pos]) for pos, p in enumerate(labels_edges_GT) if p == 1]
                G_GT = nx.DiGraph(GT_active_edges)
                ID_GT, n_clusters_GT = utils.compute_SCC_and_Clusters(G_GT,data_batch.num_nodes)


                predicted_active_edges = [(edge_list[0][pos], edge_list[1][pos]) for pos, p in enumerate(predictions) if p == 1]
                G = nx.DiGraph(predicted_active_edges)
                ID_pred, n_clusters_pred = utils.compute_SCC_and_Clusters(G,data_batch.num_nodes)



                if CONFIG['PRUNING']:
                    predictions, predicted_active_edges = utils.remove_edges_single_direction(predicted_active_edges,
                                                                                     predictions, edge_list)
                    G = nx.DiGraph(predicted_active_edges)
                    ID_pred, n_clusters_pred = utils.compute_SCC_and_Clusters(G, data_batch.num_nodes)


                # # COMPUTE GREEDY ROUNDING
                if CONFIG['ROUNDING']:
                    predictions_r = utils.compute_rounding(data_batch, predictions.view(-1), preds_prob,predicted_active_edges)
                    if predictions_r != []:
                        predicted_active_edges = [(edge_list[0][pos], edge_list[1][pos]) for pos, p in
                                                      enumerate(predictions_r) if p == 1]
                        predictions = predictions_r

                    else:
                        predicted_active_edges = [(edge_list[0][pos], edge_list[1][pos]) for pos, p in
                                                      enumerate(predictions) if p == 1]


                    G = nx.DiGraph(predicted_active_edges)
                    ID_pred, rounding_n_clusters_pred = utils.compute_SCC_and_Clusters(G, data_batch.num_nodes)


                if CONFIG['PRUNNING']:
                    predictions, predicted_active_edges = utils.remove_edges_single_direction(predicted_active_edges,
                                                                                              predictions, edge_list)
                    G = nx.DiGraph(predicted_active_edges)
                    ID_pred, n_clusters_pred = utils.compute_SCC_and_Clusters(G, data_batch.num_nodes)


                if CONFIG['SPLITTING']:
                    predictions = utils.disjoint_big_clusters(ID_pred, predictions, preds_prob, edge_list,
                                                              data_batch, predicted_active_edges, G)
                    predicted_active_edges = [(edge_list[0][pos], edge_list[1][pos]) for pos, p in
                                              enumerate(predictions) if p == 1]
                    G = nx.DiGraph(predicted_active_edges)
                    ID_pred, disjoint_n_clusters_pred = utils.compute_SCC_and_Clusters(G, data_batch.num_nodes)



                rand_index.append(metrics.adjusted_rand_score(ID_GT, ID_pred))


                TP, FP, TN, FN, P, R, FS, precision0, precision1 = compute_P_R_F(predictions.cpu().numpy(), labels_edges_GT)

                precision_1_list.append(np.sum(np.asarray([item for item in precision1])) / len(precision1))
                precision_0_list.append(np.sum(np.asarray([item for item in precision0])) / len(precision0))


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


                val_batch_time.update(time.time() - start_time)

                if i % 10 == 0:
                    print('Testing validation batch [{0}/{1}]\t'
                          '{et}<{eta}'.format(i, len(val_loader),
                                              et=str(datetime.timedelta(seconds=int(val_batch_time.sum))),
                                              eta=str(datetime.timedelta(seconds=int(val_batch_time.avg * (len(val_loader) - i))))))

    toc = time.time()

    print(['with bridges  eval lab elapsed time ' + str(toc-tic)])
    return P_list, R_list, F_list, TP_list, FP_list, FN_list, TN_list, rand_index,mutual_index, homogeneity, completeness, v_measure, precision_0_list, precision_1_list


def eval_RANK(val_loader, model,CONFIG):
    # EVAL RANK X
    val_batch_time = utils.AverageMeter('batch_time', ':6.3f')

    model.eval()
    edge_label = []

    mutual_index = []
    homogeneity = []
    completeness = []
    v_measure = []
    rand_index = []
    with torch.no_grad():
        for it, data in enumerate(val_loader):
            if it >= 0:
                start_time = time.time()

                ########### Data extraction ###########
                [bboxes, data_df, max_d] = data

                len_graphs = [len(item) for item in data_df]

                if CONFIG['CNN_MODEL']['arch'] == 'resnet50':
                    node_embeds, reid_embeds = model(torch.cat(bboxes, dim=0).cuda())
                else:
                    node_embeds = model(torch.cat(bboxes, dim=0).cuda())
                    reid_embeds= node_embeds

                dist_mat = torch.cdist(reid_embeds, reid_embeds, p=2)
                dist_mat = dist_mat.cpu().numpy()

                rerank = CONFIG['RERANK']
                if rerank:
                    distmat_qq = torch.cdist(reid_embeds, reid_embeds)
                    distmat_gg = torch.cdist(reid_embeds, reid_embeds)
                    dist_mat = utils.re_ranking(dist_mat, distmat_qq.cpu().numpy(), distmat_gg.cpu().numpy())


                max_counter = 0

                apply_cam_restrictions = True
                for g in range(len(len_graphs)):

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

                val_batch_time.update(time.time() - start_time)


                GT_active_edges = [(edge_ixs_g_np[0][pos], edge_ixs_g_np[1][pos]) for pos, p in enumerate(edge_label)
                                   if p == 1]
                G_GT = nx.DiGraph(GT_active_edges)
                ID_GT, n_clusters_GT = utils.compute_SCC_and_Clusters(G_GT, data_df[0].node.shape[0])

                G = nx.DiGraph(pred_active_edges)
                ID_pred, n_clusters_pred = utils.compute_SCC_and_Clusters(G, data_df[0].node.shape[0])


                rand_index.append(metrics.adjusted_rand_score(ID_GT, ID_pred))
                mutual_index.append(metrics.adjusted_mutual_info_score(ID_GT, ID_pred))
                homogeneity.append(metrics.homogeneity_score(ID_GT, ID_pred))
                completeness.append(metrics.completeness_score(ID_GT, ID_pred))
                v_measure.append(metrics.v_measure_score(ID_GT, ID_pred))



                if it % 10 == 0:
                    print('Testing validation batch [{0}/{1}]\t'
                          '{et}<{eta}'.format(it, len(val_loader),
                                              et=str(datetime.timedelta(seconds=int(val_batch_time.sum))),
                                              eta=str(datetime.timedelta(
                                                  seconds=int(val_batch_time.avg * (len(val_loader) - it))))))


    return rand_index, mutual_index, homogeneity, completeness, v_measure

def validate_REID_with_th(CONFIG,val_loader, cnn_model,  th_L2, max_dist_L2, th_cos):

    # RUN THIS ONCE YOU ALREADY HAVE THE OPTIMAL TH
    #  EVALS FRAME BY FRAME
    val_batch_time = utils.AverageMeter('batch_time', ':6.3f')

    cnn_model.eval()

    L2_rand_index = []
    L2_mutual_index = []
    L2_homogeneity = []
    L2_completeness = []
    L2_v_measure = []

    cos_rand_index = []
    cos_mutual_index = []
    cos_homogeneity = []
    cos_completeness = []
    cos_v_measure = []

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
                    reid_embeds = node_embeds

                # features reid distances between each pair of points
                max_counter = 0

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
                    L2_reid_distances = (F.pairwise_distance(reid_embeds[edge_ixs_g[0]], reid_embeds[edge_ixs_g[1]]).cpu().numpy())
                    L2_reid_distances_norm = L2_reid_distances / max_dist_L2
                    L2_predictions = (L2_reid_distances_norm <= th_L2) * 1

                    cos_reid_distances = (F.cosine_similarity(reid_embeds[edge_ixs_g[0]], reid_embeds[edge_ixs_g[1]]).cpu().numpy())
                    cos_reid_distances_norm = np.abs(cos_reid_distances)
                    cos_predictions = (cos_reid_distances_norm >= th_cos) * 1

                    edge_label = (np.asarray(
                        [1 if (data_df[g]['id'].values[data_df[g]['node'].values == edge_ixs_g_np[0][i]] ==
                               data_df[g]['id'].values[data_df[g]['node'].values == edge_ixs_g_np[1][i]]) else 0
                         for i in range(edge_ixs_g_np.shape[1])]))


                    GT_active_edges = [(edge_ixs_g_np[0][pos], edge_ixs_g_np[1][pos]) for pos, p in
                                       enumerate(edge_label)
                                       if p == 1]
                    G_GT = nx.DiGraph(GT_active_edges)
                    ID_GT, n_clusters_GT = utils.compute_SCC_and_Clusters(G_GT, data_df[0].node.shape[0])


                    L2_predicted_active_edges = [(edge_ixs_g_np[0][pos], edge_ixs_g_np[1][pos]) for pos, p in enumerate(L2_predictions) if  p == 1]
                    L2_G = nx.DiGraph(L2_predicted_active_edges)
                    L2_ID_pred, L2_n_clusters_pred = utils.compute_SCC_and_Clusters(L2_G, data_df[0].node.shape[0])


                    L2_rand_index.append(metrics.adjusted_rand_score(ID_GT, L2_ID_pred))
                    L2_mutual_index.append(metrics.adjusted_mutual_info_score(ID_GT, L2_ID_pred))
                    L2_homogeneity.append(metrics.homogeneity_score(ID_GT, L2_ID_pred))
                    L2_completeness.append(metrics.completeness_score(ID_GT, L2_ID_pred))
                    L2_v_measure.append(metrics.v_measure_score(ID_GT, L2_ID_pred))

                    cos_predicted_active_edges = [(edge_ixs_g_np[0][pos], edge_ixs_g_np[1][pos]) for pos, p in enumerate(cos_predictions) if p == 1]
                    cos_G = nx.DiGraph(cos_predicted_active_edges)
                    cos_ID_pred, cos_n_clusters_pred = utils.compute_SCC_and_Clusters(cos_G, data_df[0].node.shape[0])

                    cos_rand_index.append(metrics.adjusted_rand_score(ID_GT, cos_ID_pred))
                    cos_mutual_index.append(metrics.adjusted_mutual_info_score(ID_GT, cos_ID_pred))
                    cos_homogeneity.append(metrics.homogeneity_score(ID_GT, cos_ID_pred))
                    cos_completeness.append(metrics.completeness_score(ID_GT, cos_ID_pred))
                    cos_v_measure.append(metrics.v_measure_score(ID_GT, cos_ID_pred))


                val_batch_time.update(time.time() - start_time)

                if i % 10 == 0:
                    print('Testing validation batch [{0}/{1}]\t'
                          '{et}<{eta}'.format(i, len(val_loader),
                                              et=str(datetime.timedelta(seconds=int(val_batch_time.sum))),
                                              eta=str(datetime.timedelta(seconds=int(val_batch_time.avg * (len(val_loader) - i))))))


    return  L2_rand_index, L2_mutual_index, L2_homogeneity, L2_completeness, L2_v_measure,cos_rand_index, cos_mutual_index, cos_homogeneity, cos_completeness, cos_v_measure

def geometrical_association(CONFIG, val_loader):

    val_batch_time = utils.AverageMeter('batch_time', ':6.3f')

    rand_index = []
    mutual_index = []
    homogeneity = []
    completeness = []
    v_measure = []


    for i, data in enumerate(val_loader):
        if i >= 0 :

            start_time = time.time()

            ########### Data extraction ###########
            [bboxes, data_df,max_dist] = data

            len_graphs = [len(item) for item in data_df]

            max_counter = 0
            prev_max_counter = 0

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
                if CONFIG['NORM_TO_M']:
                    spatial_dist_g = torch.from_numpy(spatial_dist_g.cpu().numpy() / max_dist[g]).cuda()

                # EDGE LABELS

                edge_labels_g = torch.from_numpy(
                    np.asarray(
                        [1 if (data_df[g]['id'].values[data_df[g]['node'].values == edge_ixs_g_np[0][i]] ==
                               data_df[g]['id'].values[data_df[g]['node'].values == edge_ixs_g_np[1][i]]) else 0
                         for i in range(edge_ixs_g_np.shape[1])])).type(torch.float).cuda()

                # bajar rango a 0 de edge_ixs_g
                edge_ixs_g = edge_ixs_g - torch.min(edge_ixs_g)

                data = Data(edge_index=edge_ixs_g, y=node_label_g, edge_attr=spatial_dist_g,
                            edge_labels=edge_labels_g)

                batch.append(data)
                prev_max_counter = max_counter


            data_batch = Batch.from_data_list(batch)


            #
            # ########### Forward ###########
            #
            labels_edges_GT = edge_labels_g


            if CONFIG['NORM_TO_M']:
                predictions = (spatial_dist_g < (CONFIG['GEOM_TH'][CONFIG['DATASET_VAL']['NAME']]/max_dist[0])) * 1
            else:
                predictions = (spatial_dist_g < CONFIG['GEOM_TH'][CONFIG['DATASET_VAL']['NAME']]) * 1



            # CLUSTERING IDENTITIES MEASURES
            edge_list = data_batch.edge_index.cpu().numpy()

            GT_active_edges = [(edge_list[0][pos], edge_list[1][pos]) for pos, p in enumerate(labels_edges_GT) if p == 1]
            G_GT = nx.DiGraph(GT_active_edges)
            ID_GT, n_clusters_GT = utils.compute_SCC_and_Clusters(G_GT,data_batch.num_nodes)


            predicted_active_edges = [(edge_list[0][pos], edge_list[1][pos]) for pos, p in enumerate(predictions) if p == 1]
            G = nx.DiGraph(predicted_active_edges)
            ID_pred, n_clusters_pred = utils.compute_SCC_and_Clusters(G,data_batch.num_nodes)


            if CONFIG['SPLITTING']:
                predictions = utils.disjoint_big_clusters(ID_pred, predictions, spatial_dist_g, edge_list,
                                                          data_batch, predicted_active_edges, G)
                predicted_active_edges = [(edge_list[0][pos], edge_list[1][pos]) for pos, p in
                                          enumerate(predictions) if p == 1]
                G = nx.DiGraph(predicted_active_edges)
                ID_pred, disjoint_n_clusters_pred = utils.compute_SCC_and_Clusters(G, data_batch.num_nodes)


            # # COMPUTE GREEDY ROUNDING
            if CONFIG['ROUNDING']:
                predicted_active_edges = [(edge_list[0][pos], edge_list[1][pos]) for pos, p in   enumerate(predictions) if p == 1]
                predictions_r = utils.compute_rounding(data_batch, predictions.view(-1), spatial_dist_g, predicted_active_edges)
                if predictions_r != []:
                    predicted_active_edges_r = [(edge_list[0][pos], edge_list[1][pos]) for pos, p in
                                              enumerate(predictions_r) if p == 1]
                    predictions = predictions_r

                else:
                    predicted_active_edges = [(edge_list[0][pos], edge_list[1][pos]) for pos, p in
                                              enumerate(predictions) if p == 1]
                G = nx.DiGraph(predicted_active_edges)
                ID_pred, disjoint_n_clusters_pred = utils.compute_SCC_and_Clusters(G, data_batch.num_nodes)

            rand_index.append(metrics.adjusted_rand_score(ID_GT, ID_pred))
            mutual_index.append(metrics.adjusted_mutual_info_score(ID_GT, ID_pred))
            homogeneity.append(metrics.homogeneity_score(ID_GT, ID_pred))
            completeness.append(metrics.completeness_score(ID_GT, ID_pred))
            v_measure.append(metrics.v_measure_score(ID_GT, ID_pred))





            val_batch_time.update(time.time() - start_time)

            if i % 10 == 0:
                print('Testing validation batch [{0}/{1}]\t'
                      '{et}<{eta}'.format(i, len(val_loader),
                                          et=str(datetime.timedelta(seconds=int(val_batch_time.sum))),
                                          eta=str(datetime.timedelta(seconds=int(val_batch_time.avg * (len(val_loader) - i))))))


    return rand_index, mutual_index, homogeneity, completeness, v_measure

def geometrical_appearance_association(CONFIG, val_loader, cnn_model,th,max_dist_L2):

    val_batch_time = utils.AverageMeter('batch_time', ':6.3f')
    cnn_model.eval()

    rand_index = []
    mutual_index = []
    homogeneity = []
    completeness = []
    v_measure = []


    for i, data in enumerate(val_loader):
        if i >= 0 :

            start_time = time.time()

            ########### Data extraction ###########
            [bboxes, data_df,max_dist] = data

            len_graphs = [len(item) for item in data_df]

            if CONFIG['CNN_MODEL']['arch'] == 'resnet50':
                node_embeds, reid_embeds = cnn_model(torch.cat(bboxes, dim=0).cuda())
            else:
                node_embeds = cnn_model(torch.cat(bboxes, dim=0).cuda())
                reid_embeds = node_embeds


            max_counter = 0
            prev_max_counter = 0

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

                # features reid distances between each pair of points
                emb_dist_g = F.pairwise_distance(reid_embeds[edge_ixs_g[0]], reid_embeds[edge_ixs_g[1]]).view(
                    -1, 1)
                emb_dist_g = emb_dist_g / max_dist_L2



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
                if CONFIG['NORM_TO_M']:
                    spatial_dist_g = torch.from_numpy(spatial_dist_g.cpu().numpy() / max_dist[g]).cuda()

                edge_attr = torch.cat((spatial_dist_g.type(torch.float32), emb_dist_g), dim=1).permute(1, 0)

                # EDGE LABELS

                edge_labels_g = torch.from_numpy(
                    np.asarray(
                        [1 if (data_df[g]['id'].values[data_df[g]['node'].values == edge_ixs_g_np[0][i]] ==
                               data_df[g]['id'].values[data_df[g]['node'].values == edge_ixs_g_np[1][i]]) else 0
                         for i in range(edge_ixs_g_np.shape[1])])).type(torch.float).cuda()

                # bajar rango a 0 de edge_iuxs_g
                edge_ixs_g = edge_ixs_g - torch.min(edge_ixs_g)

                data = Data(edge_index=edge_ixs_g, y=node_label_g, edge_attr=edge_attr,
                            edge_labels=edge_labels_g)

                batch.append(data)
                prev_max_counter = max_counter


            data_batch = Batch.from_data_list(batch)

            #
            # ########### Forward ###########
            #
            labels_edges_GT = edge_labels_g



            predictions = torch.logical_and(edge_attr[0] < (CONFIG['GEOM_TH'][CONFIG['DATASET_VAL']['NAME'] ]/ max_dist[0]) * 1,  (edge_attr[1]<  th) * 1)*1


            # CLUSTERING IDENTITIES MEASURES
            edge_list = data_batch.edge_index.cpu().numpy()

            GT_active_edges = [(edge_list[0][pos], edge_list[1][pos]) for pos, p in enumerate(labels_edges_GT) if p == 1]
            G_GT = nx.DiGraph(GT_active_edges)
            ID_GT, n_clusters_GT = utils.compute_SCC_and_Clusters(G_GT,data_batch.num_nodes)

            predicted_active_edges = [(edge_list[0][pos], edge_list[1][pos]) for pos, p in enumerate(predictions) if p == 1]
            G = nx.DiGraph(predicted_active_edges)
            ID_pred, n_clusters_pred = utils.compute_SCC_and_Clusters(G,data_batch.num_nodes)


            if CONFIG['SPLITTING']:
                predictions = utils.disjoint_big_clusters(ID_pred, predictions, spatial_dist_g, edge_list,
                                                          data_batch, predicted_active_edges, G)
                predicted_active_edges = [(edge_list[0][pos], edge_list[1][pos]) for pos, p in
                                          enumerate(predictions) if p == 1]
                G = nx.DiGraph(predicted_active_edges)
                ID_pred, disjoint_n_clusters_pred = utils.compute_SCC_and_Clusters(G, data_batch.num_nodes)


            # # COMPUTE GREEDY ROUNDING
            if CONFIG['ROUNDING']:
                predicted_active_edges = [(edge_list[0][pos], edge_list[1][pos]) for pos, p in   enumerate(predictions) if p == 1]
                predictions_r = utils.compute_rounding(data_batch, predictions.view(-1), spatial_dist_g, predicted_active_edges)
                if predictions_r != []:
                    predicted_active_edges_r = [(edge_list[0][pos], edge_list[1][pos]) for pos, p in
                                              enumerate(predictions_r) if p == 1]
                    predictions = predictions_r

                else:
                    predicted_active_edges = [(edge_list[0][pos], edge_list[1][pos]) for pos, p in
                                              enumerate(predictions) if p == 1]
                G = nx.DiGraph(predicted_active_edges)
                ID_pred, disjoint_n_clusters_pred = utils.compute_SCC_and_Clusters(G, data_batch.num_nodes)

            rand_index.append(metrics.adjusted_rand_score(ID_GT, ID_pred))
            mutual_index.append(metrics.adjusted_mutual_info_score(ID_GT, ID_pred))
            homogeneity.append(metrics.homogeneity_score(ID_GT, ID_pred))
            completeness.append(metrics.completeness_score(ID_GT, ID_pred))
            v_measure.append(metrics.v_measure_score(ID_GT, ID_pred))





            val_batch_time.update(time.time() - start_time)

            if i % 10 == 0:
                print('Testing validation batch [{0}/{1}]\t'
                      '{et}<{eta}'.format(i, len(val_loader),
                                          et=str(datetime.timedelta(seconds=int(val_batch_time.sum))),
                                          eta=str(datetime.timedelta(seconds=int(val_batch_time.avg * (len(val_loader) - i))))))


    return rand_index, mutual_index, homogeneity, completeness, v_measure
