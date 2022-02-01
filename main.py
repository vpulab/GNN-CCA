
import os
import time
import shutil
import yaml
import datetime

import matplotlib
# matplotlib.use('tkAgg')
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pandas as pd
import torch
import argparse


from PIL import Image
from torch.utils.data import DataLoader,Dataset
from torch_geometric.data import Data, Batch

from libs import datasets
from models.resnet import resnet50_fc256, load_pretrained_weights
from models.mpn import MOTMPNet
from models.bdnet import bdnet,top_bdnet_neck_botdropfeat_doubot,top_bdnet_neck_doubot


import utils
from sklearn.metrics.pairwise import paired_distances


from inference import validate_REID, compute_P_R_F, geometrical_association,geometrical_appearance_association
from inference import validate_GNN_cross_camera_association, eval_RANK,validate_REID_with_th

from torchreid.utils import FeatureExtractor


def load_model(CONFIG):

    cnn_arch = CONFIG['CNN_MODEL']['arch']
    # model = MOTMPNet(self.hparams['graph_model_params']).cuda()

    if cnn_arch == 'resnet50':
        # load resnet and trained REID weights
        cnn_model = resnet50_fc256(10, loss='xent', pretrained=True).cuda()
        load_pretrained_weights(cnn_model, CONFIG['CNN_MODEL']['model_weights_path'][cnn_arch])
        # print("DESCOMENTAR LINEA CARGA PESOS MODELO")

        cnn_model.eval()

        # cnn_model.return_embeddings = True
    elif cnn_arch == 'bdnet_market':
        # cnn_model = bdnet(num_classes=751,  loss='softmax',  pretrained=True,  use_gpu= True, feature_extractor = True  )
        cnn_model = top_bdnet_neck_doubot(num_classes=751,  loss='softmax',  pretrained=True,  use_gpu= True, feature_extractor = True )

        load_pretrained_weights(cnn_model, CONFIG['CNN_MODEL']['model_weights_path'][cnn_arch])
        cnn_model.eval()

    elif cnn_arch == 'bdnet_cuhk':
        cnn_model = top_bdnet_neck_botdropfeat_doubot(num_classes=767,  loss='softmax',  pretrained=True,  use_gpu= True, feature_extractor = True  )
        load_pretrained_weights(cnn_model, CONFIG['CNN_MODEL']['model_weights_path'][cnn_arch])
        cnn_model.eval()

    elif cnn_arch == 'osnet_market':
        extractor = FeatureExtractor(
            model_name='osnet_x1_0',
            model_path= CONFIG['CNN_MODEL']['model_weights_path'][cnn_arch],
            device='cuda'
        )
        cnn_model = extractor.model

    elif cnn_arch == 'osnet_ms_c_d':
        extractor = FeatureExtractor(
            model_name='osnet_x1_0',
            model_path= CONFIG['CNN_MODEL']['model_weights_path'][cnn_arch],
            device='cuda'
        )
        cnn_model = extractor.model



    cnn_model.cuda()

    return cnn_model


def load_model_mpn(CONFIG,weights_path=None):
    if weights_path is None:
        model = MOTMPNet(CONFIG['GRAPH_NET_PARAMS'], None,CONFIG['CNN_MODEL']['arch']).cuda()
    else:
        model = MOTMPNet(CONFIG['GRAPH_NET_PARAMS'], None,CONFIG['CNN_MODEL']['arch']).cuda()

        model = utils.load_pretrained_weights(model, weights_path)
    return model

def my_collate(batch):

    bboxes_batches = [item[0] for item in batch]
    df_batches = [item[1] for item in batch]
    max_dist = [item[2] for item in batch]


    return [bboxes_batches, df_batches,max_dist]



global USE_CUDA, CONFIG

USE_CUDA = torch.cuda.is_available()

date = date = str(time.localtime().tm_year) + '-' + str(time.localtime().tm_mon).zfill(2) + '-' + str(
    time.localtime().tm_mday).zfill(2) + \
       ' ' + str(time.localtime().tm_hour).zfill(2) + ':' + str(time.localtime().tm_min).zfill(2) + ':' + str(
    time.localtime().tm_sec).zfill(2)



parser = argparse.ArgumentParser(description='Inference GNN for cross-camera association')
parser.add_argument('--ConfigPath', metavar='DIR', help='Configuration file path')


# Decode CONFIG file information
args = parser.parse_args()
CONFIG = yaml.safe_load(open(args.ConfigPath, 'r'))
# results_path = os.path.join(os.getcwd(), 'inference', str(CONFIG['ID']) + date)

# os.mkdir(results_path)
# os.mkdir(os.path.join(results_path, 'images'))
# os.mkdir(os.path.join(results_path, 'files'))


cnn_model = load_model(CONFIG)

val_dataset = datasets.EPFL_dataset(CONFIG['DATASET_VAL']['NAME'], 'validation', CONFIG, cnn_model)


validation_loader = torch.utils.data.DataLoader(val_dataset, batch_size=CONFIG['TRAINING']['BATCH_SIZE']['VAL'], shuffle=False,
                                       num_workers=CONFIG['DATALOADER']['NUM_WORKERS'], collate_fn=my_collate,pin_memory=CONFIG['DATALOADER']['PIN_MEMORY'])

if CONFIG['MODE'] == 'REID':
    val_prec0_in_history = []
    val_prec1_in_history = []

    reid_dists_l2, labels, reid_distances_cos = validate_REID(validation_loader, cnn_model,CONFIG)
    precisions = []
    precisions_1 = []
    precisions_0 = []
    P_list = []
    R_list = []
    TP_list = []
    FP_list = []
    FN_list =  []
    TN_list = []
    F_list = []

    # FOR EUCLIDEAN DISTANCE
    reid_distances_norm = reid_dists_l2 / np.max(reid_dists_l2)
    print('Max distance = '+  str(np.max(reid_dists_l2)))
    ths = np.arange(0.01, 1.01, 0.01)
    # ths = [0.5]

    for t in ths:
        preds = (reid_distances_norm <= t) * 1

        precisions.append(np.sum((preds == labels) * 1) / len(labels))

        #ACCURACY CLASS 1
        index_label_1 = np.where(np.asarray(labels == 1))
        sum_successes_pos = np.sum(preds[index_label_1] == labels[index_label_1])
        if sum_successes_pos == 0:
            precisions_1.append(0)
        else:
            precisions_1.append(sum_successes_pos / len(labels[index_label_1]))

        #ACCURACY CLASS 0
        index_label_0 = np.where(np.asarray(labels == 0))
        sum_successes_neg = np.sum(preds[index_label_0] == labels[index_label_0])
        if sum_successes_neg == 0:
            precisions_0.append(0)
        else:
            precisions_0.append(sum_successes_neg / len(labels[index_label_0]))

        TP, FP, TN, FN, P, R, F, precision_class0, precision_class1 = compute_P_R_F(preds, np.asarray(labels))

        TP_list.append(TP)
        FP_list.append(FP)
        FN_list.append(FN)
        TN_list.append(TN)
        P_list.append(P)
        R_list.append(R)
        F_list.append(F)


    # MCA = (np.asarray(precisions_0) + np.asarray(precisions_1)) / 2
    # a=1
    # plt.figure()
    # plt.title('REID in ' + CONFIG['DATASET_VAL']['NAME'] + 'MCA: {max_v:.2f}'.format(max_v=np.max(MCA) * 100,
    #                                                                                      max_v2=np.max(
    #                                                                                          precisions) * 100))
    # plt.plot(ths, MCA, label='Mean Class Accuracy (MCA)')
    # plt.xlabel('Threshold')
    # plt.legend()
    # plt.savefig('Classification REID - '+ CONFIG['DATASET_VAL']['NAME'] +'.pdf')

    print('EUCLIDEAN DISTANCE')
    th_max_F = ths[np.where(F_list == np.max(F_list))]
    th_max_F = ths[np.where(F_list == np.max(F_list))]

    print('Max th = ' + str(th_max_F))
    P_max_F = np.asarray(P_list)[np.where(F_list == np.max(F_list))]
    R_max_F = np.asarray(R_list)[np.where(F_list == np.max(F_list))]
    F_max_F = np.asarray(F_list)[np.where(F_list == np.max(F_list))]

    TP_max_F = np.asarray(TP_list)[np.where(F_list == np.max(F_list))]
    FP_max_F = np.asarray(FP_list)[np.where(F_list == np.max(F_list))]
    FN_max_F = np.asarray(FN_list)[np.where(F_list == np.max(F_list))]
    TN_max_F = np.asarray(TN_list)[np.where(F_list == np.max(F_list))]

    print('P= '+ str(P_max_F))
    print('R= '+ str(R_max_F))
    print('F= '+ str(F_max_F))
    print('TP= ' + str(TP_max_F))
    print('FP= ' + str(FP_max_F))
    print('FN= ' + str(FN_max_F))
    print('TN= '+ str(TN_max_F))




    # plt.figure()
    # plt.title('REID in '+ CONFIG['DATASET_VAL']['NAME'] +' R: {max_v:.2f} P: {max_v3:.2f} F: {max_v2:.2f} th: {th:.2f}'.format(max_v=float(R_max_F) * 100,
    #                                                                                             max_v3=float(P_max_F) * 100,
    #                                                                                             max_v2=float(np.max(np.asarray(F_list))) * 100,th=float(th_max_F)))
    #
    # plt.plot(ths, R_list, label='Recall')
    # plt.plot(ths, P_list, label='Precision')
    # plt.plot(ths, F_list, label='F-Score')
    #
    # plt.xlabel('Threshold')
    # plt.legend()
    # plt.savefig('Cross-Cam PRF REID - '+CONFIG['DATASET_VAL']['NAME']+'.pdf')


    # COSINE DISTANCE
    precisions = []
    precisions_1 = []
    precisions_0 = []
    P_list = []
    R_list = []
    TP_list = []
    FP_list = []
    FN_list = []
    TN_list = []
    F_list = []


    reid_distances_norm = np.abs(reid_distances_cos)
    # print('Max distance = ' + str(np.max(reid_distances_cos)))
    ths = np.arange(0.01, 1.01, 0.01)
    # ths = [0.5]

    for t in ths:
        preds = (reid_distances_norm >= t) * 1

        precisions.append(np.sum((preds == labels) * 1) / len(labels))

        # ACCURACY CLASS 1
        index_label_1 = np.where(np.asarray(labels == 1))
        sum_successes_pos = np.sum(preds[index_label_1] == labels[index_label_1])
        if sum_successes_pos == 0:
            precisions_1.append(0)
        else:
            precisions_1.append(sum_successes_pos / len(labels[index_label_1]))

        # ACCURACY CLASS 0
        index_label_0 = np.where(np.asarray(labels == 0))
        sum_successes_neg = np.sum(preds[index_label_0] == labels[index_label_0])
        if sum_successes_neg == 0:
            precisions_0.append(0)
        else:
            precisions_0.append(sum_successes_neg / len(labels[index_label_0]))

        TP, FP, TN, FN, P,R, F, precision_class0, precision_class1 = compute_P_R_F(preds, np.asarray(labels))

        TP_list.append(TP)
        FP_list.append(FP)
        FN_list.append(FN)
        TN_list.append(TN)
        P_list.append(P)
        R_list.append(R)
        F_list.append(F)

    MCA = (np.asarray(precisions_0) + np.asarray(precisions_1)) / 2
    a=1
    plt.figure()
    plt.title('COSINE DISTANCE in ' + CONFIG['DATASET_VAL']['NAME'] + 'MCA: {max_v:.2f}'.format(max_v=np.max(MCA) * 100,
                                                                                         max_v2=np.max(
                                                                                             precisions) * 100))
    plt.plot(ths, MCA, label='Mean Class Accuracy (MCA)')
    plt.xlabel('Threshold')
    plt.legend()
    plt.savefig('Classification REID - '+ CONFIG['DATASET_VAL']['NAME'] +'.pdf')

    print('COSINE DISTANCE')
    th_max_F = ths[np.where(F_list == np.max(F_list))]
    print('Max th = ' + str(th_max_F))
    P_max_F = np.asarray(P_list)[np.where(F_list == np.max(F_list))]
    R_max_F = np.asarray(R_list)[np.where(F_list == np.max(F_list))]
    F_max_F = np.asarray(F_list)[np.where(F_list == np.max(F_list))]

    TP_max_F = np.asarray(TP_list)[np.where(F_list == np.max(F_list))]
    FP_max_F = np.asarray(FP_list)[np.where(F_list == np.max(F_list))]
    FN_max_F = np.asarray(FN_list)[np.where(F_list == np.max(F_list))]
    TN_max_F = np.asarray(TN_list)[np.where(F_list == np.max(F_list))]

    print('P= ' + str(P_max_F))
    print('R= ' + str(R_max_F))
    print('F= ' + str(F_max_F))
    print('TP= ' + str(TP_max_F))
    print('FP= ' + str(FP_max_F))
    print('FN= ' + str(FN_max_F))
    print('TN= ' + str(TN_max_F))

    plt.figure()
    plt.title('COSINE DISTANCE in '+ CONFIG['DATASET_VAL']['NAME'] +' R: {max_v:.2f} P: {max_v3:.2f} F: {max_v2:.2f} th: {th:.2f}'.format(max_v=float(R_max_F) * 100,
                                                                                                max_v3=float(P_max_F) * 100,
                                                                                                max_v2=float(np.max(np.asarray(F_list))) * 100,th=float(th_max_F)))

    plt.plot(ths, R_list, label='Recall')
    plt.plot(ths, P_list, label='Precision')
    plt.plot(ths, F_list, label='F-Score')

    plt.xlabel('Threshold')
    plt.legend()
    plt.savefig('Cross-Cam PRF REID - '+CONFIG['DATASET_VAL']['NAME']+'.pdf')

elif CONFIG['MODE'] == 'GNN_eval':
    dirname = os.path.basename(CONFIG['PRETRAINED_GNN_MODEL'])
    info_label = CONFIG['ID']
    results_path = os.path.join(os.getcwd(), 'results_inference', dirname + info_label + date)

    os.mkdir(results_path)
    os.mkdir(os.path.join(results_path, 'files'))
    with open(os.path.join(results_path, 'files', 'config.yaml'), 'w') as file:
        yaml.safe_dump(CONFIG, file)

    mpn_model = load_model_mpn(CONFIG, CONFIG['PRETRAINED_GNN_MODEL'])
    mpn_model.cuda()
    mpn_model.eval()
    val_loss_in_history = []
    prec0 = []
    prec1 = []

    epoch = 0

    P_list, R_list, F_list, TP_list, FP_list, FN_list, TN_list, rand_index,  mutual_index, homogeneity, completeness, v_measure, prec0, prec1 = validate_GNN_cross_camera_association(CONFIG, validation_loader, cnn_model, mpn_model)

    a=1
    P = np.mean(np.asarray(P_list))
    R = np.mean(np.asarray(R_list))
    F = np.mean(np.asarray(F_list))
    TP = np.sum(np.asarray(TP_list))
    FP = np.sum(np.asarray(FP_list))
    FN = np.sum(np.asarray(FN_list))
    TN = np.sum(np.asarray(TN_list))
    RI = np.mean(np.asarray(rand_index))
    MI = np.mean(np.asarray(mutual_index))
    hom = np.mean(np.asarray(homogeneity))
    com = np.mean(np.asarray(completeness))
    v = np.mean(np.asarray(v_measure))
    prec0 = np.mean(np.asarray(prec0))
    prec1 = np.mean(np.asarray(prec1))

    f = open(results_path + '/results.txt', "w")
    f.write('P= ' + str(P) + '\n')
    f.write('R= ' + str(R) + '\n')
    f.write('F= ' + str(F)+ '\n')
    f.write('TP= ' + str(TP)+ '\n')
    f.write('FP= ' + str(FP)+ '\n')
    f.write('FN= ' + str(FN)+ '\n')
    f.write('TN= ' + str(TN)+ '\n')
    f.write('Rand index mean = ' + str(RI) + '\n')
    f.write('Mutual index mean = ' + str(MI)+ '\n')
    f.write('homogeneity mean = ' + str(hom)+ '\n')
    f.write('completeness mean = ' + str(com)+ '\n')
    f.write('v_measure mean = ' + str(v)+ '\n')
    f.write('Mean prec 0 = ' + str(prec0)+ '\n')
    f.write('Mean prec 1 = ' + str(prec1)+ '\n')

    f.close()


    print('P= '+ str(P))
    print('R= '+ str(R))
    print('F= '+ str(F))
    print('TP= ' + str(TP))
    print('FP= ' + str(FP))
    print('FN= ' + str(FN))
    print('TN= '+ str(TN))
    print('Rand index mean = ' + str(RI))
    print( 'Mutual index mean = ' + str(MI) )
    print( 'homogeneity mean = ' + str(hom) )
    print( 'completeness mean = ' + str(com) )
    print( 'v_measure mean = ' + str(v) )
    print('Mean prec 0 = ' + str(prec0) )
    print('Mean prec 1 = ' + str(prec1) )

elif CONFIG['MODE'] == 'eval_RANK':

    validation_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1,
                                                    shuffle=False,
                                                    num_workers=CONFIG['DATALOADER']['NUM_WORKERS'],
                                                    collate_fn=my_collate,
                                                    pin_memory=CONFIG['DATALOADER']['PIN_MEMORY'])

    rand_index, mutual_index, homogeneity, completeness, v_measure = eval_RANK(validation_loader, cnn_model, CONFIG)

    RI = np.mean(np.asarray(rand_index))
    MI = np.mean(np.asarray(mutual_index))
    hom = np.mean(np.asarray(homogeneity))
    com = np.mean(np.asarray(completeness))
    v = np.mean(np.asarray(v_measure))

    print('Rand index mean = ' + str(RI))
    print('Mutual index mean = ' + str(MI))
    print('homogeneity mean = ' + str(hom))
    print('completeness mean = ' + str(com))
    print('v_measure mean = ' + str(v))

elif CONFIG['MODE'] == 'REID_th':

    validation_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1,
                                                    shuffle=False,
                                                    num_workers=CONFIG['DATALOADER']['NUM_WORKERS'],
                                                    collate_fn=my_collate,
                                                    pin_memory=CONFIG['DATALOADER']['PIN_MEMORY'])

    val_prec0_in_history = []
    val_prec1_in_history = []
    th_L2 = CONFIG['OPT_TH']['L2'][CONFIG['CNN_MODEL']['arch']][CONFIG['DATASET_VAL']['NAME']] # con Resnet50 all
    th_cos = CONFIG['OPT_TH']['COS'][CONFIG['CNN_MODEL']['arch']][CONFIG['DATASET_VAL']['NAME']]
    max_dist_L2 = CONFIG['MAX_DIST_L2'][CONFIG['CNN_MODEL']['arch']][CONFIG['DATASET_VAL']['NAME']]
    L2_rand_index, L2_mutual_index, L2_homogeneity, L2_completeness, L2_v_measure,cos_rand_index, cos_mutual_index, cos_homogeneity, cos_completeness, cos_v_measure =\
        validate_REID_with_th(CONFIG,validation_loader, cnn_model, th_L2, max_dist_L2, th_cos)


    L2_RI = np.mean(np.asarray(L2_rand_index))
    L2_MI = np.mean(np.asarray(L2_mutual_index))
    L2_H = np.mean(np.asarray(L2_homogeneity))
    L2_C = np.mean(np.asarray(L2_completeness))
    L2_V = np.mean(np.asarray(L2_v_measure))

    COS_RI = np.mean(np.asarray(cos_rand_index))
    COS_MI = np.mean(np.asarray(cos_mutual_index))
    COS_H = np.mean(np.asarray(cos_homogeneity))
    COS_C = np.mean(np.asarray(cos_completeness))
    COS_V = np.mean(np.asarray(cos_v_measure))



    print( 'L2 Rand index mean = ' + str(L2_RI) )
    print( 'L2 Mutual index mean = ' + str(L2_MI) )
    print( 'L2 homogeneity mean = ' + str(L2_H) )
    print( 'L2 completeness mean = ' + str(L2_C))
    print( 'L2 v_measure mean = ' + str(L2_V) )

    print('COS Rand index mean = ' + str(COS_RI))
    print('COS Mutual index mean = ' + str(COS_MI))
    print('COS homogeneity mean = ' + str(COS_H))
    print('COS completeness mean = ' + str(COS_C))
    print('COS v_measure mean = ' + str(COS_V))

elif CONFIG['MODE'] == 'geometrical_association':

    rand_index, mutual_index, homogeneity, completeness, v_measure     = geometrical_association(CONFIG, validation_loader)

    RI = np.mean(np.asarray(rand_index))
    MI = np.mean(np.asarray(mutual_index))
    hom = np.mean(np.asarray(homogeneity))
    com = np.mean(np.asarray(completeness))
    v = np.mean(np.asarray(v_measure))

    print('Rand index mean = ' + str(RI))
    print('Mutual index mean = ' + str(MI))
    print('homogeneity mean = ' + str(hom))
    print('completeness mean = ' + str(com))
    print('v_measure mean = ' + str(v))

elif CONFIG['MODE'] == 'geometrical_appearance_association':
    dirname = os.path.basename(CONFIG['PRETRAINED_GNN_MODEL'])
    info_label = CONFIG['ID']
    results_path = os.path.join(os.getcwd(), 'results_inference', dirname + info_label + date)

    os.mkdir(results_path)
    os.mkdir(os.path.join(results_path, 'files'))
    with open(os.path.join(results_path, 'files', 'config.yaml'), 'w') as file:
        yaml.safe_dump(CONFIG, file)

    th = CONFIG['OPT_TH']['L2'][CONFIG['CNN_MODEL']['arch']][CONFIG['DATASET_VAL']['NAME']]
    max_dist = CONFIG['MAX_DIST_L2'][CONFIG['CNN_MODEL']['arch']][CONFIG['DATASET_VAL']['NAME']]

    rand_index, mutual_index, homogeneity, completeness, v_measure = geometrical_appearance_association(CONFIG, validation_loader,cnn_model, th,max_dist)

    RI = np.mean(np.asarray(rand_index))
    MI = np.mean(np.asarray(mutual_index))
    hom = np.mean(np.asarray(homogeneity))
    com = np.mean(np.asarray(completeness))
    v = np.mean(np.asarray(v_measure))

    print('Rand index mean = ' + str(RI))
    print('Mutual index mean = ' + str(MI))
    print('homogeneity mean = ' + str(hom))
    print('completeness mean = ' + str(com))
    print('v_measure mean = ' + str(v))

    f = open(results_path + '/results.txt', "w")

    f.write('Rand index mean = ' + str(RI) + '\n')
    f.write('Mutual index mean = ' + str(MI) + '\n')
    f.write('homogeneity mean = ' + str(hom) + '\n')
    f.write('completeness mean = ' + str(com) + '\n')
    f.write('v_measure mean = ' + str(v) + '\n')


    f.close()