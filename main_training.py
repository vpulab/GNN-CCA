# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

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
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import torch.optim as optim
import torch.nn.functional as F
from torch import optim as optim_module
import imgaug as ia
from imgaug import augmenters as iaa

from PIL import Image
from torch.utils.data import DataLoader,Dataset
from torch_geometric.data import Data, Batch

from datasets import datasets
from models.resnet import resnet50_fc256, load_pretrained_weights
from models.mpn import MOTMPNet
from models.bdnet import bdnet,top_bdnet_neck_botdropfeat_doubot

from torch_geometric.utils import to_networkx
import networkx as nx
from skimage.io import imread

import utils
from sklearn.metrics.pairwise import paired_distances


from train import train, validate, validate_REID, compute_P_R_F, geometrical_association
from train import validate_GNN_cross_camera_association, eval_RANK,validate_REID_with_th

from torchreid.utils import FeatureExtractor


def load_model(CONFIG):

    cnn_arch = CONFIG['CNN_MODEL']['arch']
    # model = MOTMPNet(self.hparams['graph_model_params']).cuda()

    if cnn_arch == 'resnet50':
        # load resnet and trained REID weights
        cnn_model = resnet50_fc256(10, loss='xent', pretrained=True).cuda()
        load_pretrained_weights(cnn_model, CONFIG['CNN_MODEL']['model_weights_path'][cnn_arch])
        cnn_model.eval()

        # cnn_model.return_embeddings = True
    elif cnn_arch == 'bdnet_market':
        cnn_model = bdnet(num_classes=751,  loss='softmax',  pretrained=True,  use_gpu= True, feature_extractor = True  )
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
    # frames_batches = [item[1] for item in batch]
    # ids_batches = [item[2] for item in batch]
    # ids_cam_batches  = [item[3] for item in batch]
    # bboxes, frames, ids, ids_cam

    return [bboxes_batches, df_batches]#[bboxes_batches, frames_batches, ids_batches, ids_cam_batches]



global USE_CUDA, CONFIG

USE_CUDA = torch.cuda.is_available()

date = date = str(time.localtime().tm_year) + '-' + str(time.localtime().tm_mon).zfill(2) + '-' + str(
    time.localtime().tm_mday).zfill(2) + \
       ' ' + str(time.localtime().tm_hour).zfill(2) + ':' + str(time.localtime().tm_min).zfill(2) + ':' + str(
    time.localtime().tm_sec).zfill(2)



parser = argparse.ArgumentParser(description='Training GNN for cross-camera association')
parser.add_argument('--ConfigPath', metavar='DIR', help='Configuration file path')


# Decode CONFIG file information
args = parser.parse_args()
CONFIG = yaml.safe_load(open(args.ConfigPath, 'r'))
results_path = os.path.join(os.getcwd(), 'results', str(CONFIG['ID']) + date)

os.mkdir(results_path)
os.mkdir(os.path.join(results_path, 'images'))
os.mkdir(os.path.join(results_path, 'files'))


cnn_model = load_model(CONFIG)


dataset_dir = os.path.join(CONFIG['DATASET_TRAIN']['ROOT'],CONFIG['DATASET_TRAIN']['NAME'])
subset_train_dir = os.path.join(CONFIG['DATASET_TRAIN']['ROOT'],CONFIG['DATASET_TRAIN']['NAME'])
subset_val_dir = os.path.join(CONFIG['DATASET_VAL']['ROOT'],CONFIG['DATASET_VAL']['NAME'])


train_set = torchvision.datasets.ImageFolder(subset_train_dir)
val_set = torchvision.datasets.ImageFolder(subset_val_dir)

train_dataset = datasets.EPFL_dataset(train_set, 'train', CONFIG, cnn_model)
val_dataset = datasets.EPFL_dataset(val_set, 'validation', CONFIG, cnn_model)

# print("SHUFFLE FALSE")
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=CONFIG['TRAINING']['BATCH_SIZE']['TRAIN'], shuffle=True,
                                       num_workers=CONFIG['DATALOADER']['NUM_WORKERS'], collate_fn=my_collate,pin_memory=CONFIG['DATALOADER']['PIN_MEMORY'])

validation_loader = torch.utils.data.DataLoader(val_dataset, batch_size=CONFIG['TRAINING']['BATCH_SIZE']['VAL'], shuffle=False,
                                       num_workers=CONFIG['DATALOADER']['NUM_WORKERS'], collate_fn=my_collate,pin_memory=CONFIG['DATALOADER']['PIN_MEMORY'])
print('suffle')
if CONFIG['MODE'] == 'GNN':
    #LOAD MPN NETWORK#

    mpn_model = load_model_mpn(CONFIG)
    mpn_model.cuda()
    num_params_mpn  = sum([np.prod(p.size()) for p in mpn_model.parameters()])

    ## LOSS AND OPTIMIZER

    # optim_class = CONFIG['TRAINING']['OPTIMIZER']['type']
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, mpn_model.parameters()), lr=CONFIG['TRAINING']['OPTIMIZER']['args']['lr'])

    # avg per epoch
    training_loss_avg = []
    training_precision_1_avg = []
    training_precision_0_avg = []

    val_loss_avg = []
    val_precision_1_avg = []
    val_precision_0_avg = []
    val_prec_in_history = []

    # all, per iteration
    train_loss_in_history = []
    train_prec0_in_history = []
    train_prec1_in_history = []
    train_prec_in_history = []

    val_loss_in_history = []
    val_prec0_in_history = []
    val_prec1_in_history = []


    ## TRAINING
    best_prec = 0

    for epoch in range(0, CONFIG['TRAINING']['EPOCHS']):
        epoch_start = time.time()

        train_losses,train_precision_1, train_precision_0,train_loss_in_history, train_prec1_in_history,train_prec0_in_history,train_prec_in_history = \
            train(CONFIG, train_loader, cnn_model, mpn_model, epoch, optimizer,results_path,train_loss_in_history,train_prec1_in_history,train_prec0_in_history,train_prec_in_history,train_dataset,dataset_dir)

        training_loss_avg.append(train_losses.avg)
        training_precision_1_avg.append(train_precision_1.avg)
        training_precision_0_avg.append(train_precision_0.avg)

        val_losses, val_precision_1, val_precision_0,val_loss_in_history,val_prec1_in_history,val_prec0_in_history,val_prec_in_history = \
           validate(CONFIG,validation_loader, cnn_model, mpn_model, results_path,epoch,val_loss_in_history,val_prec1_in_history,val_prec0_in_history,val_prec_in_history,val_dataset,dataset_dir )

        val_loss_avg.append(val_losses.avg)
        val_precision_1_avg.append(val_precision_1.avg)
        val_precision_0_avg.append(val_precision_0.avg)

        plt.figure()
        plt.plot(training_precision_1_avg, label='Training Prec class 1')
        plt.plot(training_precision_0_avg, label='Training Prec class 0')
        plt.plot(val_precision_1_avg, label='Validation Prec class 1')
        plt.plot(val_precision_0_avg, label='Validation Prec class 0')
        plt.plot((np.asarray(training_precision_1_avg) + np.asarray(training_precision_0_avg)) / 2, '--',   label='Training MCA')
        plt.plot((np.asarray(val_precision_1_avg) + np.asarray(val_precision_0_avg)) / 2, '--', label='Validation MCA')
        plt.ylabel('Precision'), plt.xlabel('Epoch')
        plt.legend()
        plt.savefig(results_path + '/images/Precision Per Epoch.pdf', bbox_inches='tight')
        plt.close()
        plt.figure()
        plt.plot(training_loss_avg, label='Training loss')
        plt.plot(val_loss_avg,  label='Validation loss')
        plt.ylabel('Loss'), plt.xlabel('Epoch')
        plt.legend()
        plt.savefig(results_path + '/images/Loss per Epoch.pdf', bbox_inches='tight')
        plt.close()


        is_best = (val_precision_1.avg + val_precision_0.avg)/2 > best_prec
        best_prec = max((val_precision_1.avg + val_precision_0.avg)/2, best_prec)
        # SEGMENTATION
        utils.save_checkpoint({
            'epoch': epoch + 1,
            'model_state_dict': mpn_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_prec': best_prec,
            'model_parameters': num_params_mpn,

            'CONFIG': CONFIG
        }, is_best, results_path, CONFIG['ID'])


        print('Elapsed time for epoch {}: {time:.3f} minutes'.format(epoch, time=(time.time() - epoch_start) / 60))

elif CONFIG['MODE'] == 'REID':
    val_prec0_in_history = []
    val_prec1_in_history = []

    reid_dists, labels, reid_distances_cos = validate_REID(validation_loader, cnn_model,CONFIG)
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
    reid_distances_norm = reid_dists / np.max(reid_dists)
    print('Max distance = '+  str(np.max(reid_dists)))
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

        TP, FP, TN, FN, P,R, F = compute_P_R_F(preds, np.asarray(labels))

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

        TP, FP, TN, FN, P, R, F = compute_P_R_F(preds, np.asarray(labels))

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
    mpn_model = load_model_mpn(CONFIG, CONFIG['PRETRAINED_GNN_MODEL'])
    mpn_model.cuda()
    mpn_model.eval()
    val_loss_in_history = []
    val_prec0_in_history = []
    val_prec1_in_history = []
    val_prec_in_history = []

    epoch = 0


    P_list, R_list, F_list, TP_list, FP_list, FN_list, TN_list, rand_index, rand_index_rounding, P_r_list, R_r_list, F_r_list, TP_r_list, FP_r_list, FN_r_list, TN_r_list,\
        mutual_index, homogeneity, completeness, v_measure = validate_GNN_cross_camera_association(CONFIG, validation_loader, cnn_model, mpn_model)

    a=1
    P = np.mean(np.asarray(P_list))
    R = np.mean(np.asarray(R_list))
    F = np.mean(np.asarray(F_list))
    TP = np.sum(np.asarray(TP_list))
    FP = np.sum(np.asarray(FP_list))
    FN = np.sum(np.asarray(FN_list))
    TN = np.sum(np.asarray(TN_list))
    RI = np.mean(np.asarray(rand_index))
    RI_r = np.mean(np.asarray(rand_index_rounding))
    MI = np.mean(np.asarray(mutual_index))
    hom = np.mean(np.asarray(homogeneity))
    com = np.mean(np.asarray(completeness))
    v = np.mean(np.asarray(v_measure))


    print('P= '+ str(P))
    print('R= '+ str(R))
    print('F= '+ str(F))
    print('TP= ' + str(TP))
    print('FP= ' + str(FP))
    print('FN= ' + str(FN))
    print('TN= '+ str(TN))
    print('Rand index mean = ' + str(RI))
    print('Rand index rounding mean = ' + str(RI_r))
    print( 'Mutual index mean = ' + str(MI) )
    print( 'homogeneity mean = ' + str(hom) )
    print( 'completeness mean = ' + str(com) )
    print( 'v_measure mean = ' + str(v) )

elif CONFIG['MODE'] == 'eval_RANK':

    validation_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1,
                                                    shuffle=False,
                                                    num_workers=CONFIG['DATALOADER']['NUM_WORKERS'],
                                                    collate_fn=my_collate,
                                                    pin_memory=CONFIG['DATALOADER']['PIN_MEMORY'])

    P_list, R_list, F_list, TP_list, FP_list, FN_list, TN_list, r_index = eval_RANK(validation_loader, cnn_model, CONFIG)

    P = np.mean(np.asarray(P_list))
    R = np.mean(np.asarray(R_list))
    F = np.mean(np.asarray(F_list))
    TP = np.sum(np.asarray(TP_list))
    FP = np.sum(np.asarray(FP_list))
    FN = np.sum(np.asarray(FN_list))
    TN = np.sum(np.asarray(TN_list))
    RI = np.mean(np.asarray(r_index))


    print('P= '+ str(P))
    print('R= '+ str(R))
    print('F= '+ str(F))
    print('TP= ' + str(TP))
    print('FP= ' + str(FP))
    print('FN= ' + str(FN))
    print('TN= '+ str(TN))
    print('Rand index mean = ' + str(RI))

elif CONFIG['MODE'] == 'REID_th':

    validation_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1,
                                                    shuffle=False,
                                                    num_workers=CONFIG['DATALOADER']['NUM_WORKERS'],
                                                    collate_fn=my_collate,
                                                    pin_memory=CONFIG['DATALOADER']['PIN_MEMORY'])

    val_prec0_in_history = []
    val_prec1_in_history = []
    th_LAB = 0.55 # con Resnet50 all
    max_LAB = 33.79 #conresnet50 all
    # th_LAB = 0.75 # cosine distance resnet50 all

    P_list, R_list, F_list, TP_list, FP_list, FN_list, TN_list, r_index, m_index,homogeneity, completeness, v_measure = validate_REID_with_th(validation_loader, cnn_model, th_LAB, max_LAB,'L2')
    P = np.mean(np.asarray(P_list))
    R = np.mean(np.asarray(R_list))
    F = np.mean(np.asarray(F_list))
    TP = np.sum(np.asarray(TP_list))
    FP = np.sum(np.asarray(FP_list))
    FN = np.sum(np.asarray(FN_list))
    TN = np.sum(np.asarray(TN_list))
    RI = np.mean(np.asarray(r_index))
    MI = np.mean(np.asarray(m_index))
    hom = np.mean(np.asarray(homogeneity))
    com = np.mean(np.asarray(completeness))
    v = np.mean(np.asarray(v_measure))


    print('P= ' + str(P))
    print('R= ' + str(R))
    print('F= ' + str(F))
    print('TP= ' + str(TP))
    print('FP= ' + str(FP))
    print('FN= ' + str(FN))
    print('TN= ' + str(TN))
    print( 'Rand index mean = ' + str(RI) )
    print( 'Mutual index mean = ' + str(MI) )
    print( 'homogeneity mean = ' + str(hom) )
    print( 'completeness mean = ' + str(com) )
    print( 'v_measure mean = ' + str(v) )

elif CONFIG['MODE'] == 'geometrical_association':

    P_list, R_list, F_list, TP_list, FP_list, FN_list, TN_list, rand_index, rand_index_rounding, P_r_list, R_r_list, F_r_list, TP_r_list, FP_r_list, FN_r_list, TN_r_list, \
    mutual_index, homogeneity, completeness, v_measure,  fowlkes_index = geometrical_association(CONFIG, validation_loader)

    P = np.mean(np.asarray(P_list))
    R = np.mean(np.asarray(R_list))
    F = np.mean(np.asarray(F_list))
    TP = np.sum(np.asarray(TP_list))
    FP = np.sum(np.asarray(FP_list))
    FN = np.sum(np.asarray(FN_list))
    TN = np.sum(np.asarray(TN_list))
    RI = np.mean(np.asarray(rand_index))
    RI_r = np.mean(np.asarray(rand_index_rounding))
    MI = np.mean(np.asarray(mutual_index))
    hom = np.mean(np.asarray(homogeneity))
    com = np.mean(np.asarray(completeness))
    v = np.mean(np.asarray(v_measure))
    fowl = np.mean(np.asarray( fowlkes_index))
    # sil = np.mean(np.asarray( silhouette_index))


    print('P= '+ str(P))
    print('R= '+ str(R))
    print('F= '+ str(F))
    print('TP= ' + str(TP))
    print('FP= ' + str(FP))
    print('FN= ' + str(FN))
    print('TN= '+ str(TN))
    print('Rand index mean = ' + str(RI))
    print('Rand index rounding mean = ' + str(RI_r))
    print( 'Mutual index mean = ' + str(MI) )
    print( 'homogeneity mean = ' + str(hom) )
    print( 'completeness mean = ' + str(com) )
    print( 'v_measure mean = ' + str(v) )
    print('fowlkes mean = ' + str(fowl))
    # print('silhouette mean = ' + str(sil))
