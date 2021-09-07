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
from torch.utils.data import DataLoader,Dataset, ConcatDataset
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
    max_dist = [item[2] for item in batch]
    # frames_batches = [item[1] for item in batch]
    # ids_batches = [item[2] for item in batch]
    # ids_cam_batches  = [item[3] for item in batch]
    # bboxes, frames, ids, ids_cam

    return [bboxes_batches, df_batches,max_dist] #, path]#[bboxes_batches, frames_batches, ids_batches, ids_cam_batches]



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
if CONFIG['TRAINING']['ONLY_DIST'] or CONFIG['TRAINING']['ONLY_APPEARANCE']:
    CONFIG['GRAPH_NET_PARAMS']['encoder_feats_dict']['edges']['edge_in_dim'] = 2
    CONFIG['GRAPH_NET_PARAMS']['encoder_feats_dict']['edges']['edge_out_dim'] = 4
    CONFIG['GRAPH_NET_PARAMS']['edge_model_feats_dict']['fc_dims'] = [4]
    CONFIG['GRAPH_NET_PARAMS']['classifier_feats_dict']['edge_in_dim'] = 4
    CONFIG['GRAPH_NET_PARAMS']['classifier_feats_dict']['edge_fc_dims'] = [2]


results_path = os.path.join(os.getcwd(), 'results', str(CONFIG['ID']) + date)
os.mkdir(results_path)
os.mkdir(os.path.join(results_path, 'images'))
os.mkdir(os.path.join(results_path, 'files'))

with open(os.path.join(results_path, 'files', 'config.yaml'), 'w') as file:
    yaml.safe_dump(CONFIG, file)

shutil.copyfile('train.py', os.path.join(results_path, 'train.py'))
shutil.copyfile('main_training.py', os.path.join(results_path, 'main_training.py'))

cnn_model = load_model(CONFIG)

train_datasets = []

for d in CONFIG['DATASET_TRAIN']['NAME']:

    dataset_dir = os.path.join(CONFIG['DATASET_TRAIN']['ROOT'],d)
    # subset_train_dir = os.path.join(CONFIG['DATASET_TRAIN']['ROOT'],d)
    subset_val_dir = os.path.join(CONFIG['DATASET_VAL']['ROOT'],d)
    # train_set = torchvision.datasets.ImageFolder(subset_train_dir)
    # val_set = torchvision.datasets.ImageFolder(subset_val_dir)
    train_datasets.append(datasets.EPFL_dataset(d, 'train', CONFIG, cnn_model))



if len(train_datasets) > 1:
    train_dataset = torch.utils.data.ConcatDataset(train_datasets)

    total_samples = np.sum(np.asarray(train_dataset.cumulative_sizes))
    weights = list([])
    for t in train_datasets:
        weights.append(np.ones(len(t)) * (1 / (len(t))))
    weights = torch.from_numpy(np.asarray(np.concatenate(weights)))
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights),
                                                             replacement=False)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=CONFIG['TRAINING']['BATCH_SIZE']['TRAIN'],
                                               sampler=sampler,
                                               num_workers=CONFIG['DATALOADER']['NUM_WORKERS'],collate_fn=my_collate,
                                               pin_memory=CONFIG['DATALOADER']['PIN_MEMORY'])



else:
    train_dataset = train_datasets[0]
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=CONFIG['TRAINING']['BATCH_SIZE']['TRAIN'],
                                               shuffle=True,
                                               num_workers=CONFIG['DATALOADER']['NUM_WORKERS'], collate_fn=my_collate,
                                               pin_memory=CONFIG['DATALOADER']['PIN_MEMORY'])

# print("SHUFFLE FALSE")

val_dataset = datasets.EPFL_dataset([], 'validation', CONFIG, cnn_model)
validation_loader = torch.utils.data.DataLoader(val_dataset, batch_size=CONFIG['TRAINING']['BATCH_SIZE']['VAL'], shuffle=False,
                                               num_workers=CONFIG['DATALOADER']['NUM_WORKERS'], collate_fn=my_collate,pin_memory=CONFIG['DATALOADER']['PIN_MEMORY'])
#LOAD MPN NETWORK#

mpn_model = load_model_mpn(CONFIG,CONFIG['PRETRAINED_GNN_MODEL'])
mpn_model.cuda()
num_params_mpn  = sum([np.prod(p.size()) for p in mpn_model.parameters()])

## LOSS AND OPTIMIZER



if CONFIG['TRAINING']['OPTIMIZER']['type']  == 'Adam':
    if CONFIG['TRAINING']['WARMUP']['ENABLE']:
        lr_warmup_list = np.linspace(CONFIG['TRAINING']['WARMUP']['LR'], CONFIG['TRAINING']['OPTIMIZER']['args']['lr'],
                                     CONFIG['TRAINING']['WARMUP']['NUM_EPOCHS'] + 1, endpoint=False)
        lr_warmup_list = lr_warmup_list[1:]

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, mpn_model.parameters()),
                                    lr_warmup_list[0])
        flag_warmup_ended = False
    else:
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, mpn_model.parameters()), lr=CONFIG['TRAINING']['OPTIMIZER']['args']['lr'])

elif CONFIG['TRAINING']['OPTIMIZER']['type']  == 'SGD':
    if CONFIG['TRAINING']['WARMUP']['ENABLE']:
        lr_warmup_list = np.linspace(CONFIG['TRAINING']['WARMUP']['LR'], CONFIG['TRAINING']['OPTIMIZER']['args']['lr'],
                                     CONFIG['TRAINING']['WARMUP']['NUM_EPOCHS'] + 1, endpoint=False)
        lr_warmup_list = lr_warmup_list[1:]

        optimizer = torch.optim.SGD(mpn_model.parameters(),
                                    lr= lr_warmup_list[0],
                                    momentum=CONFIG['TRAINING']['OPTIMIZER']['args']['momentum'],
                                    weight_decay=CONFIG['TRAINING']['OPTIMIZER']['args']['weight_decay'])
        flag_warmup_ended = False
    else:
        optimizer = torch.optim.SGD(mpn_model.parameters(),
                                    lr =CONFIG['TRAINING']['OPTIMIZER']['args']['lr'],
                                    momentum=CONFIG['TRAINING']['OPTIMIZER']['args']['momentum'],
                                    weight_decay=CONFIG['TRAINING']['OPTIMIZER']['args']['weight_decay'])

    # Learning rate decay
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = CONFIG['TRAINING']['LR_SCHEDULER']['args']['step_size'],
                                                gamma = CONFIG['TRAINING']['LR_SCHEDULER']['args']['gamma'])


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
best_val_loss = 1000
list_lr = list([])

for epoch in range(0, CONFIG['TRAINING']['EPOCHS']):
    epoch_start = time.time()
    list_lr.append(optimizer.param_groups[0]['lr'])

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



    if CONFIG['TRAINING']['WARMUP']['ENABLE'] and not (flag_warmup_ended):
        if epoch == CONFIG['TRAINING']['WARMUP']['NUM_EPOCHS']:
            flag_warmup_ended = True

        if (flag_warmup_ended):
            # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, mpn_model.parameters()), lr=CONFIG['TRAINING']['OPTIMIZER']['args']['lr'])
            optimizer = torch.optim.SGD(mpn_model.parameters(),
                                        lr=CONFIG['TRAINING']['OPTIMIZER']['args']['lr'],
                                        momentum=CONFIG['TRAINING']['OPTIMIZER']['args']['momentum'],
                                        weight_decay=CONFIG['TRAINING']['OPTIMIZER']['args']['weight_decay'])
            # Learning rate decay SGD
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=CONFIG['TRAINING']['LR_SCHEDULER']['args'][ 'step_size'], gamma=CONFIG['TRAINING']['LR_SCHEDULER']['args']['gamma'])
        else:
            optimizer.param_groups[0]['lr'] = lr_warmup_list[epoch]

    else:
        if CONFIG['TRAINING']['OPTIMIZER']['type'] == 'SGD':
            scheduler.step()

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


    plt.figure()
    plt.plot(list_lr, 'r', label='Training')
    plt.ylabel('Learning Rate'), plt.xlabel('Epoch')
    plt.legend()
    plt.savefig(results_path + '/images/LR per Epoch.pdf', bbox_inches='tight')
    plt.close()



    # is_best = (val_precision_1.avg + val_precision_0.avg)/2 > best_prec
    is_best = (val_loss_avg[-1]) < best_val_loss
    # best_prec = max((val_precision_1.avg + val_precision_0.avg)/2, best_prec)
    best_val_loss = min(val_loss_avg[-1],best_val_loss)

    # SEGMENTATION
    utils.save_checkpoint({
        'epoch': epoch + 1,
        'model_state_dict': mpn_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'prec': (val_precision_1.avg + val_precision_0.avg)/2,
        'prec1': val_precision_1_avg,
        'prec0': val_precision_0_avg,
        'best_loss':  best_val_loss,
        'model_parameters': num_params_mpn,

        'CONFIG': CONFIG
    }, is_best, results_path, CONFIG['ID'])


    print('Elapsed time for epoch {}: {time:.3f} minutes'.format(epoch, time=(time.time() - epoch_start) / 60))

