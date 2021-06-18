
import os
import time
import shutil
import yaml

import matplotlib.pyplot as plt
import numpy as np

import torch
import argparse
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


from models.resnet import resnet50_fc256, load_pretrained_weights






global USE_CUDA, CONFIG

USE_CUDA = torch.cuda.is_available()

parser = argparse.ArgumentParser(description='Training GNN for cross-camera association')
parser.add_argument('--ConfigPath', metavar='DIR', help='Configuration file path')

# Decode CONFIG file information
args = parser.parse_args()
CONFIG = yaml.safe_load(open(args.ConfigPath, 'r'))


cnn_model = load_model(CONFIG)

dataset_dir = os.path.join(CONFIG['DATASET_TRAIN']['ROOT'],CONFIG['DATASET_TRAIN']['NAME'])

