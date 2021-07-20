import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import random
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import os
import pandas as pd
from skimage.io import imread
from skimage.transform import ProjectiveTransform,warp
import cv2
import utils

from imgaug import augmenters as iaa
from torchreid.data.transforms import build_transforms

COL_NAMES_EPFL = ('id', 'xmin', 'ymin', 'xmax', 'ymax', 'frame', 'lost', 'occluded', 'generated','label')


# noinspection PyTypeChecker
class EPFL_dataset(Dataset):

    def __init__(self, name, mode, CONFIG, cnn_model):

        # self.imageFolderDataset = imageFolderDataset
        self.mode = mode
        self.cnn_model = cnn_model

        if CONFIG['DATASET_TRAIN']['IMAUG'] is False:
            transforms_list = ['random_flip']
        else:
            transforms_list = ['random_flip', 'color_jitter', 'random_erase']

        self.transform_tr, self.transform_te = build_transforms(
            CONFIG['DATASET_TRAIN']['RESIZE'][CONFIG['CNN_MODEL']['arch']][0],
            CONFIG['DATASET_TRAIN']['RESIZE'][CONFIG['CNN_MODEL']['arch']][1], transforms=transforms_list,
            norm_mean=CONFIG['DATASET_TRAIN']['MEAN'], norm_std=CONFIG['DATASET_TRAIN']['STD'])

        if mode == 'train':

            self.transform = self.transform_tr
            self.path = os.path.join(CONFIG['DATASET_TRAIN']['ROOT'],name)
            self.max_dist = CONFIG['CONV_TO_M'][name]


            self.cameras = os.listdir(os.path.join(CONFIG['DATASET_TRAIN']['ROOT'],name))
            self.cameras = [item for item in self.cameras if item[0] is not '.']
            self.cameras.sort()
            self.num_cameras = len(self.cameras)

            self.data_det = pd.DataFrame()
            self.list_corners_x = list()
            self.list_corners_y = list()

            x_corners = np.array([0, 360, 360, 0])
            y_corners = np.array([100, 100, 288, 288])
            # x_corners = np.array([360, 360, 0,0])
            # y_corners = np.array([50, 288, 288,50])
            for c in self.cameras:
                seq_path = os.path.join(os.path.join(CONFIG['DATASET_TRAIN']['ROOT'],name,c))
                detections_file_path = os.path.join(seq_path, 'gt', 'gt.txt')
                det_df = pd.read_csv(detections_file_path, header=None, sep=" ")

                # Number and order of columns is always assumed to be the same
                det_df = det_df[det_df.columns[:len(COL_NAMES_EPFL)]]
                det_df.columns = COL_NAMES_EPFL
                det_df = det_df[det_df['lost'] == 0]
                det_df['xmin'] = det_df['xmin'].values + 1
                det_df['width'] = (det_df['xmax'] - det_df['xmin']).values
                det_df['height'] = (det_df['ymax'] - det_df['ymin']).values
                det_df['id_cam'] = int(c[-1:])


                # Get ground-plane coordinates

                bb_mid_low_x = det_df['xmin'].values + np.round(det_df['width'].values / 2)
                bb_mid_low_y = det_df['ymax'].values

                homog_file = os.path.join(seq_path,'Homography.txt')
                H = np.asarray(pd.read_csv(homog_file, header=None, sep="\t"))

                xw, yw = utils.apply_homography_image_to_world(bb_mid_low_x, bb_mid_low_y, H)

                det_df['xw'] = xw
                det_df['yw'] = yw

                xw_corners, yw_corners = utils.apply_homography_image_to_world(x_corners, y_corners, H)
                corners = np.concatenate((np.expand_dims(x_corners,axis=1),np.expand_dims(y_corners,axis= 1)), axis=1)
                self.outputCorners = cv2.perspectiveTransform(corners[None,:,:].astype(float), H)

                self.list_corners_x.append(xw_corners)
                self.list_corners_y.append(yw_corners)

                self.data_det = self.data_det.append(det_df)

            # We only consider for training frames with at least 2 nodes
            frames = np.arange(np.max(self.data_det['frame'].values)+1)
            nodes_per_frame = np.bincount(self.data_det['frame'].values)
            self.frames_valid = frames[np.where(nodes_per_frame > 1)]

        else: #VALIDATION
            self.path = os.path.join(CONFIG['DATASET_VAL']['ROOT'], CONFIG['DATASET_VAL']['NAME'])

            self.max_dist = CONFIG['CONV_TO_M'][CONFIG['DATASET_VAL']['NAME']]

            self.transform = self.transform_te
            if CONFIG['MODE'] == 'TOP_DB_eval':
                self.transform = transforms.Compose([
                    transforms.Resize(CONFIG['TOP_DB_RESIZE']),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=CONFIG['DATASET_VAL']['MEAN'],
                                         std=CONFIG['DATASET_VAL']['STD'])])


            self.cameras = os.listdir(os.path.join(CONFIG['DATASET_VAL']['ROOT'], CONFIG['DATASET_VAL']['NAME']))
            self.cameras = [item for item in self.cameras if item[0] is not '.']
            self.cameras.sort()
            self.num_cameras = len(self.cameras)

            self.data_det = pd.DataFrame()
            for c in self.cameras:
                seq_path = os.path.join(os.path.join(CONFIG['DATASET_VAL']['ROOT'],CONFIG['DATASET_VAL']['NAME'],c))
                detections_file_path = os.path.join(seq_path, 'gt', 'gt.txt')
                det_df = pd.read_csv(detections_file_path, header=None, sep=" ")

                # Number and order of columns is always assumed to be the same
                det_df = det_df[det_df.columns[:len(COL_NAMES_EPFL)]]
                det_df.columns = COL_NAMES_EPFL
                det_df = det_df[det_df['lost'] == 0]
                det_df['xmin'] = det_df['xmin'].values + 1
                det_df['width'] = (det_df['xmax'] - det_df['xmin']).values
                det_df['height'] = (det_df['ymax'] - det_df['ymin']).values
                det_df['id_cam'] = int(c[-1:])


                # Get ground-plane coordinates

                bb_mid_low_x = det_df['xmin'].values + np.round(det_df['width'].values / 2)
                bb_mid_low_y = det_df['ymax'].values

                homog_file = os.path.join(seq_path,'Homography.txt')
                H = np.asarray(pd.read_csv(homog_file, header=None, sep="\t"))

                xw, yw = utils.apply_homography_image_to_world(bb_mid_low_x, bb_mid_low_y, H)

                det_df['xw'] = xw
                det_df['yw'] = yw

                # xw_corners, yw_corners = utils.apply_homography_image_to_world(x_corners, y_corners, H)
                # corners = np.concatenate((np.expand_dims(x_corners,axis=1),np.expand_dims(y_corners,axis= 1)), axis=1)
                # outputCorners = cv2.perspectiveTransform(corners[None,:,:].astype(float), H)

                # self.list_corners_x.append(xw_corners)
                # self.list_corners_y.append(yw_corners)

                self.data_det = self.data_det.append(det_df)

            # We only consider for training frames with at least 2 nodes
            frames = np.arange(np.max(self.data_det['frame'].values+1))
            nodes_per_frame = np.bincount(self.data_det['frame'].values)
            self.frames_valid = frames[np.where(nodes_per_frame > 1)]



    def __getitem__(self, index):
        # print('index = ' + str(index))
        # print('dataset = ' + str(self.path))
        #
        # if index>= len(self.frames_valid):
        #     a = 1

        frame = self.frames_valid[index]
        data_f = self.data_det[self.data_det['frame'] == frame]
        cams = np.unique(data_f['id_cam'].values)

        #CNN model to extract reid embeddings

        # Get a list with the frames images of the cameras
        img_f = list()
        for i in range(len(self.cameras)):
            if i in cams:
                c = self.cameras[i]
                frame_path = os.path.join(self.path, c, 'img1', str(frame).zfill(6) + '.jpg')
                img = imread(frame_path)
                img_f.append(img)

            else:
                img_f.append([])

        bboxes = []
        frames = []
        ids = []
        ids_cam = []

        for i in range(len(data_f)):
            det = data_f.iloc[i]
            bb_img =img_f[det['id_cam']][int(max(0, det['ymin'])): int(max(0, det['ymax'])),
                     int(max(0, det['xmin'])): int(max(0, det['xmax']))]

            bb_img = Image.fromarray(bb_img)
            bb_img = self.transform(bb_img)


            bboxes.append(bb_img)
            frames.append(self.path)
            # ids.append(det['id'])
            # ids_cam.append(det['id_cam'])



        bboxes = torch.stack(bboxes, dim = 0)




        return [bboxes, data_f,self.max_dist]#[bboxes, frames, ids, ids_cam]




    def __len__(self):
        return len(self.frames_valid)



#
# plt.figure()
# plt.plot(np.bincount(self.data_det['frame'].values))
# plt.xlabel('Frame')
# plt.ylabel('Nodes')
# plt.title('EPFL-Terrace #Nodes per Frame. Avg: 18.8' )