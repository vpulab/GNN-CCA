import os
import os.path as osp
import sys
import numpy as np
from skimage.io import imread

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
GT_COL_NAMES_EPFL = ('id', 'xmin', 'ymin', 'xmax', 'ymax', 'frame', 'lost', 'occluded', 'generated','label')

if __name__ == "__main__":

    gt_path = './gt/gt.txt'
    # gt_path2 = '/home/elg/Repositories/mot_neural_solver_cross_camera_association/data/eval_gt/terrace1-c0/gt/gt.txt'
    gt_data = pd.read_csv(gt_path,sep= " ")

    # Number and order of columns is always assumed to be the same
    gt_data = gt_data[gt_data.columns[:len(GT_COL_NAMES_EPFL)]]
    gt_data.columns = GT_COL_NAMES_EPFL
    gt_data = gt_data[gt_data['lost'] == 0]
    max_frame = np.max(np.unique(gt_data['frame'].values))
    min_frame = np.min(np.unique(gt_data['frame'].values))


    for f in range(30,max_frame):
        gt_f = gt_data[gt_data['frame'] == f]
        img_path = './img1/' + f'{f:06}' + '.jpg'
        img = imread(img_path)
        # Display the image
        plt.imshow(img)
        plt.title('Frame '+ str(f))
        for b in range(0,len(gt_f)):
            w = gt_f.iloc[b]['xmax'] -gt_f.iloc[b]['xmin']
            h =  gt_f.iloc[b]['ymax'] -gt_f.iloc[b]['ymin']
            plt.gca().add_patch(Rectangle((gt_f.iloc[b]['xmin'],gt_f.iloc[b]['ymin']),w,h,linewidth=1,edgecolor='r',facecolor='none'))
            plt.gca().text(gt_f.iloc[b]['xmin'],gt_f.iloc[b]['ymin'],str(gt_f.iloc[b]['id']),color='r',size = 20)


        plt.show(block = False)
        plt.pause(0.01)
        plt.close()