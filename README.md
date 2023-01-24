


# Graph Neural Networks for Cross-Camera Data Association

IEEE TCSVT Paper:  https://ieeexplore.ieee.org/document/9893862

## Setup & Running
**Requirements**

The repository has been tested in the following software.
* Ubuntu 16.04
* Python 3.6
* Anaconda
* Pycharm

**1. Clone repository**

```
git clone https://github.com/elun15/GNN-Cross-Camera-Association.git
```

**2. Anaconda environment**

To create and setup the Anaconda Envirmorent run the following terminal command from the repository folder:
```
conda env create -f env_gnn.yml
conda activate env_gnn
```
**3. Install Torchreid library**


    git clone https://github.com/KaiyangZhou/deep-person-reid.git
    cd deep-person-reid/
    python setup.py develop
    cd ..
        


**4. Download and prepare EPFL dataset**

This repo is evaluated on <u>EPFL Terrace (seq. 1), Laboratory (seq. 6p), and Basketball</u> sequence.

**4a**. To automatically download the sequences run
```
download_dataset.sh
```
or,

 **4b**. To do it by your own download the EPFL video sequences at  [https://www.epfl.ch/labs/cvlab/data/data-pom-index-php/](https://www.epfl.ch/labs/cvlab/data/data-pom-index-php/). Then, place each .avi sequence in their corresponding path, e.g. *./datasets/EPFL-Terrace/terrace1-c0/terrace1-c0.avi* and name each .avi as the name of the folder containing it.
 
**5. Run** 
```
python ./libs/preprocess_EPFL.py
```
 in order to extract frame images. 

**6. Ground-truth** 

 The EPFL GT (we already provide it, no need to download it)  can be found at [https://bitbucket.org/merayxu/multiview-object-tracking-dataset/src/master/](https://bitbucket.org/merayxu/multiview-object-tracking-dataset/src/master/). 


**7. Download pre-trained REID models**

  Download the pre-trained REID models from [here](https://www-vpu.eps.uam.es/publications/gnn_cca/files/trained_models.zip)   , unzip the 4 folders and place them under *./trained_models/*

**8. Download  a pre-trained GNN-CCA model**

We provide the weights of the GNN trained on the S1 set (see paper for detailes).
Download the pre-trained weights from 
[here](http://www-vpu.eps.uam.es/publications/gnn_cca/files/GNN_S1_Resnet50MCD_SGD0005_cosine20_BS64_BCE_all_step_BNcls_L4_2021-11-10%2019_01_49.zip)	 and place the folder *GNN_S1_Resnet50MCD_SGD0005_cosine20_BS64_BCE_all_step_BNcls_L4_2021-11-10 19:01:49* under *./results/* folder.

**9. Inference Running**

To inference the previous model run:
```
python main.py --ConfigPath "config/config_inference.yaml"
```
**10. Training**

For training run:
```
python main_training.py --ConfigPath "config/config_training.yaml"
```


## Citation

If you find this code and work useful, please consider citing:
```
@ARTICLE{9893862,
  author={Luna, Elena and SanMiguel, Juan C. and Martínez, José M. and Carballeira, Pablo},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  title={Graph Neural Networks for Cross-Camera Data Association}, 
  year={2022},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TCSVT.2022.3207223}}
}
```

