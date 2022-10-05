

## GNN for CCA

IEEE TCSVT Paper:  https://ieeexplore.ieee.org/document/9893862


# Setup & Running
**Requirements**

The repository has been tested in the following software.
* Ubuntu 16.04
* Python 3.6
* Anaconda
* Pycharm

**Clone repository**

```
git clone https://github.com/elun15/GNN-Cross-Camera-Association.git
```

**Anaconda environment**

To create and setup the Anaconda Envirmorent run the following terminal command from the repository folder:
```
conda env create -f env_gnn.yml
conda activate env_gnn
```


**Download and prepare EPFL dataset**
This repo is evaluated on EPFL Terrace, Laboratory and Basketball sequence.

 1. Download the EPFL video sequences at  [https://www.epfl.ch/labs/cvlab/data/data-pom-index-php/](https://www.epfl.ch/labs/cvlab/data/data-pom-index-php/)
 2. Place each .avi sequence in their corresponding path, e.g. *./datasets/EPFL-Terrace/terrace1-c0/terrace1-c0.avi*
 3. Run *.libs/preprocess_EPFL.py* in order to extract frame images. 
 4. The EPFL GT, that we already provide,  can be found at [https://bitbucket.org/merayxu/multiview-object-tracking-dataset/src/master/](https://bitbucket.org/merayxu/multiview-object-tracking-dataset/src/master/). 


**Download pre-trained REID models**

 5. Download the pre-trained REID models from https://1drv.ms/u/s!AufOLvb5OB5fhx0os9hCDdkFfT6l?e=roljmV  , unzip the 4 folders and place them under *./trained_models/*

**Download  a pre-trained GNN-CCA model**
We provide the weights of the GNN trained on the S1 set (see paper for detailes).
 

 6.  Download the pre-trained weights from https://1drv.ms/u/s!AufOLvb5OB5fhx7O9KIJDqKLj8Uu?e=hbyR7T and place the folder *GNN_S1_Resnet50MCD_SGD0005_cosine20_BS64_BCE_all_step_BNcls_L4_2021-11-10 19:01:49* under *./results/* folder.

**Inference Running**

To inference the previous model run:
```
python main.py --ConfigPath "config/config_inference.yaml"
```

**Training**

For training run:
```
python main_training.py --ConfigPath "config/config_training.yaml"
```



# Citation

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




