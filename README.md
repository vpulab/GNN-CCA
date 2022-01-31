## Graph Neural Networks for Cross-Camera Data Association



# Files and folders
* **./config** : for configuration .yaml files
* **./datasets** : class for dataset transformations
* **./libs** : 3rd party libraries
* **./misc** : misc and utils scripts
* **./models** : for different network architectures definitions
*  **./scripts** : sh script to run several runnings with different config files
* **main_training.py** : main file to run when training
* **main.py** : main file to run when inference
* **train.py** : train class
* **inference.py** : inference class
* **utils.py** : different tools/utilities classes
* **env_gnn.yml**: anaconda environment dependencies


# Setup
**Requirements**

The repository has been tested in the following software.
* Ubuntu 16.04
* Python 3.6
* Anaconda
* Pycharm

**Anaconda environment**

To create and setup the Anaconda Envirmorent run the following terminal command from the repository folder:
```
$ conda env create -f env_gnn.yml
$ conda activate env_gnn
```

**Clone repository**

```
$ git clone https://github.com/elun15/GNN-Cross-Camera-Association.git
```


# EPFL Dataset
This repo is evaluated on EPFL Terrace, Laboratory and Basketball sequence.

EPFL videos can be downloaded at https://www.epfl.ch/labs/cvlab/data/data-pom-index-php/

EPFL GT can be found at https://bitbucket.org/merayxu/multiview-object-tracking-dataset/src/master/

# Citation

If you find this code and work useful, please consider citing:




