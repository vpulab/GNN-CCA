# This file should be executed from terminal in Folder 'Scripts' with the following command:
# source Run_Tests.sh

#!/bin/bash
#conda init
conda activate env_gnn


cd ..
chmod +x main_training.py
#python main_training.py --ConfigPath "./config/config_training.yaml"
python main_training.py --ConfigPath "./config/config_training_152.yaml"
python main_training.py --ConfigPath "./config/config_training_152_2.yaml"
python main_training.py --ConfigPath "./config/config_training_152.yaml"
python main_training.py --ConfigPath "./config/config_training_152_2.yaml"
#python main_training.py --ConfigPath "./config/config_training_152_3.yaml"
#python main_training.py --ConfigPath "./config/config_training_152_4.yaml"

#python main_training.py --ConfigPath "./config/config_training_152_5.yaml"
#python main_training.py --ConfigPath "./config/config_training_152_6.yaml"

#python main_training.py --ConfigPath "./config/config_training_152_7.yaml"
#python main_training.py --ConfigPath "./config/config_training_152_8.yaml"


#

