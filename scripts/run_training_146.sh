# This file should be executed from terminal in Folder 'Scripts' with the following command:
# source Run_Tests.sh

#!/bin/bash
#conda init
conda activate env_gnn

cd ..
chmod +x main_training.py
#python main_training.py --ConfigPath "./config/config_training_146.yaml"
python main_training.py --ConfigPath "./config/config_training_152_9.yaml"


#

