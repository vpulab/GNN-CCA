# This file should be executed from terminal in Folder 'Scripts' with the following command:
# source Run_Tests.sh

#!/bin/bash

source activate mot_neural_solver


cd ..
chmod +x main_training.py
python main_training.py --ConfigPath "./config/config_training.yaml"
python main_training.py --ConfigPath "./config/config_training1.yaml"
python main_training.py --ConfigPath "./config/config_training2.yaml"
#python main_training.py --ConfigPath "./config/config_training3.yaml"
#python main_training.py --ConfigPath "./config/config_training4.yaml"
#python main_training.py --ConfigPath "./config/config_training5.yaml"
#python main_training.py --ConfigPath "./config/config_training6.yaml"
#python main_training.py --ConfigPath "./config/config_training7.yaml"
#python main_training.py --ConfigPath "./config/config_training8.yaml"
#python main_training.py --ConfigPath "./config/config_training9.yaml"





