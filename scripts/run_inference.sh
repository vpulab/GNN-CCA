# This file should be executed from terminal in Folder 'Scripts' with the following command:
# source Run_Tests.sh

#!/bin/bash
#conda init
conda activate env_gnn

cd ..
chmod +x main.py

python main.py --ConfigPath "./config/config_inference.yaml"
python main.py --ConfigPath "./config/config_inference2.yaml"
python main.py --ConfigPath "./config/config_inference3.yaml"
python main.py --ConfigPath "./config/config_inference4.yaml"
python main.py --ConfigPath "./config/config_inference5.yaml"
python main.py --ConfigPath "./config/config_inference6.yaml"
python main.py --ConfigPath "./config/config_inference7.yaml"
python main.py --ConfigPath "./config/config_inference8.yaml"