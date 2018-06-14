#!/bin/bash
export PATH=/home/siddiqui/anaconda3/bin/:$PATH

# sudo userdocker run -it -v /netscratch:/netscratch dlcc/tensorflow_opencv /netscratch/siddiqui/Repositories/FCN-TensorFlow/train.sh
cd /netscratch/siddiqui/Repositories/FCN-TensorFlow/
python trainer_fcn.py --modelName NASNet 