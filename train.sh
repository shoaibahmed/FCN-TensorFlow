#!/bin/bash
export PATH=/home/siddiqui/anaconda3/bin/:$PATH
source activate tf-nightly

# sudo userdocker run -it -v /netscratch:/netscratch dlcc/tensorflow_opencv /netscratch/siddiqui/Repositories/FCN-TensorFlow/train.sh
cd /netscratch/siddiqui/Repositories/FCN-TensorFlow/
python trainer_fcn.py -s -t --modelName IncResV2 --trainFileName ./data/train_pre_encoded.csv --valFileName ./data/train_pre_encoded.csv --testFileName ./data/train_pre_encoded.csv --trainingEpochs 2 --useSparseLabels --learningRate 1e-4 --maxImageSize 1024 --tensorboardVisualization
