#!/bin/bash
export PATH=/home/siddiqui/anaconda3/bin/:$PATH
source activate tf-nightly
cd /netscratch/siddiqui/Repositories/FCN-TensorFlow/

# sudo userdocker run -it -v /netscratch:/netscratch dlcc/tensorflow_opencv /netscratch/siddiqui/Repositories/FCN-TensorFlow/train.sh

echo "Training FCN model"
python trainer_fcn.py -s --trainModel --modelName IncResV2 --trainFileName ./data/train_pre_encoded.csv --valFileName ./data/val_pre_encoded.csv --testFileName ./data/val_pre_encoded.csv --trainingEpochs 1 --useSparseLabels --learningRate 1e-4 --weightDecayLambda 1e-4 --maxImageSize 1024 --tensorboardVisualization --boundaryWeight 50.0

echo "Testing trained FCN model"
python trainer_fcn.py --testModel --modelName IncResV2 --trainFileName ./data/train_pre_encoded.csv --valFileName ./data/val_pre_encoded.csv --testFileName ./data/val_pre_encoded.csv --useSparseLabels --maxImageSize 1024
