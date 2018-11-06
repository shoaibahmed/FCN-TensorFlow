#!/bin/python

import os
import sys

import cv2
import numpy as np
from optparse import OptionParser
import datetime as dt

import tensorflow.contrib.slim as slim
import tensorflow as tf
from tensorflow.python.platform import gfile

import shutil
import wget
import tarfile

# Constants
TRAIN = 0
VAL = 1
TEST = 2

COLORS = np.array([[0, 0, 0], [0, 128, 0], [0, 0, 128], [192, 224, 224]]) # RGB
LABELS = np.array([0, 1, 1, 2]) # Give high weight to the boundary

# Command line options
parser = OptionParser()

# General settings
parser.add_option("-t", "--trainModel", action="store_true", dest="trainModel", default=False, help="Train model")
parser.add_option("-c", "--testModel", action="store_true", dest="testModel", default=False, help="Test model")
parser.add_option("-d", "--debug", action="store_true", dest="debug", default=False, help="Enable debugging model - high verbosity")
parser.add_option("--tensorboardVisualization", action="store_true", dest="tensorboardVisualization", default=False, help="Enable tensorboard visualization")

# Input Reader Params
parser.add_option("--trainFileName", action="store", type="string", dest="trainFileName", default="./data/train.csv", help="File containing the training file names")
parser.add_option("--valFileName", action="store", type="string", dest="valFileName", default="./data/val.csv", help="File containing the validation file names")
parser.add_option("--testFileName", action="store", type="string", dest="testFileName", default="./data/test.csv", help="File containing the test file names")
parser.add_option("--statsFileName", action="store", type="string", dest="statsFileName", default="stats.txt", help="Image database statistics (mean, var)")
parser.add_option("--maxImageSize", action="store", type="int", dest="maxImageSize", default=2048, help="Maximum size of the larger dimension while preserving aspect ratio")
parser.add_option("--imageChannels", action="store", type="int", dest="imageChannels", default=3, help="Number of channels in image for feeding into the network")
parser.add_option("--shufflePerBatch", action="store_true", dest="shufflePerBatch", default=False, help="Shuffle input for every batch")
parser.add_option("--mapLabelsFromRGB", action="store_true", dest="mapLabelsFromRGB", default=False, help="Map labels from RGB to integers (if data is in form [H, W, 3])")
parser.add_option("--useSparseLabels", action="store_true", dest="useSparseLabels", default=False, help="Use sparse labels (Mask shape: [H, W, 1] instead of [H, W, C] where C is the number of classes)")
parser.add_option("--boundaryWeight", action="store", type="float", dest="boundaryWeight", default=10.0, help="Weight to be given to the boundary for computing the total loss")
parser.add_option("--numParallelLoaders", action="store", type="int", dest="numParallelLoaders", default=8, help="Number of parallel loaders to be used for data loading")

# Trainer Params
parser.add_option("--learningRate", action="store", type="float", dest="learningRate", default=1e-4, help="Learning rate")
parser.add_option("--weightDecayLambda", action="store", type="float", dest="weightDecayLambda", default=5e-5, help="Weight Decay Lambda")
parser.add_option("--trainingEpochs", action="store", type="int", dest="trainingEpochs", default=5, help="Training epochs")
parser.add_option("--batchSize", action="store", type="int", dest="batchSize", default=1, help="Batch size")
parser.add_option("--displayStep", action="store", type="int", dest="displayStep", default=5, help="Progress display step")
parser.add_option("--saveStep", action="store", type="int", dest="saveStep", default=1000, help="Progress save step")
parser.add_option("--evaluateStep", action="store", type="int", dest="evaluateStep", default=100000, help="Progress evaluation step")
parser.add_option("--evaluateStepDontSaveImages", action="store_true", dest="evaluateStepDontSaveImages", default=False, help="Don't save images on evaluate step")
parser.add_option("--trainImagesOutputDirectory", action="store", type="string", dest="trainImagesOutputDirectory", default="./outputImages_train", help="Directory for saving output images for train set")
parser.add_option("--valImagesOutputDirectory", action="store", type="string", dest="valImagesOutputDirectory", default="./outputImages_val", help="Directory for saving output images for validation set")
parser.add_option("--testImagesOutputDirectory", action="store", type="string", dest="testImagesOutputDirectory", default="./outputImages_test", help="Directory for saving output images for test set")

# Directories
parser.add_option("--logsDir", action="store", type="string", dest="logsDir", default="./logs-str", help="Directory for saving logs")
parser.add_option("--pretrainedModelsDir", action="store", type="string", dest="pretrainedModelsDir", default="./pretrained/", help="Directory containing the pretrained models")
parser.add_option("--outputModelDir", action="store", type="string", dest="outputModelDir", default="./output-str/", help="Directory for saving the model")
parser.add_option("--outputModelName", action="store", type="string", dest="outputModelName", default="Model", help="Name to be used for saving the model")

# Network Params
parser.add_option("-m", "--modelName", action="store", dest="modelName", default="NASNet", choices=["NASNet", "IncResV2"], help="Name of the model to be used")
parser.add_option("-s", "--startTrainingFromScratch", action="store_true", dest="startTrainingFromScratch", default=False, help="Start training from scratch")
parser.add_option("--numClasses", action="store", type="int", dest="numClasses", default=3, help="Number of classes")
parser.add_option("--ignoreLabel", action="store", type="int", dest="ignoreLabel", default=255, help="Label to ignore for loss computation")
parser.add_option("--useSkipConnections", action="store_true", dest="useSkipConnections", default=False, help="Use skip connections")
parser.add_option("--concatenateFeatureMaps", action="store_true", dest="concatenateFeatureMaps", default=False, help="Concatenate feature maps instead of point-wise summation")
parser.add_option("--useDeformableConvolution", action="store_true", dest="useDeformableConvolution", default=False, help="Use deformable convolution")
parser.add_option("--performBoundaryDetection", action="store_true", dest="performBoundaryDetection", default=False, help="Perform boundary detection using Hough line transform")
parser.add_option("--useCRFPostProcessing", action="store_true", dest="useCRFPostProcessing", default=False, help="Use CRF based post-processing")

# Extra features
parser.add_option("--adversarialExamples", action="store_true", dest="adversarialExamples", default=False, help="Generate adversarial examples")
parser.add_option("--inverseOptimization", action="store_true", dest="inverseOptimization", default=False, help="Perform inverse optimization to generate images from mask")

# Parse command line options
(options, args) = parser.parse_args()

# Verification
assert options.batchSize == 1, "Error: Only batch size of 1 is supported due to aspect aware scaling!"
try:
	import pydensecrf.densecrf as dcrf
except:
	# print("Error: Failed to import pydensecrf! CRF post-processing will not work.")
	assert not options.useCRFPostProcessing, "Error: Failed to import pydensecrf!"

options.outputModelDir = os.path.join(options.outputModelDir, "trained-" + options.modelName + "_concat" if options.concatenateFeatureMaps else "_sum")
options.outputModelName = options.outputModelName + "_" + options.modelName
if options.useDeformableConvolution:
	options.outputModelDir += "_deform"
	options.outputModelName += "_deform"

options.trainImagesOutputDirectory = os.path.join(options.outputModelDir, options.trainImagesOutputDirectory)
options.valImagesOutputDirectory = os.path.join(options.outputModelDir, options.valImagesOutputDirectory)
options.testImagesOutputDirectory = os.path.join(options.outputModelDir, options.testImagesOutputDirectory)

print (options)

# Check if the pretrained directory exists
if not os.path.exists(options.pretrainedModelsDir):
	print ("Warning: Pretrained models directory not found!")
	os.makedirs(options.pretrainedModelsDir)
	assert os.path.exists(options.pretrainedModelsDir)

# Clone the repository if not already existent
if not os.path.exists(os.path.join(options.pretrainedModelsDir, "models/research/slim")):
	print ("Cloning TensorFlow models repository")
	import git # gitpython

	class Progress(git.remote.RemoteProgress):
		def update(self, op_code, cur_count, max_count=None, message=''):
			print (self._cur_line)

	git.Repo.clone_from("https://github.com/tensorflow/models.git", os.path.join(options.pretrainedModelsDir, "models"), progress=Progress())
	print ("Repository sucessfully cloned!")

# Add the path to the tensorflow models repository
sys.path.append(os.path.join(options.pretrainedModelsDir, "models/research/slim"))
sys.path.append(os.path.join(options.pretrainedModelsDir, "models/research/slim/nets"))

import inception_resnet_v2
import resnet_v1
import nasnet.nasnet as nasnet

# Import FCN Model
if options.modelName == "NASNet":
	print ("Downloading pretrained NASNet model")
	nasCheckpointFile = checkpointFileName = os.path.join(options.pretrainedModelsDir, options.modelName, 'model.ckpt')
	if not os.path.isfile(nasCheckpointFile + '.index'):
		# Download file from the link
		url = 'https://storage.googleapis.com/download.tensorflow.org/models/nasnet-a_large_04_10_2017.tar.gz'
		fileName = wget.download(url, options.pretrainedModelsDir)
		print ("File downloaded: %s" % fileName)

		# Extract the tar file
		tar = tarfile.open(fileName)
		tar.extractall(path=os.path.join(options.pretrainedModelsDir, options.modelName))
		tar.close()

	# Update image sizes
	# options.imageHeight = options.imageWidth = 331

elif options.modelName == "IncResV2":
	print ("Downloading pretrained Inception ResNet v2 model")
	incResV2CheckpointFile = checkpointFileName = os.path.join(options.pretrainedModelsDir, options.modelName, 'inception_resnet_v2_2016_08_30.ckpt')
	if not os.path.isfile(incResV2CheckpointFile):
		# Download file from the link
		url = 'http://download.tensorflow.org/models/inception_resnet_v2_2016_08_30.tar.gz'
		fileName = wget.download(url, options.pretrainedModelsDir)
		print ("File downloaded: %s" % fileName)

		# Extract the tar file
		tar = tarfile.open(fileName)
		tar.extractall(path=os.path.join(options.pretrainedModelsDir, options.modelName))
		tar.close()

	# Update image sizes
	# options.imageHeight = options.imageWidth = 299

else:
	print ("Error: Model not found!")
	exit (-1)

# Reads an image from a file, decodes it into a dense tensor
def parseFunction(imgFileName, rowMaskImageName, colMaskImageName):
	# TODO: Replace with decode_image (decode_image doesn't return shape)
	# Load the original image
	imageString = tf.read_file(imgFileName)
	img = tf.image.decode_jpeg(imageString)
	img = tf.image.resize_images(img, [options.maxImageSize, options.maxImageSize], preserve_aspect_ratio=True)
	img.set_shape([None, None, options.imageChannels])
	img = tf.cast(img, tf.float32) # Convert to float tensor

	# Load the segmentation mask
	imageString = tf.read_file(rowMaskImageName)
	rowMask = tf.image.decode_png(imageString)
	imageString = tf.read_file(colMaskImageName)
	colMask = tf.image.decode_png(imageString)

	# if options.mapLabelsFromRGB:
	# 	assert False # Not working at this point
	# 	# TODO: Optimize this mapping
	# 	if options.useSparseLabels:
	# 		# raise NotImplementedError
	# 		maskNew = tf.zeros(shape=[tf.shape(mask)[0], tf.shape(mask)[1]])
	# 		for idx, color in enumerate(colors):
	# 			maskNew = tf.cond(tf.reduce_all(tf.equal(mask, color), axis=-1), lambda: maskNew, lambda: maskNew)
	# 			# maskNew = tf.cond(tf.equal(mask, color), lambda: labels[idx], lambda: maskNew)

	# 	else:
	# 		semanticMap = []
	# 		for color in colors:
	# 			classMap = tf.reduce_all(tf.equal(mask, color), axis=-1)
	# 			semanticMap.append(classMap)
	# 		mask = tf.to_float(tf.stack(semanticMap, axis=-1))

	rowMask = tf.image.resize_images(rowMask, [options.maxImageSize, options.maxImageSize], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, preserve_aspect_ratio=True)
	rowMask = tf.cast(rowMask, tf.int32) # Convert to float tensor

	colMask = tf.image.resize_images(colMask, [options.maxImageSize, options.maxImageSize], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, preserve_aspect_ratio=True)
	colMask = tf.cast(colMask, tf.int32) # Convert to float tensor

	return imgFileName, img, rowMask, colMask

def dataAugmentationFunction(imgFileName, img, rowMask, colMask):
	with tf.name_scope('flipLR'):
		randomVar = tf.random_uniform(maxval=2, dtype=tf.int32, shape=[]) # Random variable: two possible outcomes (0 or 1)
		img = tf.cond(pred=tf.equal(randomVar, 0), true_fn=lambda: tf.image.flip_left_right(img), false_fn=lambda: img)
		rowMask = tf.cond(pred=tf.equal(randomVar, 0), true_fn=lambda: tf.image.flip_left_right(rowMask), false_fn=lambda: rowMask)
		colMask = tf.cond(pred=tf.equal(randomVar, 0), true_fn=lambda: tf.image.flip_left_right(colMask), false_fn=lambda: colMask)

	with tf.name_scope('flipUD'):
		randomVar = tf.random_uniform(maxval=2, dtype=tf.int32, shape=[]) # Random variable: two possible outcomes (0 or 1)
		img = tf.cond(pred=tf.equal(randomVar, 0), true_fn=lambda: tf.image.flip_up_down(img), false_fn=lambda: img)
		rowMask = tf.cond(pred=tf.equal(randomVar, 0), true_fn=lambda: tf.image.flip_up_down(rowMask), false_fn=lambda: rowMask)
		colMask = tf.cond(pred=tf.equal(randomVar, 0), true_fn=lambda: tf.image.flip_up_down(colMask), false_fn=lambda: colMask)

	img = tf.image.random_brightness(img, max_delta=32.0 / 255.0)
	img = tf.image.random_saturation(img, lower=0.5, upper=1.5)
	img = tf.image.random_contrast(img, lower=0.5, upper=1.5)

	# Make sure the image is still in [0, 255]
	img = tf.clip_by_value(img, 0.0, 255.0)

	return imgFileName, img, rowMask, colMask

def loadDataset(currentDataFile, dataAugmentation=False):
	print ("Loading data from file: %s" % (currentDataFile))
	dataClasses = {}
	with open(currentDataFile) as f:
		imageFileNames = f.readlines()
		originalImageNames = []
		rowMaskImageNames = []
		colMaskImageNames = []
		for imName in imageFileNames:
			imName = imName.strip().split(',')

			originalImageNames.append(imName[0])
			rowMaskImageNames.append(imName[1])
			colMaskImageNames.append(imName[2])

		originalImageNames = tf.constant(originalImageNames)
		rowMaskImageNames = tf.constant(rowMaskImageNames)
		colMaskImageNames = tf.constant(colMaskImageNames)

	numFiles = len(imageFileNames)
	print ("Dataset loaded")
	print ("Number of files found: %d" % (numFiles))

	dataset = tf.data.Dataset.from_tensor_slices((originalImageNames, rowMaskImageNames, colMaskImageNames))
	dataset = dataset.map(parseFunction, num_parallel_calls=options.numParallelLoaders)

	# Data augmentation
	if dataAugmentation:
		dataset = dataset.map(dataAugmentationFunction, num_parallel_calls=options.numParallelLoaders)

	# Data shuffling
	if options.shufflePerBatch:
		dataset = dataset.shuffle(buffer_size=numFiles)

	dataset = dataset.batch(options.batchSize)

	return dataset

def writeMaskToImage(img, rowMask, colMask, directory, fileName, append='', overlay=True):
	fileName = fileName[0].decode("utf-8") 
	_, fileName = os.path.split(fileName) # Crop the complete path name
	if img is not None:
		img = img[0]
	rowMask = rowMask[0]
	colMask = colMask[0]
	fileNameRoot, fileNameExt = os.path.splitext(fileName)

	for maskName, mask in zip(["-row", "-col"], [rowMask, colMask]):
		outputFileName = os.path.join(directory, fileNameRoot + maskName + append + fileNameExt)
		if options.debug:
			print ("Saving predicted segmentation mask:", outputFileName)

		if img is not None:
			rgbMask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.float32)
			for color, label in zip(COLORS, LABELS):
				binaryMap = mask[:, :, 0] == label
				rgbMask[binaryMap, 0] = color[0]
				rgbMask[binaryMap, 1] = color[1]
				rgbMask[binaryMap, 2] = color[2]

			if overlay:
				rgbMask = np.uint8(cv2.addWeighted(img.astype(np.float32), 0.5, rgbMask, 0.5, 0.0))

			# Write the resulting image to file
			cv2.imwrite(outputFileName, rgbMask)
		else:
			cv2.imwrite(outputFileName, mask)

# Performs the upsampling of the given images
def attachDecoder(net, endPoints, inputShape, trainModel, activation=tf.nn.leaky_relu, numFilters=256, filterSize=(3, 3), strides=(2, 2), padding='same', batchNorm=True):
	if options.useDeformableConvolution:
		# Attach deformable convolution head here
		import tensorlayer
		with tf.name_scope('DeformableConv'):
			net = activation(net)
			offset1 = tl.layers.Conv2d(net, 18, (3, 3), (1, 1), act=act, padding='SAME', name='offset')
			net = tl.layers.DeformableConv2d(net, offset1, numFilters, (3, 3), act=act, name='deformable')

	with tf.name_scope('Decoder'), tf.variable_scope('Decoder'):
		out = tf.layers.conv2d_transpose(activation(net), numFilters, filterSize, strides=strides, padding='valid')
		if batchNorm:
			out = tf.layers.batch_normalization(out, training=trainModel, name='decoder_bn_1')

		if options.useSkipConnections:
			endPointName = 'Mixed_6a'
			endPoint = endPoints[endPointName]
			encShape = tf.shape(endPoint)
			if options.concatenateFeatureMaps:
				out = tf.concat((tf.image.resize_bilinear(out, [encShape[1], encShape[2]], align_corners=True), endPoint), axis=-1)
			else:
				out = tf.image.resize_bilinear(out, [encShape[1], encShape[2]], align_corners=True) + tf.layers.conv2d(endPoint, numFilters, (1, 1), strides=(1, 1), padding='same')

		out = tf.layers.conv2d(activation(out), numFilters, filterSize, strides=(1, 1), padding=padding)
		if batchNorm:
			out = tf.layers.batch_normalization(out, training=trainModel, name='decoder_bn_2')

		out = tf.layers.conv2d_transpose(activation(out), numFilters, filterSize, strides=strides, padding='valid')
		if batchNorm:
			out = tf.layers.batch_normalization(out, training=trainModel, name='decoder_bn_3')

		if options.useSkipConnections:
			endPointName = 'MaxPool_5a_3x3'
			endPoint = endPoints[endPointName]
			encShape = tf.shape(endPoint)
			if options.concatenateFeatureMaps:
				out = tf.concat((tf.image.resize_bilinear(out, [encShape[1], encShape[2]], align_corners=True), endPoint), axis=-1)
			else:
				out = tf.image.resize_bilinear(out, [encShape[1], encShape[2]], align_corners=True) + tf.layers.conv2d(endPoint, numFilters, (1, 1), strides=(1, 1), padding='same')

		out = tf.layers.conv2d(activation(out), numFilters, filterSize, strides=(1, 1), padding=padding)
		if batchNorm:
			out = tf.layers.batch_normalization(out, training=trainModel, name='decoder_bn_4')

		out = tf.layers.conv2d_transpose(activation(out), numFilters, filterSize, strides=strides, padding='valid')
		if batchNorm:
			out = tf.layers.batch_normalization(out, training=trainModel, name='decoder_bn_5')

		if options.useSkipConnections:
			endPointName = 'Conv2d_4a_3x3'
			endPoint = endPoints[endPointName]
			encShape = tf.shape(endPoint)
			if options.concatenateFeatureMaps:
				out = tf.concat((tf.image.resize_bilinear(out, [encShape[1], encShape[2]], align_corners=True), endPoint), axis=-1)
			else:
				out = tf.image.resize_bilinear(out, [encShape[1], encShape[2]], align_corners=True) + tf.layers.conv2d(endPoint, numFilters, (1, 1), strides=(1, 1), padding='same')

		out = tf.layers.conv2d(activation(out), numFilters, filterSize, strides=(1, 1), padding=padding)
		if batchNorm:
			out = tf.layers.batch_normalization(out, training=trainModel, name='decoder_bn_6')

		out = tf.layers.conv2d_transpose(activation(out), numFilters, filterSize, strides=strides, padding='valid')
		if batchNorm:
			out = tf.layers.batch_normalization(out, training=trainModel, name='decoder_bn_7')

		if options.useSkipConnections:
			# out = out + tf.layers.conv2d(endPoints['Conv2d_2b_3x3'], numFilters, (1, 1), strides=(1, 1), padding='same')
			endPointName = 'Conv2d_2b_3x3'
			endPoint = endPoints[endPointName]
			encShape = tf.shape(endPoint)
			if options.concatenateFeatureMaps:
				out = tf.concat((tf.image.resize_bilinear(out, [encShape[1], encShape[2]], align_corners=True), endPoint), axis=-1)
			else:
				out = tf.image.resize_bilinear(out, [encShape[1], encShape[2]], align_corners=True) + tf.layers.conv2d(endPoint, numFilters, (1, 1), strides=(1, 1), padding='same')

		out = tf.layers.conv2d(activation(out), numFilters, filterSize, strides=(1, 1), padding=padding)
		if batchNorm:
			out = tf.layers.batch_normalization(out, training=trainModel, name='decoder_bn_8')

		# Match dimensions (convolutions with 'valid' padding reducing the dimensions)
		out = tf.image.resize_bilinear(out, [inputShape[1], inputShape[2]], align_corners=True) # TODO: Is it useful or it doesn't matter?
		out = tf.layers.conv2d(activation(out), numFilters, filterSize, strides=(1, 1), padding='same')
		if batchNorm:
			out = tf.layers.batch_normalization(out, training=trainModel, name='decoder_bn_9')

		outRow = tf.layers.conv2d(activation(out), 64, filterSize, strides=(1, 1), padding=padding) # Obtain per pixel predictions for row
		outCol = tf.layers.conv2d(activation(out), 64, filterSize, strides=(1, 1), padding=padding) # Obtain per pixel predictions for column
		if batchNorm:
			outRow = tf.layers.batch_normalization(outRow, training=trainModel, name='decoder_bn_row_1')
			outCol = tf.layers.batch_normalization(outCol, training=trainModel, name='decoder_bn_col_1')

		outRowTwo = tf.layers.conv2d(activation(outRow), 64, filterSize, strides=(1, 1), padding=padding) # Obtain per pixel predictions for row
		outColTwo = tf.layers.conv2d(activation(outCol), 64, filterSize, strides=(1, 1), padding=padding) # Obtain per pixel predictions for column
		if batchNorm:
			outRowTwo = tf.layers.batch_normalization(outRowTwo, training=trainModel, name='decoder_bn_row_2')
			outColTwo = tf.layers.batch_normalization(outColTwo, training=trainModel, name='decoder_bn_col_2')

		outRow = tf.concat([outRow, activation(outRowTwo)], axis=-1)
		outCol = tf.concat([outCol, activation(outColTwo)], axis=-1)

		outRow = tf.layers.conv2d(outRow, options.numClasses, filterSize, strides=(1, 1), padding=padding) # Obtain per pixel predictions for row
		outCol = tf.layers.conv2d(outCol, options.numClasses, filterSize, strides=(1, 1), padding=padding) # Obtain per pixel predictions for column
	return outRow, outCol


# Create dataset objects
trainDataset = loadDataset(options.trainFileName, dataAugmentation=True)
trainIterator = trainDataset.make_initializable_iterator()

valDataset = loadDataset(options.valFileName)
valIterator = valDataset.make_initializable_iterator()

testDataset = loadDataset(options.testFileName)
testIterator = testDataset.make_initializable_iterator()

# Data placeholders
datasetSelectionPlaceholder = tf.placeholder(dtype=tf.int32, shape=(), name='DatasetSelectionPlaceholder')
inputBatchImageNames, inputBatchImages, inputBatchRowMasks, inputBatchColMasks = tf.cond(tf.equal(datasetSelectionPlaceholder, TRAIN), lambda: trainIterator.get_next(), 
															lambda: tf.cond(tf.equal(datasetSelectionPlaceholder, VAL), lambda: valIterator.get_next(), lambda: testIterator.get_next()))
print ("Data shape: %s | Row mask shape: %s | Column mask shape: %s" % (str(inputBatchImages.get_shape()), str(inputBatchRowMasks.get_shape()), str(inputBatchColMasks.get_shape())))

# if options.trainModel:
with tf.name_scope('Model'):
	if options.inverseOptimization:
		inputBatchRowMasks = tf.placeholder(tf.float32, shape=[])
		scaledInputBatchImages = tf.Variable(initial_value=tf.zeros_like(inputBatchImages), trainable=True, dtype=tf.float32, validate_shape=False) # Same size as the mask

	else:
		# Scaling only for NASNet and IncResV2
		scaledInputBatchImages = tf.scalar_mul((1.0 / 255.0), inputBatchImages)
		scaledInputBatchImages = tf.subtract(scaledInputBatchImages, 0.5)
		scaledInputBatchImages = tf.multiply(scaledInputBatchImages, 2.0)


	# Create model
	if options.modelName == "NASNet":
		arg_scope = nasnet.nasnet_large_arg_scope()
		with slim.arg_scope(arg_scope):
			logits, endPoints = nasnet.build_nasnet_large(scaledInputBatchImages, is_training=False, num_classes=options.numClasses)

	elif options.modelName == "IncResV2":
		arg_scope = inception_resnet_v2.inception_resnet_v2_arg_scope()
		with slim.arg_scope(arg_scope):
			# logits, endPoints = inception_resnet_v2.inception_resnet_v2(scaledInputBatchImages, is_training=False)
			with tf.variable_scope('InceptionResnetV2', 'InceptionResnetV2', [scaledInputBatchImages], reuse=None) as scope:
				with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=False):
				  net, endPoints = inception_resnet_v2.inception_resnet_v2_base(scaledInputBatchImages, scope=scope, activation_fn=tf.nn.relu)

		variablesToRestore = slim.get_variables_to_restore(include=["InceptionResnetV2"])

	else:
		print ("Error: Model not found!")
		exit (-1)

# TODO: Attach the decoder to the encoder
print (endPoints.keys())
if options.useSkipConnections:
	print ("Adding skip connections %s from the encoder to the decoder!" % ("via concatenation" if options.concatenateFeatureMaps else "via point-wise summation"))

predictedRowLogits, predictedColLogits = attachDecoder(net, endPoints, tf.shape(scaledInputBatchImages), options.trainModel)
predictedRowMask = tf.expand_dims(tf.argmax(predictedRowLogits, axis=-1), -1, name="predictedRowMask")
predictedColMask = tf.expand_dims(tf.argmax(predictedColLogits, axis=-1), -1, name="predictedColMask")

if options.tensorboardVisualization:
	tf.summary.image('Original Image', inputBatchImages, max_outputs=3)
	tf.summary.image('Desired Row Mask', tf.to_float(inputBatchRowMasks), max_outputs=3)
	tf.summary.image('Desired Col Mask', tf.to_float(inputBatchColMasks), max_outputs=3)
	tf.summary.image('Predicted Row Mask', tf.to_float(predictedRowMask), max_outputs=3)
	tf.summary.image('Predicted Col Mask', tf.to_float(predictedColMask), max_outputs=3)

with tf.name_scope('Loss'):
	# Reshape 4D tensors to 2D, each row represents a pixel, each column a class
	predictedRowMaskFlattened = tf.reshape(predictedRowLogits, (-1, tf.shape(predictedRowLogits)[1] * tf.shape(predictedRowLogits)[2], options.numClasses), name="fcnRowLogits")
	predictedColMaskFlattened = tf.reshape(predictedColLogits, (-1, tf.shape(predictedColLogits)[1] * tf.shape(predictedColLogits)[2], options.numClasses), name="fcnColLogits")
	inputRowMaskFlattened = tf.reshape(inputBatchRowMasks, (-1, tf.shape(inputBatchRowMasks)[1] * tf.shape(inputBatchRowMasks)[2]))
	inputColMaskFlattened = tf.reshape(inputBatchColMasks, (-1, tf.shape(inputBatchColMasks)[1] * tf.shape(inputBatchColMasks)[2]))

	# Define loss
	weightsRow = tf.cast(inputRowMaskFlattened != options.ignoreLabel, dtype=tf.float32)
	weightsCol = tf.cast(inputColMaskFlattened != options.ignoreLabel, dtype=tf.float32)
	weightsRow = tf.cond(pred=tf.equal(weightsRow, 2), true_fn=lambda: options.boundaryWeight, false_fn=lambda: weightsRow)
	weightsCol = tf.cond(pred=tf.equal(weightsCol, 2), true_fn=lambda: options.boundaryWeight, false_fn=lambda: weightsCol)
	crossEntropyLossRow = tf.losses.sparse_softmax_cross_entropy(labels=inputRowMaskFlattened, logits=predictedRowMaskFlattened, weights=weightsRow)
	crossEntropyLossCol = tf.losses.sparse_softmax_cross_entropy(labels=inputColMaskFlattened, logits=predictedColMaskFlattened, weights=weightsCol)
	regLoss = options.weightDecayLambda * tf.reduce_sum(tf.losses.get_regularization_losses())
	crossEntropyLoss = crossEntropyLossRow + crossEntropyLossCol
	loss = tf.add(crossEntropyLoss, regLoss, name="totalLoss")

with tf.name_scope('Optimizer'):
	# Define Optimizer
	if options.inverseOptimization:
		optimizer = tf.train.AdamOptimizer(learning_rate=options.learningRate).minimize(loss, var_list=[scaledInputBatchImages]) # Var list contains the input image proxy variable
	else:
		optimizer = tf.train.AdamOptimizer(learning_rate=options.learningRate)

		# Op to calculate every variable gradient
		gradients = tf.gradients(loss, tf.trainable_variables())
		gradients = list(zip(gradients, tf.trainable_variables()))

		# Op to update all variables according to their gradient
		updateOps = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		with tf.control_dependencies(updateOps):
			applyGradients = optimizer.apply_gradients(grads_and_vars=gradients)

# Initializing the variables
init = tf.global_variables_initializer()
init_local = tf.local_variables_initializer()

if options.tensorboardVisualization:
	# Create a summary to monitor cost tensor
	tf.summary.scalar("cross_entropy_row", crossEntropyLossRow)
	tf.summary.scalar("cross_entropy_col", crossEntropyLossCol)
	tf.summary.scalar("cross_entropy", crossEntropyLoss)
	tf.summary.scalar("reg_loss", regLoss)
	tf.summary.scalar("total_loss", loss)

	# Create summaries to visualize weights
	for var in tf.trainable_variables():
		tf.summary.histogram(var.name, var)
	# Summarize all gradients
	for grad, var in gradients:
		if grad is not None:
			tf.summary.histogram(var.name + '/gradient', grad)

	# Merge all summaries into a single op
	mergedSummaryOp = tf.summary.merge_all()

# 'Saver' op to save and restore all the variables
saver = tf.train.Saver()

# GPU config
config = tf.ConfigProto()
config.gpu_options.allow_growth=True

# Train model
if options.trainModel:
	with tf.Session(config=config) as sess:
		# Initialize all variables
		sess.run(init)
		sess.run(init_local)

		if options.startTrainingFromScratch:
			print ("Removing previous checkpoints and logs")
			if os.path.exists(options.logsDir): 
				shutil.rmtree(options.logsDir)
			if os.path.exists(options.trainImagesOutputDirectory): 
				shutil.rmtree(options.trainImagesOutputDirectory)
			if os.path.exists(options.valImagesOutputDirectory): 
				shutil.rmtree(options.valImagesOutputDirectory)
			if os.path.exists(options.testImagesOutputDirectory): 
				shutil.rmtree(options.testImagesOutputDirectory)
			if os.path.exists(options.outputModelDir): 
				shutil.rmtree(options.outputModelDir)
			
			os.makedirs(options.outputModelDir)
			os.makedirs(options.trainImagesOutputDirectory)
			os.makedirs(options.valImagesOutputDirectory)
			os.makedirs(options.testImagesOutputDirectory)

			# Load the pre-trained Inception ResNet v2 model
			restorer = tf.train.Saver(variablesToRestore)
			restorer.restore(sess, incResV2CheckpointFile)

		# Restore checkpoint
		else:
			print ("Restoring from checkpoint")
			saver = tf.train.import_meta_graph(os.path.join(options.outputModelDir, options.outputModelName + ".meta"))
			saver.restore(sess, os.path.join(options.outputModelDir, options.outputModelName))

		if options.tensorboardVisualization:
			# Op for writing logs to Tensorboard
			summaryWriter = tf.summary.FileWriter(options.logsDir, graph=tf.get_default_graph())

		print ("Starting network training")
		globalStep = 0

		# Keep training until reach max iterations
		for epoch in range(options.trainingEpochs):
			# Initialize the dataset iterators
			sess.run(trainIterator.initializer)
			
			try:
				step = 0
				while True:
					# Debug mode
					if options.debug:
						[predRowMask, predColMask, gtRowMask, gtColMask] = sess.run([predictedRowMask, predictedColMask, inputBatchRowMasks, inputBatchColMasks], feed_dict={datasetSelectionPlaceholder: TRAIN})
						print ("Row | Prediction shape: %s | GT shape: %s" % (str(predRowMask.shape), str(gtRowMask.shape)))
						print ("Column | Prediction shape: %s | GT shape: %s" % (str(predColMask.shape), str(gtColMask.shape)))
						assert (predRowMask.shape == gtRowMask.shape), "Error: Prediction and ground-truth row shapes don't match"
						assert (predColMask.shape == gtColMask.shape), "Error: Prediction and ground-truth col shapes don't match"
						if np.isnan(np.sum(predRowMask)) or np.isnan(np.sum(predColMask)):
							print ("Error: NaN encountered!")
							exit (-1)

						print ("Unique labels in row prediction:", np.unique(predRowMask))
						print ("Unique labels in column prediction:", np.unique(predColMask))
						print ("Unique labels in row GT:", np.unique(gtRowMask))
						print ("Unique labels in col GT:", np.unique(gtColMask))

						# Verify end point shapes
						for endPointName in endPoints:
							endPointOutput = sess.run(endPoints[endPointName], feed_dict={datasetSelectionPlaceholder: TRAIN})
							print ("End point: %s | Shape: %s" % (endPointName, str(endPointOutput.shape)))

					# Run optimization op (backprop)
					if options.tensorboardVisualization:
						_, summary = sess.run([applyGradients, mergedSummaryOp], feed_dict={datasetSelectionPlaceholder: TRAIN})
						summaryWriter.add_summary(summary, global_step=globalStep) # Write logs at every iteration
					else:
						_ = sess.run(applyGradients, feed_dict={datasetSelectionPlaceholder: TRAIN})
					
					if step % options.displayStep == 0:
						# Calculate batch loss
						[fileName, originalImage, trainLoss, predictedRowSegMask, predictedColSegMask] = \
							sess.run([inputBatchImageNames, inputBatchImages, loss, predictedRowMask, predictedColMask], feed_dict={datasetSelectionPlaceholder: TRAIN})
						print ("Epoch: %d | Iteration: %d | Minibatch Loss: %f" % (epoch, step, trainLoss))

						# Save image results
						writeMaskToImage(originalImage, predictedRowSegMask, predictedColSegMask, options.trainImagesOutputDirectory, fileName)

					step += 1
					globalStep += 1

			except tf.errors.OutOfRangeError:
				print('Done training for %d epochs, %d steps.' % (epoch, step))

			if step % options.saveStep == 0:
				# Save model weights to disk
				outputFileName = os.path.join(options.outputModelDir, options.outputModelName)
				saver.save(sess, outputFileName)
				print ("Model saved: %s" % (outputFileName))

			# Check the accuracy on validation set
			sess.run(valIterator.initializer)
			averageValLoss = 0.0
			iterations = 0
			try:
				while True:
					[fileName, originalImage, valLoss, predictedRowSegMask, predictedColSegMask] = \
						sess.run([inputBatchImageNames, inputBatchImages, loss, predictedRowMask, predictedColMask], feed_dict={datasetSelectionPlaceholder: VAL})
					
					# Save image results
					writeMaskToImage(originalImage, predictedRowSegMask, predictedColSegMask, options.valImagesOutputDirectory, fileName)

					print ("Iteration: %d | Validation loss: %f" % (iterations, valLoss))
					averageValLoss += valLoss
					iterations += 1

			except tf.errors.OutOfRangeError:
				print('Evaluation on validation set completed!')

			averageValLoss /= iterations
			print('Average validation loss: %f' % (averageValLoss))

			# # Check the accuracy on test data
			# if step % options.saveStepBest == 0:
			# 	# Report loss on test data
			# 	[testLoss] = sess.run([loss], feed_dict={inputBatchImages: batchImagesTest, inputBatchLabels: batchLabelsTest, inputKeepProbability: 1.0})
			# 	print ("Test loss: %f" % testLoss)

			# 	# If its the best loss achieved so far, save the model
			# 	if testLoss < bestLoss:
			# 		bestLoss = testLoss
			# 		# bestModelSaver.save(sess, best_checkpoint_dir + 'checkpoint.data')
			# 		bestModelSaver.save(sess, checkpointPrefix, global_step=0, latest_filename=checkpointStateName)
			# 		print ("Best model saved in file: %s" % checkpointPrefix)
			# 	else:
			# 		print ("Previous best accuracy: %f" % bestLoss)

		# Save final model weights to disk
		outputFileName = os.path.join(options.outputModelDir, options.outputModelName)
		saver.save(sess, outputFileName)
		print ("Model saved: %s" % (outputFileName))

		# Report loss on test data
		sess.run(testIterator.initializer)
		averageTestLoss = 0.0
		iterations = 0
		try:
			while True:
				[fileName, originalImage, testLoss, predictedRowSegMask, predictedColSegMask] = \
					sess.run([inputBatchImageNames, inputBatchImages, loss, predictedRowMask, predictedColMask], feed_dict={datasetSelectionPlaceholder: TEST})
				
				# Save image results
				writeMaskToImage(originalImage, predictedRowSegMask, predictedColSegMask, options.testImagesOutputDirectory, fileName)

				print ("Iteration: %d | Test loss: %f" % (iterations, testLoss))
				averageTestLoss += testLoss
				iterations += 1

		except tf.errors.OutOfRangeError:
			print('Evaluation on test set completed!')

		averageTestLoss /= iterations
		print('Average test loss: %f' % (averageTestLoss))

		print ("Optimization completed!")


if options.testModel:
	print ("Testing saved model")

	if os.path.exists(options.testImagesOutputDirectory):
		shutil.rmtree(options.testImagesOutputDirectory)
	os.makedirs(options.testImagesOutputDirectory)
	
	# Now we make sure the variable is now a constant, and that the graph still produces the expected result.
	with tf.Session(config=config) as sess:
		modelFileName = os.path.join(options.outputModelDir, options.outputModelName)
		saver.restore(sess, modelFileName)

		sess.run(testIterator.initializer)
		iterations = 0
		averageTestLoss = 0.0
		try:
			while True:
				[fileName, originalImage, testLoss, predictedRowSegMask, predictedColSegMask, predictedRowSegLogits, predictedColSegLogits] = \
					sess.run([inputBatchImageNames, inputBatchImages, loss, predictedRowMask, predictedColMask, predictedRowLogits, predictedColLogits], feed_dict={datasetSelectionPlaceholder: TEST})

				# Save image results
				writeMaskToImage(originalImage, predictedRowSegMask, predictedColSegMask, options.testImagesOutputDirectory, fileName)

				if options.performBoundaryDetection:
					processedMasks = []
					processedImages = []
					for idx, mask in enumerate([predictedRowSegMask, predictedColSegMask]):
						# Use hough transform to infer lines
						booleanMask = mask[0, :, :, 0] == (options.numClasses - 1)
						newImage = np.zeros(booleanMask.shape, dtype=np.uint8)
						newImage[booleanMask] = 255
						newImage = cv2.dilate(newImage, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations = 1)
						edges = cv2.Canny(newImage, 50, 150, apertureSize = 3)
						# cv2.imwrite('canny.jpg',edges)
						constantDivisor = 6.0
						minVotes = int(newImage.shape[1] / constantDivisor) # Row rows => width / constantDivisor
						if idx == 1: # If column
							minVotes = int(newImage.shape[0] / 8.0) # For cols => height / constantDivisor
						lines = cv2.HoughLines(edges, 1, (np.pi if idx == 1 else np.pi / 2), minVotes)
						img = originalImage[0].copy()
						# print ("Lines shape:", lines.shape)
						if lines is not None:
							# Filter only horizontal or vertical lines
							newLines = []
							numRejectedLines = 0
							for i in range(0, len(lines)):
								theta = lines[i][0][1]
								theta = int(theta * (180.0 / np.pi)) # Convert radians to degrees
								# Any vertical line will have 0 degree and horizontal lines will have 90 degree
								if ((idx == 1) and (theta != 0)) or ((idx == 0) and (theta != 90)):
									numRejectedLines += 1
								else:
									newLines.append(lines[i])
							
							lines = newLines
							print ("Number of lines rejected:", numRejectedLines)

							# Cluster the lines
							from sklearn.cluster import MeanShift
							# meanShift = MeanShift(bandwidth=20.0 if len(lines) < 10 else 10.0)
							meanShift = MeanShift(bandwidth=20.0 if idx == 1 else 10.0) # Higher bandwidth for columns since they are well-separated
							clusters = meanShift.fit_predict(np.array(lines)[:, 0, :])
							clusterNumbers = np.unique(clusters)
							clusterCenters = meanShift.cluster_centers_
							print ("Number of clusters found:", len(clusterNumbers))

							for i in range(0, len(lines)):
								rho = lines[i][0][0]
								theta = lines[i][0][1]
								# if (idx == 0 and theta != 0) or (idx == 1 and theta != 90):
								# 	return
								a = np.cos(theta)
								b = np.sin(theta)
								x0 = a*rho
								y0 = b*rho
								x1 = int(x0 + 1000*(-b))
								y1 = int(y0 + 1000*(a))
								x2 = int(x0 - 1000*(-b))
								y2 = int(y0 - 1000*(a))

								cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)

							for i in range(0, clusterCenters.shape[0]):
								rho = clusterCenters[i, 0]
								theta = clusterCenters[i, 1]
								a = np.cos(theta)
								b = np.sin(theta)
								x0 = a*rho
								y0 = b*rho
								x1 = int(x0 + 1000*(-b))
								y1 = int(y0 + 1000*(a))
								x2 = int(x0 - 1000*(-b))
								y2 = int(y0 - 1000*(a))

								cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)

						processedMasks.append(img[np.newaxis, :, :, :])
						processedImages.append(newImage[np.newaxis, :, :, np.newaxis])

					# Save image results
					writeMaskToImage(None, processedMasks[0], processedMasks[1], options.testImagesOutputDirectory, fileName, append='-hough')
					writeMaskToImage(None, processedImages[0], processedImages[1], options.testImagesOutputDirectory, fileName, append='-hough-debug')
					
				if options.useCRFPostProcessing:
					# TODO: Incorporate dense CRF
					processedMasks = []
					for predictedMask in [predictedRowSegLogits, predictedColSegLogits]:
						unary = predictedMask[0]
						unary = -np.log(unary)
						unary = unary.transpose(1, 0, 2)
						w, h, c = unary.shape
						unary = unary.transpose(1, 0, 2).reshape(options.numClasses, -1)
						unary = np.ascontiguousarray(unary)
						resizedImg = np.ascontiguousarray(originalImage[0]).astype(np.uint8)

						d = dcrf.DenseCRF2D(w, h, options.numClasses)
						d.setUnaryEnergy(unary)
						d.addPairwiseBilateral(sxy=5, srgb=3, rgbim=resizedImg, compat=1)

						q = d.inference(50)
						mask = np.argmax(q, axis=0).reshape(w, h).transpose(1, 0)
						mask = np.array(mask, dtype=np.uint8)
						processedMasks.append(mask[np.newaxis, :, :, np.newaxis])

					# Save image results
					writeMaskToImage(originalImage, processedMasks[0], processedMasks[1], options.testImagesOutputDirectory, fileName, append='-crf')
				
				print ("Iteration: %d | Test loss: %f" % (iterations, testLoss))
				averageTestLoss += testLoss
				iterations += 1

		except tf.errors.OutOfRangeError:
			print('Evluation on test set completed!')

		averageTestLoss /= iterations
		print('Average test loss: %f' % (averageTestLoss))

	print ("Model evaluation completed!")
