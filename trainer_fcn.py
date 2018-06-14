#!/bin/python

import os
import sys

import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
from optparse import OptionParser
import datetime as dt

import shutil
import wget
import tarfile

# Command line options
parser = OptionParser()

# General settings
parser.add_option("-t", "--trainModel", action="store_true", dest="trainModel", default=False, help="Train model")
parser.add_option("-c", "--testModel", action="store_true", dest="testModel", default=False, help="Test model")
parser.add_option("-v", "--verbose", action="store", type="int", dest="verbose", default=0, help="Verbosity level")
parser.add_option("--tensorboardVisualization", action="store_true", dest="tensorboardVisualization", default=False, help="Enable tensorboard visualization")

# Input Reader Params
parser.add_option("--trainFileName", action="store", type="string", dest="trainFileName", default="train.idl", help="File containing the training file names")
parser.add_option("--valFileName", action="store", type="string", dest="valFileName", default="val.idl", help="File containing the validation file names")
parser.add_option("--testFileName", action="store", type="string", dest="testFileName", default="test.idl", help="File containing the test file names")
parser.add_option("--statsFileName", action="store", type="string", dest="statsFileName", default="stats.txt", help="Image database statistics (mean, var)")
parser.add_option("--maxImageSize", action="store", type="int", dest="maxImageSize", default=2048, help="Maximum size of the larger dimension while preserving aspect ratio")
# parser.add_option("--imageWidth", action="store", type="int", dest="imageWidth", default=640, help="Image width for feeding into the network")
# parser.add_option("--imageHeight", action="store", type="int", dest="imageHeight", default=512, help="Image height for feeding into the network")
parser.add_option("--imageChannels", action="store", type="int", dest="imageChannels", default=3, help="Number of channels in image for feeding into the network")
parser.add_option("--shufflePerBatch", action="store_true", dest="shufflePerBatch", default=False, help="Shuffle input for every batch")

# Trainer Params
parser.add_option("--learningRate", action="store", type="float", dest="learningRate", default=1e-4, help="Learning rate")
parser.add_option("--trainingEpochs", action="store", type="int", dest="trainingEpochs", default=5, help="Training epochs")
parser.add_option("--batchSize", action="store", type="int", dest="batchSize", default=1, help="Batch size")
parser.add_option("--displayStep", action="store", type="int", dest="displayStep", default=5, help="Progress display step")
parser.add_option("--saveStep", action="store", type="int", dest="saveStep", default=1000, help="Progress save step")
parser.add_option("--evaluateStep", action="store", type="int", dest="evaluateStep", default=100000, help="Progress evaluation step")
parser.add_option("--evaluateStepDontSaveImages", action="store_true", dest="evaluateStepDontSaveImages", default=False, help="Don't save images on evaluate step")
parser.add_option("--imagesOutputDirectory", action="store", type="string", dest="imagesOutputDirectory", default="./outputImages", help="Directory for saving output images")

# Directories
parser.add_option("--logsDir", action="store", type="string", dest="logsDir", default="./logs", help="Directory for saving logs")
# parser.add_option("--checkpointDir", action="store", type="string", dest="checkpointDir", default="./checkpoints/", help="Directory for saving checkpoints")
parser.add_option("--pretrainedModelsDir", action="store", type="string", dest="pretrainedModelsDir", default="./pretrained/", help="Directory containing the pretrained models")
parser.add_option("--outputModelDir", action="store", type="string", dest="modelDir", default="", help="Directory for saving the model")
parser.add_option("--outputModelName", action="store", type="string", dest="modelName", default="", help="Name to be used for saving the model")

# Network Params
parser.add_option("-m", "--modelName", action="store", dest="modelName", default="NASNet", choices=["NASNet", "IncResV2"], help="Name of the model to be used")
parser.add_option("-s", "--startTrainingFromScratch", action="store_true", dest="startTrainingFromScratch", default=False, help="Start training from scratch")
parser.add_option("--numClasses", action="store", type="int", dest="numClasses", default=3, help="Number of classes")
parser.add_option("--ignoreLabel", action="store", type="int", dest="ignoreLabel", default=255, help="Label to ignore for loss computation")
parser.add_option("--neuronAliveProbability", action="store", type="float", dest="neuronAliveProbability", default=0.5, help="Probability of keeping a neuron active during training")

# Parse command line options
(options, args) = parser.parse_args()

# Verification
assert options.batchSize == 1, "Error: Only batch size of 1 is supported due to aspect aware scaling!"
options.outputModelDir = "trained-" + options.modelName
options.outputModelName = "Model-" + options.modelName
print (options)

# Check if the pretrained directory exists
if not os.path.exists(options.pretrainedModelsDir):
	print ("Warning: Pretrained models directory not found!")
	os.mkdir(options.pretrainedModelsDir)
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
	print ("Loading NASNet")
	nasCheckpointFile = checkpointFileName = os.path.join(options.pretrainedModelsDir, 'model.ckpt')
	if not os.path.isfile(nasCheckpointFile + '.index'):
		# Download file from the link
		url = 'https://storage.googleapis.com/download.tensorflow.org/models/nasnet-a_large_04_10_2017.tar.gz'
		fileName = wget.download(url, options.pretrainedModelsDir)
		print ("File downloaded: %s" % fileName)

		# Extract the tar file
		tar = tarfile.open(fileName)
		tar.extractall()
		tar.close()

	# Update image sizes
	# options.imageHeight = options.imageWidth = 331

elif options.modelName == "IncResV2":
	print ("Loading Inception ResNet v2")
	incResV2CheckpointFile = checkpointFileName = os.path.join(options.pretrainedModelsDir, 'inception_resnet_v2_2016_08_30.ckpt')
	if not os.path.isfile(incResV2CheckpointFile):
		# Download file from the link
		url = 'http://download.tensorflow.org/models/inception_resnet_v2_2016_08_30.tar.gz'
		fileName = wget.download(url, options.pretrainedModelsDir)
		print ("File downloaded: %s" % fileName)

		# Extract the tar file
		tar = tarfile.open(fileName)
		tar.extractall()
		tar.close()

	# Update image sizes
	# options.imageHeight = options.imageWidth = 299

else:
	print ("Error: Model not found!")
	exit (-1)

# Reads an image from a file, decodes it into a dense tensor
def _parse_function(imgFileName, gtFileName):
	# Load the original image
	image_string = tf.read_file(imgFileName)
	img = tf.image.decode_jpeg(image_string)
	img = tf.image.resize_images(img, [options.maxImageSize, options.maxImageSize], preserve_aspect_ratio=True)
	# img.set_shape([options.imageHeight, options.imageWidth, options.imageChannels])
	img = tf.cast(img, tf.float32) # Convert to float tensor

	# Load the segmentation mask
	image_string = tf.read_file(gtFileName)
	mask = tf.image.decode_jpeg(image_string)
	mask = tf.image.resize_images(mask, [options.maxImageSize, options.maxImageSize], method=ResizeMethod.NEAREST_NEIGHBOR, preserve_aspect_ratio=True)
	mask = tf.cast(mask, tf.float32) # Convert to float tensor

	return filename, img, mask

def loadDataset(currentDataFile):
	print ("Loading data from file: %s" % (currentDataFile))
	dataClasses = {}
	with open(currentDataFile) as f:
		imageFileNames = f.readlines()
		originalImageNames = []
		maskImageNames = []
		for imName in imageFileNames:
			imName = imName.strip().split(' ')

			originalImageNames.append(imName[0])
			maskImageNames.append(imName[1])

		originalImageNames = tf.constant(originalImageNames)
		maskImageNames = tf.constant(maskImageNames)

	numFiles = len(imageFileNames)
	print ("Dataset loaded")
	print ("Number of files found: %d" % (numFiles))

	dataset = tf.contrib.data.Dataset.from_tensor_slices((originalImageNames, maskImageNames))
	dataset = dataset.map(_parse_function)
	if options.shufflePerBatch:
		dataset = dataset.shuffle(buffer_size=numFiles)
	dataset = dataset.batch(options.batchSize)

	return dataset

# Create dataset objects
trainDataset, _ = loadDataset(options.trainFileName)
trainIterator = trainDataset.make_initializable_iterator()

valDataset, _ = loadDataset(options.valFileName)
valIterator = valDataset.make_initializable_iterator()

testDataset, _ = loadDataset(options.testFileName)
testIterator = testDataset.make_initializable_iterator()

globalStep = tf.train.get_or_create_global_step()

if options.trainModel:
	with tf.name_scope('Model'):
		# Data placeholders
		inputBatchImageNames, inputBatchImages, inputBatchMasks = Iterator.get_next()
		print ("Data shape: %s | Mask shape: %s" % (str(inputBatchImages.get_shape()), str(inputBatchMasks.get_shape())))

		# Data placeholders
		inputBatchImagesPlaceholder = tf.placeholder(dtype=tf.float32, shape=[None, None, None, options.imageChannels], name="inputBatchImages")

		# Scaling only for NASNet and IncResV2
		scaledInputBatchImages = tf.scalar_mul((1.0 / 255.0), inputBatchImagesPlaceholder)
		scaledInputBatchImages = tf.subtract(scaledInputBatchImages, 0.5)
		scaledInputBatchImages = tf.multiply(scaledInputBatchImages, 2.0)

		# Create model
		if options.model == "NASNet":
			arg_scope = nasnet.nasnet_large_arg_scope()
			with slim.arg_scope(arg_scope):
				logits, endPoints = nasnet.build_nasnet_large(scaledInputBatchImages, is_training=False, num_classes=options.numClasses)

		elif options.model == "IncResV2":
			arg_scope = inception_resnet_v2.inception_resnet_v2_arg_scope()
			with slim.arg_scope(arg_scope):
				logits, endPoints = inception_resnet_v2.inception_resnet_v2(scaledInputBatchImages, is_training=False)

		else:
			print ("Error: Model not found!")
			exit (-1)

	# TODO: Attach the decoder to the encoder
	print (endPoints.keys())
	exit (-1)
	attachDecoder()

	with tf.name_scope('Loss'):
		# Reshape 4D tensors to 2D, each row represents a pixel, each column a class
		predictedMask = tf.reshape(predictedMask, (-1, options.numClasses), name="fcnLogits")
		inputMask = tf.reshape(inputBatchMasks, (-1, options.numClasses))

		# Define loss
		weights = tf.cast(inputMask != options.ignoreLabel, dtype=tf.float32)
		weights[weights == 2] = 5 # High weight to the boundary
		crossEntropyLoss = tf.nn.weighted_cross_entropy_with_logits(targets=inputMask, logits=predictedMask, pos_weight=weights, name="weightedCrossEntropy")
		loss = tf.reduce_sum(slim.losses.get_regularization_losses()) + crossEntropyLoss

	with tf.name_scope('Optimizer'):
		# Define Optimizer
		optimizer = tf.train.AdamOptimizer(learning_rate=options.learningRate)

		# Op to calculate every variable gradient
		gradients = tf.gradients(loss, tf.trainable_variables())
		gradients = list(zip(gradients, tf.trainable_variables()))
		# Op to update all variables according to their gradient
		applyGradients = optimizer.apply_gradients(grads_and_vars=gradients)

	# Initializing the variables
	init = tf.global_variables_initializer()
	init_local = tf.local_variables_initializer()

	if options.tensorboardVisualization:
		# Create a summary to monitor cost tensor
		tf.scalar_summary("loss", loss)

		# Create summaries to visualize weights
		for var in tf.trainable_variables():
			tf.histogram_summary(var.name, var)
		# Summarize all gradients
		for grad, var in gradients:
			tf.histogram_summary(var.name + '/gradient', grad)

		# Merge all summaries into a single op
		mergedSummaryOp = tf.merge_all_summaries()

	# 'Saver' op to save and restore all the variables
	saver = tf.train.Saver()
	# bestModelSaver = tf.train.Saver()

	bestLoss = 1e9
	step = 1

# Train model
if options.trainModel:
	with tf.Session() as sess:
		# Initialize all variables
		sess.run(init)
		sess.run(init_local)

		if options.startTrainingFromScratch:
			print ("Removing previous checkpoints and logs")
			os.system("rm -rf " + options.logsDir)
			os.system("rm -rf " + options.imagesOutputDirectory)
			os.system("rm -rf " + options.modelDir)
			os.system("mkdir " + options.imagesOutputDirectory)
			os.system("mkdir " + options.modelDir)

			# Load the pre-trained Inception ResNet v2 model
			variables_to_restore = slim.get_variables_to_restore(include=["InceptionResnetV2"])
			restorer = tf.train.Saver(variables_to_restore)
			restorer.restore(sess, inc_res_v2_checkpoint_file)

		# Restore checkpoint
		else:
			print ("Restoring from checkpoint")
			saver = tf.train.import_meta_graph(options.modelDir + options.modelName + ".meta")
			saver.restore(sess, options.modelDir + options.modelName)

		if options.tensorboardVisualization:
			# Op for writing logs to Tensorboard
			summaryWriter = tf.train.SummaryWriter(options.logsDir, graph=tf.get_default_graph())

		print ("Starting network training")
		
		# Keep training until reach max iterations
		while True:
			batchImagesTrain, batchLabelsTrain = inputReader.getTrainBatch()
			# print ("Batch images shape: %s, Batch labels shape: %s" % (batchImagesTrain.shape, batchLabelsTrain.shape))

			# If training iterations completed
			if batchImagesTrain is None:
				print ("Training completed")
				break

			# Run optimization op (backprop)
			if options.tensorboardVisualization:
				_, summary = sess.run([applyGradients, mergedSummaryOp], feed_dict={inputBatchImages: batchImagesTrain, inputBatchLabels: batchLabelsTrain, inputKeepProbability: options.neuronAliveProbability})
				# Write logs at every iteration
				summaryWriter.add_summary(summary, step)
			else:
				[trainLoss, _] = sess.run([loss, applyGradients], feed_dict={inputBatchImages: batchImagesTrain, inputBatchLabels: batchLabelsTrain, inputKeepProbability: options.neuronAliveProbability})
				print ("Iteration: %d, Minibatch Loss: %f" % (step, trainLoss))

			if step % options.displayStep == 0:
				# Calculate batch loss
				# [trainLoss] = sess.run([loss], feed_dict={inputBatchImages: batchImagesTrain, inputBatchLabels: batchLabelsTrain})
				[trainLoss, trainImagesProbabilityMap] = sess.run([loss, probabilities], feed_dict={inputBatchImages: batchImagesTrain, inputBatchLabels: batchLabelsTrain, inputKeepProbability: 1.0})

				# print ("Iter " + str(step) + ", Minibatch Loss= " + "{:.6f}".format(trainLoss))
				# print ("Iteration: %d, Minibatch Loss: %f" % (step, trainLoss))

				# Save image results
				print ("Saving images")
				print (trainImagesProbabilityMap.shape)
				inputReader.saveLastBatchResults(trainImagesProbabilityMap, isTrain=True)
			step += 1

			if step % options.saveStep == 0:
				# Save model weights to disk
				saver.save(sess, options.modelDir + options.modelName)
				print ("Model saved: %s" % (options.modelDir + options.modelName))

			#Check the accuracy on test data
			if step % options.evaluateStep == 0:
				# Report loss on test data
				batchImagesTest, batchLabelsTest = inputReader.getTestBatch()

				if options.evaluateStepDontSaveImages:
					[testLoss] = sess.run([loss], feed_dict={inputBatchImages: batchImagesTest, inputBatchLabels: batchLabelsTest, inputKeepProbability: 1.0})
					print ("Test loss: %f" % testLoss)

				else:
					[testLoss, testImagesProbabilityMap] = sess.run([loss, vgg_fcn.probabilities], feed_dict={inputBatchImages: batchImagesTest, inputBatchLabels: batchLabelsTest, inputKeepProbability: 1.0})
					print ("Test loss: %f" % testLoss)

					# Save image results
					print ("Saving images")
					inputReader.saveLastBatchResults(testImagesProbabilityMap, isTrain=False)

				# #Check the accuracy on test data
				# if step % options.saveStepBest == 0:
				# 	# Report loss on test data
				# 	batchImagesTest, batchLabelsTest = inputReader.getTestBatch()
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
		saver.save(sess, options.modelDir + options.modelName)
		print ("Model saved: %s" % (options.modelDir + options.modelName))

		# # Write Graph to file
		# print ("Writing Graph to File")
		# os.system("rm -rf " + options.graphDir)
		# tf.train.write_graph(sess.graph_def, options.graphDir, options.inputGraphName, as_text=False) #proto

		# # We save out the graph to disk, and then call the const conversion routine.
		# inputGraphPath = options.graphDir + options.inputGraphName
		# inputSaverDefPath = ""
		# inputBinary = True
		# # inputCheckpointPath = checkpointPrefix + "-0"
		# inputCheckpointPath = options.checkpointDir + 'model.ckpt' + '-' + str(lastSaveStep)

		# outputNodeNames = "Model/probabilities"
		# restoreOpName = "save/restore_all"
		# fileNameTensorName = "save/Const:0"
		# outputGraphPath = options.graphDir + options.outputGraphName
		# clearDevices = True

		# freeze_graph.freeze_graph(inputGraphPath, inputSaverDefPath, inputBinary, inputCheckpointPath, outputNodeNames, restoreOpName, fileNameTensorName, outputGraphPath, clearDevices, "")

		# Report loss on test data
		batchImagesTest, batchLabelsTest = inputReader.getTestBatch()
		testLoss = sess.run(loss, feed_dict={inputBatchImages: batchImagesTest, inputBatchLabels: batchLabelsTest, inputKeepProbability: 1.0})
		print ("Test loss (current): %f" % testLoss)

		print ("Optimization Finished!")

		# # Report accuracy on test data using best fitted model
		# ckpt = tf.train.get_checkpoint_state(options.checkpointDir)
		# if ckpt and ckpt.model_checkpoint_path:
		# 	saver.restore(sess, ckpt.model_checkpoint_path)

		# # Report loss on test data
		# batchImagesTest, batchLabelsTest = inputReader.getTestBatch()
		# testLoss = sess.run(loss, feed_dict={inputBatchImages: batchImagesTest, inputBatchLabels: batchLabelsTest, inputKeepProbability: 1.0})
		# print ("Test loss (best model): %f" % testLoss)

# Test model
if options.testModel:
	print ("Testing saved model")

	os.system("rm -rf " + options.imagesOutputDirectory)
	os.system("mkdir " + options.imagesOutputDirectory)
	
	# Now we make sure the variable is now a constant, and that the graph still produces the expected result.
	with tf.Session() as session:
		saver = tf.train.import_meta_graph(options.modelDir + options.modelName + ".meta")
		saver.restore(session, options.modelDir + options.modelName)

		# Get reference to placeholders
		outputNode = session.graph.get_tensor_by_name("Decoder_InceptionResnetV2/probabilities:0")
		inputBatchImages = session.graph.get_tensor_by_name("FCN_INC_RES_V2/inputBatchImages:0")
		inputKeepProbability = session.graph.get_tensor_by_name("FCN_INC_RES_V2/inputKeepProbability:0")
	
		# sess.run(tf.initialize_all_variables())
		# Sample 50 test batches
		numBatches = 50
		for i in range(1, numBatches):
			print ("Prcessing batch # %d" % i)
			batchImagesTest, batchLabelsTest = inputReader.getTestBatch(readMask = False) # For testing on datasets without GT mask
			# output = session.run(outputNode, feed_dict={inputBatchImages: batchImagesTest, inputKeepProbability: 1.0})
			[imagesProbabilityMap] = session.run([outputNode], feed_dict={inputBatchImages: batchImagesTest, inputKeepProbability: 1.0})

			# Save image results
			print ("Saving images")
			# print (imagesProbabilityMap.shape)
			imagesProbabilityMap = np.reshape(imagesProbabilityMap, [-1, options.imageHeight, options.imageWidth, options.numClasses])
			inputReader.saveLastBatchResults(imagesProbabilityMap, isTrain=False)

	print ("Model tested")