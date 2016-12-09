import os
import numpy as np
import freeze_graph
import tensorflow as tf
from tensorflow.python.platform import gfile
from optparse import OptionParser
import datetime as dt

# Import FCN Model
import fcn2_vgg
import loss

# Command line options
parser = OptionParser()

# General settings
parser.add_option("-l", "--loadModel", action="store_true", dest="loadModel", default=False, help="Load model")
parser.add_option("-t", "--trainModel", action="store_true", dest="trainModel", default=False, help="Train model")
parser.add_option("-c", "--testModel", action="store_true", dest="testModel", default=False, help="Test model")
parser.add_option("-s", "--startTrainingFromScratch", action="store_true", dest="startTrainingFromScratch", default=False, help="Start training from scratch")
parser.add_option("-v", "--verbose", action="store", type="int", dest="verbose", default=0, help="Verbosity level")
parser.add_option("--tensorboardVisualization", action="store_true", dest="tensorboardVisualization", default=False, help="Enable tensorboard visualization")

# Input Reader Params
parser.add_option("--trainFileName", action="store", type="string", dest="trainFileName", default="train.idl", help="IDL file name for training")
parser.add_option("--testFileName", action="store", type="string", dest="testFileName", default="test.idl", help="IDL file name for testing")
parser.add_option("--statsFileName", action="store", type="string", dest="statsFileName", default="stats.txt", help="Image database statistics (mean, var)")
parser.add_option("--imageWidth", action="store", type="int", dest="imageWidth", default=640, help="Image width for feeding into the network")
parser.add_option("--imageHeight", action="store", type="int", dest="imageHeight", default=512, help="Image height for feeding into the network")
parser.add_option("--imageChannels", action="store", type="int", dest="imageChannels", default=3, help="Number of channels in image for feeding into the network")

# Trainer Params
parser.add_option("--learningRate", action="store", type="float", dest="learningRate", default=1e-6, help="Learning rate")
parser.add_option("--trainingEpochs", action="store", type="int", dest="trainingEpochs", default=10, help="Training epochs")
parser.add_option("--batchSize", action="store", type="int", dest="batchSize", default=5, help="Batch size")
parser.add_option("--displayStep", action="store", type="int", dest="displayStep", default=20, help="Progress display step")
parser.add_option("--saveStep", action="store", type="int", dest="saveStep", default=1000, help="Progress save step")
parser.add_option("--evaluateStep", action="store", type="int", dest="evaluateStep", default=100000, help="Progress evaluation step")
parser.add_option("--evaluateStepDontSaveImages", action="store_true", dest="evaluateStepDontSaveImages", default=False, help="Don't save images on evaluate step")
parser.add_option("--imagesOutputDirectory", action="store", type="string", dest="imagesOutputDirectory", default="./outputImages", help="Directory for saving output images")

# Directories
parser.add_option("--logsDir", action="store", type="string", dest="logsDir", default="./logs", help="Directory for saving logs")
parser.add_option("--checkpointDir", action="store", type="string", dest="checkpointDir", default="./checkpoints/", help="Directory for saving checkpoints")
parser.add_option("--graphDir", action="store", type="string", dest="graphDir", default="./graph/", help="Directory for saving graphs")
parser.add_option("--inputGraphName", action="store", type="string", dest="inputGraphName", default="Graph_CNN.pb", help="Name of the graph to be saved")
parser.add_option("--outputGraphName", action="store", type="string", dest="outputGraphName", default="Graph_Freezed.pb", help="Name of the graph to be loaded")

# Network Params
parser.add_option("--numClasses", action="store", type="int", dest="numClasses", default=2, help="Number of classes")
parser.add_option("--neuronAliveProbability", action="store", type="float", dest="neuronAliveProbability", default=0.5, help="Probability of keeping a neuron active during training")

# Parse command line options
(options, args) = parser.parse_args()
print options

# bestCheckpointDir = options.checkpointDir + "best_model/"
# checkpointPrefix = os.path.join(bestCheckpointDir, "saved_checkpoint")
checkpointStateName = "checkpoint_state"

# Import custom data
import inputReader
inputReader = inputReader.InputReader(options)

if options.trainModel:
	with tf.variable_scope('FCN_VGG'):
		# Data placeholders
		inputBatchImages = tf.placeholder(dtype=tf.float32, shape=[None, 512, 640, 3], name="inputBatchImages")
		inputBatchLabels = tf.placeholder(dtype=tf.float32, shape=[None, 512, 640, options.numClasses], name="inputBatchLabels")
		inputKeepProbability = tf.placeholder(dtype=tf.float32, name="inputKeepProbability")

	vgg_fcn = fcn2_vgg.FCN2VGG(batchSize = options.batchSize, statsFile=options.statsFileName, enableTensorboardVisualization=options.tensorboardVisualization)

	with tf.name_scope('Model'):
		# Construct model
		vgg_fcn.build(inputBatchImages, inputKeepProbability, num_classes=options.numClasses, 
						random_init_fc8=True, debug=(options.verbose > 0))

	with tf.name_scope('Loss'):
		# Define loss
		loss = loss.loss(vgg_fcn.softmax, inputBatchLabels, options.numClasses)

	with tf.name_scope('Optimizer'):
		# Define Optimizer
		#optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
		optimizer = tf.train.AdamOptimizer(learning_rate=options.learningRate)

		# Op to calculate every variable gradient
		gradients = tf.gradients(loss, tf.trainable_variables())
		gradients = list(zip(gradients, tf.trainable_variables()))
		# Op to update all variables according to their gradient
		applyGradients = optimizer.apply_gradients(grads_and_vars=gradients)

	# Initializing the variables
	init = tf.initialize_all_variables()

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
	lastSaveStep = 0

# Train model
if options.trainModel:
	with tf.Session() as sess:
		if options.loadModel:
			print "Loading"

			# Restore Graph
			with gfile.FastGFile(options.graphDir + options.outputGraphName, 'rb') as f:
				graphDef = tf.GraphDef()
				graphDef.ParseFromString(f.read())
				sess.graph.as_default()
				tf.import_graph_def(graphDef, name='')

				print "Graph Loaded"

			saver = tf.train.Saver(tf.all_variables())  # defaults to saving all variables - in this case w and b

			# Restore Model
			ckpt = tf.train.get_checkpoint_state(options.checkpointDir)
			if ckpt and ckpt.model_checkpoint_path:
				saver.restore(sess, ckpt.model_checkpoint_path)
				print "Model loaded Successfully!"
			else:
				print "Model not found"
				exit()

		if options.startTrainingFromScratch:
			print "Removing previous checkpoints and logs"
			os.system("rm -rf " + options.checkpointDir)
			os.system("rm -rf " + options.logsDir)
			os.system("rm -rf " + options.imagesOutputDirectory)
			os.system("mkdir " + options.checkpointDir)
			os.system("mkdir " + options.imagesOutputDirectory)

		# Restore checkpoint
		else:
			ckpt = tf.train.get_checkpoint_state(options.checkpointDir)
			if ckpt and ckpt.model_checkpoint_path:
				saver.restore(sess, ckpt.model_checkpoint_path)
				print(ckpt.model_checkpoint_path)
				print "Checkpoint loaded Successfully!"
			else:
				print "Checkpoint not found"
				exit()

			# Restore iteration number
			nameComps = ckpt.model_checkpoint_path.split('-')
			step = int(nameComps[1])
			inputReader.restoreCheckpoint(step)
			lastSaveStep = step

		if options.tensorboardVisualization:
			# Op for writing logs to Tensorboard
			summaryWriter = tf.train.SummaryWriter(options.logsDir, graph=tf.get_default_graph())

		print "Starting network training"
		sess.run(init)

		# Keep training until reach max iterations
		while True:
			batchImagesTrain, batchLabelsTrain = inputReader.getTrainBatch()
			# print "Batch shapes:"
			# print (batchImagesTrain.shape)
			# print (batchLabelsTrain.shape)
			# If training iterations completed
			if batchImagesTrain is None:
				print "Training completed"
				break

			# Run optimization op (backprop)
			if options.tensorboardVisualization:
				_, summary = sess.run([applyGradients, mergedSummaryOp], feed_dict={inputBatchImages: batchImagesTrain, inputBatchLabels: batchLabelsTrain, inputKeepProbability: options.neuronAliveProbability})
				# Write logs at every iteration
				summaryWriter.add_summary(summary, step)
			else:
				_ = sess.run([applyGradients], feed_dict={inputBatchImages: batchImagesTrain, inputBatchLabels: batchLabelsTrain, inputKeepProbability: options.neuronAliveProbability})

			if step % options.displayStep == 0:
				# Calculate batch loss
				# [trainLoss] = sess.run([loss], feed_dict={inputBatchImages: batchImagesTrain, inputBatchLabels: batchLabelsTrain})
				[trainLoss, trainImagesProbabilityMap] = sess.run([loss, vgg_fcn.probabilities], feed_dict={inputBatchImages: batchImagesTrain, inputBatchLabels: batchLabelsTrain, inputKeepProbability: 1})

				# print "Iter " + str(step) + ", Minibatch Loss= " + "{:.6f}".format(trainLoss)
				print "Iter " + str(step) + ", Minibatch Loss= ", trainLoss

				# Save image results
				print "Saving images"
				inputReader.saveLastBatchResults(trainImagesProbabilityMap, isTrain=True)
			step += 1

			if step % options.saveStep == 0:
				# Save model weights to disk
				saver.save(sess, options.checkpointDir + 'model.ckpt', global_step = step)
				print "Model saved in file: %s" % options.checkpointDir
				lastSaveStep = step

			#Check the accuracy on test data
			if step % options.evaluateStep == 0:
				# Report loss on test data
				batchImagesTest, batchLabelsTest = inputReader.getTestBatch()

				if options.evaluateStepDontSaveImages:
					[testLoss] = sess.run([loss], feed_dict={inputBatchImages: batchImagesTest, inputBatchLabels: batchLabelsTest, inputKeepProbability: 1})
					print "Test loss:", testLoss

				else:
					[testLoss, testImagesProbabilityMap] = sess.run([loss, vgg_fcn.probabilities], feed_dict={inputBatchImages: batchImagesTest, inputBatchLabels: batchLabelsTest, inputKeepProbability: 1})
					print "Test loss:", testLoss

					# Save image results
					print "Saving images"
					inputReader.saveLastBatchResults(testImagesProbabilityMap, isTrain=False)

				# #Check the accuracy on test data
				# if step % options.saveStepBest == 0:
				# 	# Report loss on test data
				# 	batchImagesTest, batchLabelsTest = inputReader.getTestBatch()
				# 	[testLoss] = sess.run([loss], feed_dict={inputBatchImages: batchImagesTest, inputBatchLabels: batchLabelsTest, inputKeepProbability: 1})
				# 	print "Test loss:", testLoss

				# 	# If its the best loss achieved so far, save the model
				# 	if testLoss < bestLoss:
				# 		bestLoss = testLoss
				# 		# bestModelSaver.save(sess, best_checkpoint_dir + 'checkpoint.data')
				# 		bestModelSaver.save(sess, checkpointPrefix, global_step=0, latest_filename=checkpointStateName)
				# 		print "Best model saved in file: %s" % checkpointPrefix
				# 	else:
				# 		print "Previous best accuracy: ", bestLoss

		# Save final model weights to disk
		saver.save(sess, options.checkpointDir + 'model.ckpt', global_step = step)
		print "Model saved in file: %s" % options.checkpointDir
		lastSaveStep = step

		# Write Graph to file
		print "Writing Graph to File"
		os.system("rm -rf " + options.graphDir)
		tf.train.write_graph(sess.graph_def, options.graphDir, options.inputGraphName, as_text=False) #proto

		# We save out the graph to disk, and then call the const conversion routine.
		inputGraphPath = options.graphDir + options.inputGraphName
		inputSaverDefPath = ""
		inputBinary = True
		# inputCheckpointPath = checkpointPrefix + "-0"
		inputCheckpointPath = options.checkpointDir + 'model.ckpt' + '-' + str(lastSaveStep)

		outputNodeNames = "Model/probabilities"
		restoreOpName = "save/restore_all"
		fileNameTensorName = "save/Const:0"
		outputGraphPath = options.graphDir + options.outputGraphName
		clearDevices = True

		freeze_graph.freeze_graph(inputGraphPath, inputSaverDefPath, inputBinary, inputCheckpointPath, outputNodeNames, restoreOpName, fileNameTensorName, outputGraphPath, clearDevices, "")

		# Report loss on test data
		batchImagesTest, batchLabelsTest = inputReader.getTestBatch()
		testLoss = sess.run(loss, feed_dict={inputBatchImages: batchImagesTest, inputBatchLabels: batchLabelsTest, inputKeepProbability: 1})
		print "Test loss (current):", testLoss

		print "Optimization Finished!"

		# # Report accuracy on test data using best fitted model
		# ckpt = tf.train.get_checkpoint_state(options.checkpointDir)
		# if ckpt and ckpt.model_checkpoint_path:
		# 	saver.restore(sess, ckpt.model_checkpoint_path)

		# # Report loss on test data
		# batchImagesTest, batchLabelsTest = inputReader.getTestBatch()
		# testLoss = sess.run(loss, feed_dict={inputBatchImages: batchImagesTest, inputBatchLabels: batchLabelsTest, inputKeepProbability: 1})
		# print "Test loss (best model):", testLoss

# Test model
if options.testModel:
	print "Testing saved Graph"
	outputGraphPath = options.graphDir + options.outputGraphName
	# Now we make sure the variable is now a constant, and that the graph still produces the expected result.
	with tf.Session() as session:
		outputGraphDef = tf.GraphDef()
		with open(outputGraphPath, "rb") as f:
			outputGraphDef.ParseFromString(f.read())
			session.graph.as_default()
			_ = tf.import_graph_def(outputGraphDef, name="")

		# Print all variables in graph
		print "Printing all variable names"
		allVars = outputGraphDef.node
		for node in allVars:
			print node.name
	
		# sess.run(tf.initialize_all_variables())
		outputNode = session.graph.get_tensor_by_name("Model/probabilities:0")
		inputBatchImages = session.graph.get_tensor_by_name("FCN_VGG/inputBatchImages:0")
		inputKeepProbability = session.graph.get_tensor_by_name("VGG/inputKeepProbability:0")

		batchImagesTest, batchLabelsTest = inputReader.getTestBatch()
		batchImagesTest = batchImagesTest[0:1, :, :, :]
		batchLabelsTest = batchLabelsTest[0:1, :, :, :]
		startTime = dt.datetime.now()
		output = session.run(outputNode, feed_dict={inputBatchImages: batchImagesTest, inputKeepProbability: 1})
		print "Output shape:", output.shape
		endTime = dt.datetime.now()

		print "Time consumed in executing graph:", ((endTime.microsecond - startTime.microsecond) / 1e6)

		# assert(len(output) == len(options.batchSize))

	print "Graph tested"