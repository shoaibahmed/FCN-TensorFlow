import os
import numpy as np
import skimage
import skimage.io

class InputReader:
	def __init__(self, options):
		self.options = options

		# Reads pathes of images together with their labels
		self.imageList = self.readImageNames(self.options.trainFileName)
		self.imageListTest = self.readImageNames(self.options.testFileName)

		self.currentIndex = 0
		self.totalEpochs = 0
		self.totalImages = len(self.imageList)
		self.totalImagesTest = len(self.imageListTest)

	def readImageNames(self, imageListFile):
		"""Reads a .txt file containing pathes and labeles
		Args:
		   imageListFile: a .txt file with one /path/to/image per line
		Returns:
		   List with all fileNames in file imageListFile
		"""
		f = open(imageListFile, 'r')
		fileNames = []
		for line in f:
			# Get file name
			fileNames.append(line.strip())
		return fileNames

	def readImagesFromDisk(self, fileNames):
		"""Consumes a list of filenames and returns image with mask
		Args:
		  fileNames: List of image files
		Returns:
		  Two 4-D numpy arrays: The input images as well as well their corresponding binary mask
		"""
		images = []
		masks = []
		for i in xrange(0, len(fileNames)):
			maskImageName = fileNames[i]
			# maskImageName = maskImageName[:-15] + '_mask/mask' + maskImageName[-9:]
			lastSlashIndex = maskImageName[::-1].index('/')
			imageName = maskImageName[-lastSlashIndex:]
			if "-" not in imageName:
				maskImageName = maskImageName[:-lastSlashIndex] + 'mask' + maskImageName[-9:]
			else:
				maskImageName = maskImageName[:-lastSlashIndex] + 'mask' + maskImageName[-11:-6] + maskImageName[-4:]
			
			if self.options.verbose > 1:
				print ("Image: %s" % fileNames[i])
				print ("Mask: %s" % maskImageName)

			# Read image
			img = skimage.io.imread(fileNames[i])
			images.append(img)

			# Read mask
			mask = skimage.io.imread(maskImageName)

			# Convert the mask to [H, W, options.numClasses]
			backgroundClass = (mask == 0).astype(np.uint8)
			foregroundClass = (mask == 255).astype(np.uint8)
			mask = np.stack([backgroundClass, foregroundClass], axis=2)
			masks.append(mask)

		# Convert list to ndarray
		images = np.array(images)
		masks = np.array(masks)

		return images, masks

	def getTrainBatch(self):
		"""Returns training images and masks in batch
		Args:
		  None
		Returns:
		  Two 4-D numpy arrays: training images and masks in batch.
		"""
		print ("Training epochs completed: %f" % (self.totalEpochs + (float(self.currentIndex) / self.totalImages)))
		
		if self.totalEpochs >= self.options.trainingEpochs:
			return None, None

		self.indices = np.random.choice(self.totalImages, self.options.batchSize)
		imageBatch, maskBatch = self.readImagesFromDisk([self.imageList[index] for index in self.indices])

		self.currentIndex = self.currentIndex + self.options.batchSize
		if self.currentIndex > self.totalImages:
			self.currentIndex = self.currentIndex - self.totalImages
			self.totalEpochs = self.totalEpochs + 1

		return imageBatch, maskBatch

	def getTestBatch(self):
		"""Returns testing images and masks in batch
		Args:
		  None
		Returns:
		  Two 4-D numpy arrays: test images and masks in batch.
		"""
		# Optional Image and Label Batching
		self.indices = np.random.choice(self.totalImagesTest, self.options.batchSize)
		imageBatch, maskBatch = self.readImagesFromDisk([self.imageListTest[index] for index in self.indices])
		return imageBatch, maskBatch

	def restoreCheckpoint(self, numSteps):
		"""Restores current index and epochs using numSteps
		Args:
		  numSteps: Number of batches processed
		Returns:
		  None
		"""
		processedImages = numSteps * self.options.batchSize
		self.totalEpochs = processedImages / self.totalImages
		self.currentIndex = processedImages % self.totalImages

	def saveLastBatchResults(self, outputImages, isTrain=True):
		"""Saves the results of last retrieved image batch
		Args:
		  outputImages: 4D Numpy array [batchSize, H, W, numClasses]
		  isTrain: If the last batch was training batch
		Returns:
		  None
		"""
		if isTrain:
			imageNames = [self.imageList[index] for index in self.indices]
		else:
			imageNames = [self.imageListTest[index] for index in self.indices]

		# Iterate over each image name and save the results
		for i in xrange(0, self.options.batchSize):
			imageName = imageNames[i].split('/')
			imageName = imageName[-1]
			if isTrain:
				imageName = self.options.imagesOutputDirectory + '/' + 'train_' + imageName[:-4] + '_prob' + imageName[-4:]
			else:
				imageName = self.options.imagesOutputDirectory + '/' + 'test_' + imageName[:-4] + '_prob' + imageName[-4:]
			# print(imageName)

			# Save foreground probability
			im = np.squeeze(outputImages[i, :, :, 1] * 255)
			im = im.astype(np.uint8)	# Convert image from float to unit8 for saving
			skimage.io.imsave(imageName, im)
