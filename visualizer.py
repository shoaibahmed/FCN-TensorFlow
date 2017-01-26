import os
import random
from os import listdir
from os.path import isfile, join
from optparse import OptionParser

import cv2

def visualize(options):
	print(options)

	for root, dirs, files in os.walk(options.outputDir):
		path = root.split('/')
		print ("Directory:", os.path.basename(root))
		
		for file in files:
			# print(file)
			if file.endswith(options.outputExtension):
				outputFileName = str(os.path.abspath(os.path.join(root, file)))

				# imageName = self.options.imagesOutputDirectory + '/' + 'test_' + imageName[:-4] + '_prob' + imageName[-4:]
				inputFileName = str(file).split('_')
				inputFileName = options.inputDir + inputFileName[1] + options.inputExtension
				print(inputFileName)

				outputIm = cv2.imread(outputFileName)
				inputIm = cv2.imread(inputFileName)

				cv2.imshow("Input", inputIm)
				cv2.imshow("Output", outputIm)

				c = cv2.waitKey(-1)
				if c == 113: # Pressed 'q'
					exit()

if __name__ == "__main__":

	# Command line options
	parser = OptionParser()
	parser.add_option("--outputDir", action="store", type="string", dest="outputDir", default=u"./outputImages/", help="Directory for reading in the output images")
	parser.add_option("--inputDir", action="store", type="string", dest="inputDir", default=u"./data/", help="Directory for reading in the input images")
	parser.add_option("--outputExtension", action="store", type="string", dest="outputExtension", default=".jpg", help="Extension of output files")
	parser.add_option("--inputExtension", action="store", type="string", dest="inputExtension", default=".jpg", help="Extension of input files")

	# Parse command line options
	(options, args) = parser.parse_args()

	visualize(options)

	print ("Done")
