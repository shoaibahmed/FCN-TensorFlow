import os
import random
import shutil
from os import listdir
from os.path import isfile, join
from optparse import OptionParser

def traverseDirectory(options):
	imagesFileTrain = open(options.imagesTrainOutputFile, 'w')
	imagesFileTest = open(options.imagesTestOutputFile, 'w')

	counter = 1
	keysCounter = 1
	for root, dirs, files in os.walk(options.rootDirectory):
		path = root.split('/')
		print ("Directory:", os.path.basename(root))
		
		for file in files:
			if file.endswith(options.searchString):
				# Check if the file isn't a mask
				if "mask" not in file:
					fileName = str(os.path.abspath(os.path.join(root, file)))

					# Specify train test split using random number
					prob = random.random()
					if prob < 0.75:				# Train sample (Prob: 0.75)
						imagesFileTrain.write(fileName + '\n')
					else:					# Validation sample (Prob: 0.25)
						imagesFileTest.write(fileName + '\n')

	imagesFileTrain.close()
	imagesFileTest.close()

if __name__ == "__main__":

	# Command line options
	parser = OptionParser()
	parser.add_option("-d", "--dir", action="store", type="string", dest="rootDirectory", default=u".", help="Root directory to be searched")
	parser.add_option("--searchString", action="store", type="string", dest="searchString", default=".png", help="Criteria for finding relevant files")
	parser.add_option("--imagesTrainOutputFile", action="store", type="string", dest="imagesTrainOutputFile", default="train.idl", help="Name of train images file")
	parser.add_option("--imagesTestOutputFile", action="store", type="string", dest="imagesTestOutputFile", default="test.idl", help="Name of test images file")

	# Parse command line options
	(options, args) = parser.parse_args()

	traverseDirectory(options)

	print ("Done")
