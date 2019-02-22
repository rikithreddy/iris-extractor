import os
import pickle
import matplotlib.pyplot as plt
import cv2

from glob import glob
from math import floor
from lib import generateData as gen, constants



def main():
	images = sorted([y for x in os.walk('./SynthEyes_data/') for y in glob(os.path.join(x[0], '*.png'))])
	landmarks = sorted([y for x in os.walk('./SynthEyes_data/') for y in glob(os.path.join(x[0], '*.pkl'))])


	for image,landmark in zip(images,landmarks):
		# #open the landmarks for the image
		img = cv2.imread(image)
		markings = pickle.load(open(landmark,'rb'))
		segmented = gen.genrate_iris_mask(markings)

		segmented = gen.returnAnd(img, segmented)
		plt.imshow(segmented, cmap='gray')
		plt.show()
		break

if __name__ == '__main__':
	main()