import os
import pickle
import matplotlib.pyplot as plt

from glob import glob
from math import floor
from lib import generateData as gen, constants


def main():
	images = sorted([y for x in os.walk('./SynthEyes_data/') for y in glob(os.path.join(x[0], '*.png'))])
	landmarks = sorted([y for x in os.walk('./SynthEyes_data/') for y in glob(os.path.join(x[0], '*.pkl'))])


	for image,landmark in zip(images,landmarks):
		# #open the landmarks for the image 
		markings = pickle.load(open(landmark,'rb'))

		iris_coordinates = gen.get_coordinates("iris", markings)
		lids_coordinates = gen.get_coordinates("lids", markings)
		pupil_coordinates = gen.get_coordinates("pupil", markings)
		
		irisMask = gen.maskSpline(coordinates = iris_coordinates)
		pupilMask = gen.maskSpline(coordinates = pupil_coordinates)
		lidsMask = gen.maskSpline(coordinates = lids_coordinates)
		
		segmented = gen.segmentIris(pupilMask, irisMask, lidsMask)
		plt.imshow(segmented, cmap='gray')
		plt.show()
		break

if __name__ == '__main__':
	main()