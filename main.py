import os
import pickle
import matplotlib.pyplot as plt


from glob import glob
from math import floor
from lib import generateData, constants

def main():
	images = sorted([y for x in os.walk('./SynthEyes_data/') for y in glob(os.path.join(x[0], '*.png'))])
	landmarks = sorted([y for x in os.walk('./SynthEyes_data/') for y in glob(os.path.join(x[0], '*.pkl'))])


	for x,y in zip(images,landmarks):
		# #open the landmarks for the image 
		x = pickle.load(open(y,'rb'))
		y = x['ldmks']['ldmks_iris_2d']
		k=[]
		for  l in y:
			k.append((floor(l[1]),floor(l[0])))

		irisMask = generateData.maskSpline(coordinates = k)
		
		y = x['ldmks']['ldmks_lids_2d']
		t =[]
		for  l in y:
			t.append((floor(l[1]),floor(l[0])))

		lidsMask = generateData.maskSpline(coordinates = t)
		y = x['ldmks']['ldmks_pupil_2d']
		t =[]
		for  l in y:
			t.append((floor(l[1]),floor(l[0])))
		pupilMask = generateData.maskSpline(coordinates = t)

		segmented = generateData.segmentIris(pupilMask, irisMask, lidsMask)
		plt.imshow(segmented, cmap='gray')
		plt.show()
		break

if __name__ == '__main__':
	main()