import numpy as np 
import matplotlib.pyplot as plt

from math import floor
from scipy import interpolate
from PIL import Image, ImageDraw
from scipy.interpolate import splprep, splev

from lib.constants import *

# Generates a image and draws a polygon with the given co-ordinates
def maskpolygon(coordinates):
	# Get an image with L mode: 8-bit pixels, black and white 
	# of desired resolution
	img = Image.new('L', (IMAGE_HEIGHT, IMAGE_WIDTH), 0)
	ImageDraw.Draw(img).polygon(coordinates, outline=1, fill=1)
	mask = np.array(img)
	return mask

# used to generate a Spline  around given polygon co-ordinates
def maskSpline(coordinates):
	# Get an image with L mode: 8-bit pixels, black and white 
	# of desired resolution
	img = Image.new('L', (IMAGE_WIDTH, IMAGE_HEIGHT), 0)
	mask = np.array(img)

	# Split x,y co-ordinates into lists
	x_coordinates = [ x[0] for x in coordinates ]
	y_coordinates = [ x[1] for x in coordinates ]
	points = [ x_coordinates, y_coordinates ]

	# Generate a spline curve fitting the points provided
	tck, u = splprep(points, u=None, s=0.0, per=1) 
	u_new = np.linspace(u.min(), u.max(), 1000)

	# Generate pixels for spline region
	x, y = splev(u_new, tck, der=0)
	x = x.astype(int)
	y = y.astype(int)

	# Exception handling for spline going out of bounds
	x[x >= IMAGE_HEIGHT] = IMAGE_HEIGHT-1
	y[y >= IMAGE_WIDTH] = IMAGE_WIDTH-1
	

	pts = np.array([x,y]).T

	# mark boundary by value 1 in grayscale
	for i in range(len(x)):
		mask[x[i]][y[i]] = 1


	# fill mask using scanline  algorithm
	for i in range(80):
		itemindex = np.where(mask[i]==1)
		if len(itemindex[0])>1:
			ind = np.array(itemindex[0])
			#  make all 1's between boundaries
			mask[i][ind[0]:ind[-1]+1]=1
	return mask
	
# return orginal image masked with mask generated
def apply_segment(im,mask):
	
	im[:,:,0] *= mask
	im[:,:,1] *= mask
	im[:,:,2] *= mask
	return im

# Segments Iris given masks or pupil, iris, lids
def segmentIris(pupilMask, irisMask, lidsMask):
	if pupilMask.shape == irisMask.shape ==\
		lidsMask.shape == (IMAGE_HEIGHT, IMAGE_WIDTH):
		return (lidsMask*(irisMask-pupilMask))
	return None

# Given pickel file markings and an eye region
# the function returns co-ordinates of that region
def get_coordinates(region, markings):
	key = 'ldmks_{}_2d'
	if region in ["iris", "pupil", "lids"]:
			marks = markings['ldmks'][key.format(region)]
			return get_formatted_coordinates(marks)

# Given co-ordinates of a region returns
# data list in co-ordinate format
def get_formatted_coordinates(markings):
	data=[]
	for landmark in markings:
		y,x = map(int, landmark)
		data.append((x, y))
	return data



# The function takes input of the eye shape landmarks of an image and 
# Returns a iris mask
def genrate_iris_mask(markings):
	# Extact iris, pupil, lids coordinates
	iris_coordinates = get_coordinates("iris", markings)
	lids_coordinates = get_coordinates("lids", markings)
	pupil_coordinates = get_coordinates("pupil", markings)
	
	# Generate masks for each part
	irisMask = maskSpline(coordinates = iris_coordinates)
	pupilMask = maskSpline(coordinates = pupil_coordinates)
	lidsMask = maskSpline(coordinates = lids_coordinates)
	
	# Segment iris region
	segmented = segmentIris(pupilMask, irisMask, lidsMask)
	return segmented	
