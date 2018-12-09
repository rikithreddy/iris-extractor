import cv2
import numpy as np 
import matplotlib.pyplot as plt
import pickle
import scipy
import numpy
from PIL import Image, ImageDraw
from scipy import interpolate
from scipy.interpolate import splprep, splev
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure


nt = np.linspace(0, 1, 100)


# used to generate a polygon  around given polygon co-ordinates

def maskpolygon(width,height,polygon):
    
    img = Image.new('L', (width, height), 0)
#    print polygon
    ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)
    mask = np.array(img)
    return mask

# used to generate a Spline  around given polygon co-ordinates

def maskSpline(width, height, polygon,num=1):
    pts = np.array(polygon) 
    tck, u = splprep(pts.T, u=None, s=0.0, per=1) 
    u_new = np.linspace(u.min(), u.max(), 1000)
    x_new, y_new = splev(u_new, tck, der=0)
    img = Image.new('L', (width, height), 0)
    mask= np.array(img)

    # mark boundary by value 1 in grayscale

    for i in xrange(len(x_new)):
            mask[int(x_new[i])][int(y_new[i])]=num

    # fill mask using scanline  algorithm

    for i in xrange(80):
        itemindex = np.where(mask[i]==num)
        if len(itemindex[0])>1:
            ind = np.array(itemindex[0])
            #  make all 1's between boundaries
            mask[i][ind[0]:ind[-1]+1]=num
    return mask
    

# return orginal image masked with mask generated
def returnAnd(im,mask):
    tem=im
    tem[:,:,0] = im[:,:,0] * mask
    tem[:,:,1] = im[:,:,1] * mask
    tem[:,:,2] = im[:,:,2] * mask
    return tem





height = 80 
width = 120


import os
from glob import glob
images = sorted([y for x in os.walk('./SynthEyes_data/') for y in glob(os.path.join(x[0], '*.png'))])
landmarks = sorted([y for x in os.walk('./SynthEyes_data/') for y in glob(os.path.join(x[0], '*.pkl'))])

print landmarks[0:5]
print '######################################################'
print images[0:5]
print '######################################################'


for x,y in zip(images,landmarks):
    print x,y
    # #open the landmarks for the image 
    x = pickle.load(open(y,'rb'))
    y = x['ldmks']['ldmks_iris_2d']
    k=[]
    for  l in y:
        k.append((int(l[1]),int(l[0])))
    maskIris = maskSpline(height=height, width=width, polygon=k, num = 1)
    print maskIris
    y = x['ldmks']['ldmks_lids_2d']
    t =[]
    for  l in y:
        t.append((int(l[1]),int(l[0])))
    #############################################
    masklids = maskSpline(width=width,height=height,polygon=t,num = 3)
    plt.imshow(masklids,cmap='gray')
    plt.show()
    y = x['ldmks']['ldmks_pupil_2d']
    t =[]
    for  l in y:
        t.append((int(l[1]),int(l[0])))
    ######################################
    maskPupil = maskSpline(width=width,height=height,polygon=t,num = 2)
    plt.imshow(maskPupil,cmap='gray')
    plt.show()
