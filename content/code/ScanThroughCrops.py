# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 13:19:52 2018

@author: fubar
"""
import numpy as np
import matplotlib.pyplot as plt

from skimage import io, util
from skimage.morphology import disk
from skimage.util import img_as_ubyte
from skimage.filters.rank import median

from UnivariateEstimation import modal_analysis

from skimage.filters import threshold_otsu, rank
from skimage.morphology import disk, opening, closing
from skimage.feature import canny

import cv2
import random
import time
from fitEllipse1 import OverlayRANSACFit
def CropImage(image,cropsequence,displayImages):
    cropimg = util.crop(image,cropsequence)
    if displayImages == 1 :
        plt.figure(),plt.title('Cropped Image'),plt.imshow(cropimg,cmap='Greys')
    return cropimg
        
def ReadImage(filename):
    #read the image
    img = io.imread(filename ,as_grey=True)
    return img

def SaveImage(filename, img):
    #save the image
    io.imsave(filename, img)

def Img2Ubyte(image,displayImages):
    #convert to 8 bit pixel. This is for less computationally intense transformations later
    imgbyte = img_as_ubyte(image)
    if displayImages == 1 :
        plt.figure(),plt.title('8-bit Image'),plt.imshow(imgbyte,cmap='Greys')
    return imgbyte

def MedianFilter(image,displayImages):
    #median filtering
    img_med = median(image, disk(5))
    if displayImages == 1 :
        plt.figure(),plt.title('Median Filtered Image'),plt.imshow(img_med,cmap='Greys')
    return img_med

def CropSequenceGenerate(image,cropwindow):
    xcenter     = cropwindow[0][0]
    ycenter     = cropwindow[0][1]
    xwidth      = cropwindow[1][0]
    ywidth      = cropwindow[1][1]
    xrange      = image.shape[0]
    yrange      = image.shape[1]
    x1 = xcenter-xwidth/2
    x2 = xcenter + xwidth/2
    y1 = ycenter -ywidth/2
    y2 = ycenter + ywidth/2
    x1crop = int(x1)
    x2crop = int(xrange - x1crop - xwidth)
    y1crop = int(y1)
    y2crop = int(yrange -y1crop - ywidth)
    return ((x1crop, x2crop), (y1crop, y2crop))
def CustomThreshold(image,modes,displayImages):
    threshold = (modes[0]+modes[1])/2
    img_thr = image >= threshold
    if displayImages == 1 :
        plt.figure(),plt.title('Custom Thresholded Image'),plt.imshow(img_thr,cmap='Greys')
    return img_thr

def GlobalOtsuThreshold(image,displayImages):
    #global otsu thresholding
    radius = 15
    selem = disk(radius)
    threshold_global_otsu = threshold_otsu(image)
    img_otsu = image >= threshold_global_otsu
    if displayImages == 1 :
        plt.figure(),plt.title('Otsu Binarized Image'),plt.imshow(img_otsu,cmap='Greys')
    return threshold_global_otsu, img_otsu
def RemoveSpeckles(image,displayImages):
    #opening and closing to remove the speckles in background and holes in background
    radius = 5
    selem = disk(radius)
    opened = opening(image,selem=selem) 
    img_speckless = closing(opened,selem=disk(5))
    if displayImages == 1 :
        plt.figure(),plt.title('Opened and Closed Image'),plt.imshow(img_speckless,cmap='Greys')
    return img_as_ubyte(img_speckless)
def CannyEdges(image,displayImages):
    #find the canny edges
    edges = canny(image)
    if displayImages == 1 :
        plt.figure(),plt.title('Canny Edges'),plt.imshow(img_as_ubyte(edges),cmap='Greys')
    return img_as_ubyte(edges)
def CannyEdgesAndContoursOpenCV(image, nucleation_down, displayImages,debug, lower_threshold=100, upper_threshold=200):
    #find Canny edges using CV as it has advantage of removing final speckles if still present after opening and closing
    if nucleation_down == 1:
        edges = cv2.Canny(img_as_ubyte(image),lower_threshold,upper_threshold)
    else:
        edges = cv2.Canny(img_as_ubyte(np.invert(image)),lower_threshold,upper_threshold)
    #find the contours in the edges to check the number of connected componenets
    image, contours, hierarchy = cv2.findContours(edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    #contours is a list of connected contours
    n_edges = len(contours)
    if displayImages == 1 :
        title = 'Canny Edges from OpenCV with lt = ' + str(lower_threshold)+'and up = '+str(upper_threshold)
        plt.figure(),plt.title(title),plt.imshow(edges,cmap='Greys')
    if debug: print ("No of continuous edges detected " + str(n_edges))
    return edges,contours
def FitEllipse(image,iterations,displayImages,debug):
    img_overlay = image
    #fit ellipse from the edge map using random sampling of 5 points and constructing ellipse using function from openCV. Iterate this for iterations time and get a statistical data to find the inliers and outliers
    pnts = np.transpose(np.nonzero(image))
    params = np.zeros((iterations,5))
    for i,each in enumerate(params):
        #sample 5 random points and fit it to an ellipse
        sample_pnts = np.asarray(random.sample(tuple(pnts), 5))
        # Fit ellipse to points
        ellipse = cv2.fitEllipse(sample_pnts)
        if debug:print(ellipse)
        params[i][0]    =   ellipse[0][1]
        params[i][1]    =   ellipse[0][0]
        params[i][2]    =   ellipse[1][1]
        params[i][3]    =   ellipse[1][0]
        params[i][3]    =   ellipse[2]
        #draw the ellipse
        #rotate the ellipse by 90
        rot_ellipse = ((ellipse[0][1],ellipse[0][0]),(ellipse[1][1],ellipse[1][0]),ellipse[2])
        img_overlay = cv2.ellipse(img_overlay,rot_ellipse,255,1)
    return img_overlay, params
def MeanEllipse(image,params,displayImages):
    meanellipse = np.mean(params,axis=0)
    image = cv2.ellipse(image,((meanellipse[0],meanellipse[1]),(meanellipse[2],meanellipse[3]),meanellipse[4]),255,5)
    if displayImages == 1:
        plt.figure(),plt.title('Overlay of Fit Ellipses'),plt.imshow(image,cmap='Greys')
    return meanellipse
def DrawBestEllipse(image,ellipse,displayImages,debug):
    #draw the ellipse as an overlay over the roi which is img_edges
    if displayImages == 1:
        overlay = np.zeros(image.shape)
        overlay = image
        overlay =  cv2.ellipse(overlay,((ellipse[0][1],ellipse[0][0]),(ellipse[1][1],ellipse[1][0]),ellipse[2]),255,1)
        plt.figure(),plt.title('Overlay of Fit Ellipses'),plt.imshow(overlay,cmap='Greys')
def FindAdaptiveROI(image, center_ROI, aspr_ROI,displayImages, debug = True):
    """Using Kernel Density Estimate analysis, relative amount of foreground and background regions within a given ROI can be estimated.
    If the roi is varied across the viewing area, an optimum relative strength can be found when the foreground and background area are nearly equal. This sets the value of adaptive threshold.
    Since we know certain properties about our object of interest like the approximate center of object and the aspect ratio of object area, the ROI can be set accordingly"""
    #inputfilename = 'img6.png'
    #outputfilename = 'edge2.png'
    #nucleation_down = 1 # 0 for nucleation up
    #center_ROI  = (511,672)  #center of the object to be identified
    #aspr_ROI    = 2/3   # x_width/y_width for ROI. This is found by TRAINING
    #debug       = True  # flag to output ERRRORs
    #remove the strip at the bottom
    #cropsequence = ((0,44),(0,0))
    #img = ReadImage(inputfilename)
    #img = CropImage(img,cropsequence,0)
    #to mainain the aspect ratio of roi to be same as that of image, set the aspect ratio
    #asp_ratio = int(1344/(1066-44))
    #list of pad sizes to be removed along x axis
    array_x_ROI     =   np.array([100,200,300,400,500,600,700,800,902])
    array_y_ROI     =   (array_x_ROI*aspr_ROI).astype(int)
    n           =   array_x_ROI.size
    optimum_x_ROI    =  0
    optimum_y_ROI   =   0
    #set the array for relative strengths and maxima positions for the unimodal or bimodal distributions.
    array_rel_strength  =   np.zeros(n)
    array_maximum       =   np.zeros((n,2))
    displayImages = 0
    for i in np.arange(n):
        x_width = array_x_ROI[i]
        y_width = array_y_ROI[i]
        #set up the cropsequence so that pads are removed centered around the center of the image.
        cropsequence = CropSequenceGenerate(image,(center_ROI,(x_width,y_width)))
        cropimg = CropImage(image,cropsequence,0)
        imgbyte = Img2Ubyte(cropimg,0)
        img_med = MedianFilter(imgbyte,displayImages)
        maximum,rel_strength    =   modal_analysis(img_med,displayImages,debug)    #strength is zero if distribution is unimodal and close to zero if the foreground is very small compared to background or vice versa
        array_rel_strength[i]   =   rel_strength   
        array_maximum[i]        =   maximum
    if displayImages==1:
        #plot the relative strength variation  and choose the appropriate ROI
        plt.figure(),plt.title("Finding Optimum ROI by varying xROI"),plt.plot(array_x_ROI,array_rel_strength)
    #if all are unimodal distributions, then there either is no object to be found or object is beyond the ROI. This means that we need to check for bigger ROIs with progressive increase in y axis width
    max_rel_strength = np.max(array_rel_strength)
    if debug: print("maximum relative strength is " + str(max_rel_strength))
    if max_rel_strength < 0.001:
        optimum_x_ROI = 1000
    else:
        #find the optimum ROI from maximum of the relative strength vs ROI variation
        optimum_x_ROI = array_x_ROI[array_rel_strength.argsort()[-1]]
        optimum_y_ROI = array_y_ROI[array_rel_strength.argsort()[-1]]
        print("optimum_x_ROI is " + str(optimum_x_ROI))

    #if optimum ROI is less than 1000, then it probably means that the object is not occluded and search for the ROI is completed. If the ROI is not optimized then we can increase the y_width of ROI further keeping the x_width to be constant at 1022
    if optimum_x_ROI == 1000:
        array_y_ROI  = np.array([800,900,1000,1100])
        n           = array_y_ROI.size
        array_x_ROI = np.ones(n,dtype = np.int32)*902
        #set the array for relative strengths and maxima positions for the unimodal or bimodal distributions.
        array_rel_strength  =   np.zeros(n)
        array_maximum       =   np.zeros((n,2))
        displayImages = 0
        for i in np.arange(n):
            x_width = array_x_ROI[i]
            y_width = array_y_ROI[i]
            #set up the cropsequence so that pads are removed across y axis around the center of the image.
            cropsequence = CropSequenceGenerate(image,(center_ROI,(x_width,y_width)))
            cropimg = CropImage(image,cropsequence,0)
            imgbyte = Img2Ubyte(cropimg,0)
            img_med = MedianFilter(imgbyte,displayImages)
            maximum,rel_strength    =   modal_analysis(img_med,displayImages,debug)    #strength is zero if distribution is unimodal and close to zero if the foreground is very small compared to background or vice versa
            array_rel_strength[i]   =   rel_strength   
            array_maximum[i]        =   maximum
        displayImages = 1
        if displayImages == 1:
            #plot the relative strength variation  and choose the appropriate ROI
            plt.figure(),plt.title("Finding Optimum ROI by varying yROI"),plt.plot(array_y_ROI,array_rel_strength)
        max_rel_strength = np.max(array_rel_strength)
        if max_rel_strength == 0:
            optimum_x_ROI = 0
            optimum_y_ROI = 0
            if debug: print("This image needs to be discarded")
        #find the optimum ROI from maximum of the relative strength vs ROI variation
        optimum_x_ROI = array_x_ROI[array_rel_strength.argsort()[-1]]
        optimum_y_ROI = array_y_ROI[array_rel_strength.argsort()[-1]]
        if optimum_y_ROI == 1300:
            #so the whole image needs to be used for further processing
            optimum_x_ROI = 1022
            optimum_y_ROI = 1344
        #proceed with further processing with optimum ROI
    optimum_ROI = (optimum_x_ROI,optimum_y_ROI)
    if debug: print("Optimum ROI is ",optimum_ROI)
    return optimum_ROI
def FindAdaptiveROIversion2(image, center_ROI, aspr_ROI, array_ROI, displayImages, debug = True):
    """Using Kernel Density Estimate analysis, relative amount of foreground and background regions within a given ROI can be estimated.
    If the roi is varied across the viewing area, an optimum relative strength can be found when the foreground and background area are nearly equal. This sets the value of adaptive threshold.
    Since we know certain properties about our object of interest like the approximate center of object and the aspect ratio of object area, the ROI can be set accordingly"""
    #inputfilename = 'img6.png'
    #outputfilename = 'edge2.png'
    #nucleation_down = 1 # 0 for nucleation up
    #center_ROI  = (511,672)  #center of the object to be identified
    #aspr_ROI    = 2/3   # x_width/y_width for ROI. This is found by TRAINING
    #debug       = True  # flag to output ERRRORs
    #remove the strip at the bottom
    #cropsequence = ((0,44),(0,0))
    #img = ReadImage(inputfilename)
    #img = CropImage(img,cropsequence,0)
    #to mainain the aspect ratio of roi to be same as that of image, set the aspect ratio
    #asp_ratio = int(1344/(1066-44))
    #list of pad sizes to be removed along x axis
    array_x_ROI     =   array_ROI
    array_y_ROI     =   (array_x_ROI*aspr_ROI).astype(int)
    n           =   array_x_ROI.size
    optimum_x_ROI    =  0
    optimum_y_ROI   =   0
    #set the array for relative strengths and maxima positions for the unimodal or bimodal distributions.
    array_rel_strength  =   np.zeros(n)
    array_maximum       =   np.zeros((n,2))
    #displayImages = 0
    for i in np.arange(n):
        x_width = array_x_ROI[i]
        y_width = array_y_ROI[i]
        #set up the cropsequence so that pads are removed centered around the center of the image.
        cropsequence = CropSequenceGenerate(image,(center_ROI,(x_width,y_width)))
        cropimg = CropImage(image,cropsequence,0)
        imgbyte = Img2Ubyte(cropimg,0)
        img_med = MedianFilter(imgbyte,displayImages)
        maximum,rel_strength    =   modal_analysis(img_med,displayImages,debug)    #strength is zero if distribution is unimodal and close to zero if the foreground is very small compared to background or vice versa
        array_rel_strength[i]   =   rel_strength   
        array_maximum[i]        =   maximum
    #displayImages = 1
    if displayImages==1:
        #plot the relative strength variation  and choose the appropriate ROI
        plt.figure(),plt.title("Finding Optimum ROI by varying xROI"),plt.plot(array_x_ROI,array_rel_strength)
    #if all are unimodal distributions, then there either is no object to be found or object is beyond the ROI. This means that we need to check for bigger ROIs with progressive increase in y axis width
    max_rel_strength = np.max(array_rel_strength)
    if debug: print("maximum relative strength is " + str(max_rel_strength))
    if max_rel_strength < 0.001:
        optimum_x_ROI = 902
    else:
        #find the optimum ROI from maximum of the relative strength vs ROI variation
        optimum_x_ROI = array_x_ROI[array_rel_strength.argsort()[-1]]
        optimum_y_ROI = array_y_ROI[array_rel_strength.argsort()[-1]]
        #proceed with further processing with optimum ROI
    optimum_ROI = (optimum_x_ROI,optimum_y_ROI)
    if debug: print("Optimum ROI is ",optimum_ROI)
    return optimum_ROI
def OverlayFitEllipse(img_edges, confidence_parameters, new_controls, globalflags):
    """Overlay the fit ellipse, inliers and outliers and return the image"""
    #confidence parameters
    best_ellipse = confidence_parameters[0]
    pnts = confidence_parameters[1]
    norm_err = confidence_parameters[2]
    inliers = confidence_parameters[3]
    #global flags
    debug = globalflags[0]
    displayImages = globalflags[1]
    #create a color image
    img_color = cv2.merge((img_edges,img_edges,img_edges))
    if debug:print("Shape of color image is " + str(img_color.shape))
    OverlayRANSACFit(img_color, pnts, inliers, best_ellipse)
    if displayImages == 1 :
        fig,(ax1,ax2) = plt.subplots(ncols =2 ,nrows =1, figsize=(8,4))
        ax1.set_title("Normalized error of the fit")
        ax1.plot(norm_err, 'k-')
        ax2.set_title(str(new_controls))
        ax2.imshow(img_color)
    return img_color
#img = io.imread('img2.png' ,as_grey=True)
#cropsequence = CropSequenceGenerate(502,666,300,200)
#cropsequence = CropSequenceGenerate(502,666,450,300)
#cropsequence = CropSequenceGenerate(502,666,600,400)
#cropsequence = CropSequenceGenerate((502,666,542,271))

#image = img_edges
#img_overlay, params = FitEllipse(image,10,1)
#print (cropsequence)
#cropimg =CropImage(img,cropsequence,1)
#edges_cv = CannyEdgesOpenCV(img,1)