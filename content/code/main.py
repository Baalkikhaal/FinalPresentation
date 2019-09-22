# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 13:43:39 2018

@author: fubar
"""

from ScanThroughCrops import ReadImage,CropImage,Img2Ubyte,MedianFilter,CropSequenceGenerate, GlobalOtsuThreshold,RemoveSpeckles, CannyEdgesAndContoursOpenCV, FindAdaptiveROIversion2, OverlayFitEllipse #,CustomThreshold, CannyEdges, FitEllipse, MeanEllipse, DrawBestEllipse, FindAdaptiveROI
from UnivariateEstimation import modal_analysis
import numpy as np
import matplotlib.pyplot as plt
from fitEllipse1 import ConfidenceInFit, FitEllipse_LeastSquares#, FitEllipse_RANSAC, OverlayRANSACFit,FitEllipse_RANSAC_Support,
import os,os.path
from FileProcessingUtils import extractControlsFromFolderName, extractControlsFromSequence, FitAnalysis, FilterOutExperiments, PlotTimeSequence, extractTimeSequenceExperiments, FindVelocityOnExperiments
def FindEllipseFitRANSAC(img,nucleation_down,center_ROI, aspr_ROI, array_ROI, max_norm_err_sq, displayImages,debug):
    #TO DO
    #best_ellipse = FitEllipse_RandomSampling(contours,displayImages)#now that the edges are processed, send the edge map to Fitting the ellipse using random Sampling
    #img_overlay, params = FitEllipse(img_edges,iterations,displayImages)
    #draw the mean fit on top of speckless image
    #img_fit = MeanEllipse(img_edges,params,displayImages)
    #using RANSAC algorithm for fitting the edge map to ellipse
    #best_ellipse =  FitEllipse_RANSAC_Support(pnts,roi)
    #best_ellipse =  FitEllipse_RANSAC(pnts,roi,max_itts=10, max_refines=6)
    return
def FindEllipseFitLeastSquares(img,nucleation_down,center_ROI, aspr_ROI, array_ROI, max_norm_err_sq, new_controls):
    global debug
    global displayImages
    global doNotDisplayImagesLowLevel
    #Find the optimumROI
    optimumROI = FindAdaptiveROIversion2(img, center_ROI, aspr_ROI, array_ROI, displayImages, debug)  #find optimum ROI
    ##############
    #now that the optimumROI is found, proceed with the Otsu binarization and subsequent image processing routines to find the edges of the image and fit the ellipse.
    #However sometimes it is observed the optimum ROI is not able to bound the ellipse leading to undersampling of edge points for the fitting.
    #So it is required to iterate over the array of ROI after the optimum ROI so that sufficiently large ROI is chosen so that the ellipse is bounded completely
    ##############
    #
    #Start with iteration at the optimumROI. For that first find position of optimumROI in array_ROI
    #
    position = [i for i, each in enumerate(array_ROI) if each == optimumROI[0]]
    subarray_ROI = array_ROI[position[0]:]
    n = subarray_ROI.size
    for i in np.arange(n):
        cropsequence = CropSequenceGenerate(img,(center_ROI,(subarray_ROI[i],subarray_ROI[i]*aspr_ROI)))
        if doNotDisplayImagesLowLevel : displayImages = 0
        cropimg = CropImage(img,cropsequence, displayImages)
        imgbyte = Img2Ubyte(cropimg, displayImages)
        img_med = MedianFilter(imgbyte, displayImages)
        maximum, rel_strength = modal_analysis(img_med, displayImages, debug)
        #find threshold for binarization from adaptive ROI; average of the peak positions of bimodal distribution
        threshold_adaptiveROI = maximum.mean()
        threshold_global_otsu, img_otsu = GlobalOtsuThreshold(img_med, displayImages)   #Global Otsu is better than custom thresholding
        if debug: print("Threshold from Adaptive ROI is " + str(threshold_adaptiveROI) + " and that from Global Otsu is " + str(threshold_global_otsu) )
        img_speckless = RemoveSpeckles(img_otsu,displayImages)  #remove speckles
        img_edges,contours = CannyEdgesAndContoursOpenCV(img_speckless, nucleation_down, displayImages,debug) #canny edge detection using openCV
        pnts    =   np.transpose(np.nonzero(img_edges))
        roi     =   img_speckless
        best_ellipse =  FitEllipse_LeastSquares(pnts, roi, displayImages)
        rel_center = np.int32((best_ellipse[0][0],best_ellipse[0][1]))
        axes = np.int32([best_ellipse[1][1],best_ellipse[1][0]])
        if debug: print("At iteration : " + str(i))
        if debug: print("relative center of ellipse is : " + str(rel_center))
        if debug: print("With axes : " + str(axes))
        #find the center coordinates in old coordinate system
        #what is the origin of the ROI in old coordinate system
        origin_ROI = np.array(center_ROI) - 0.5*np.array([subarray_ROI[i],subarray_ROI[i]*aspr_ROI],dtype=np.int32)
        abs_center = rel_center + origin_ROI
        if debug: print ("absoluter center of ellipse is : " + str(abs_center))
        #check if the ROI bounds the fit ellipse ; else choose the next ROI
        xwidth = subarray_ROI[i]
        ywidth = subarray_ROI[i]*aspr_ROI
        fit_xwidth = best_ellipse[1][1]
        fit_ywidth = best_ellipse[1][0]
        
        if ((center_ROI[0] + xwidth/2 > abs_center[0] + fit_xwidth/2) and (center_ROI[0] - xwidth/2 < abs_center[0] - fit_xwidth/2) and (center_ROI[1] + ywidth/2 > abs_center[1] + fit_ywidth/2) and (center_ROI[1] - ywidth/2 < abs_center[1] - fit_ywidth/2)) or i ==(n-1):
            if debug: print ("iteration is :" + str(i))
            #Find confidence of fit
            perc_inliers, inliers, norm_err = ConfidenceInFit(pnts, best_ellipse, max_norm_err_sq, debug)
            #parameters to return as an 6-channel array element
            params =np.int32([perc_inliers, abs_center[0], abs_center[1], best_ellipse[1][0], best_ellipse[1][1],best_ellipse[2]])
            displayImages = 1
            #bundle the confidence parameters to avoid sending too many parameters to overlay fit
            confidence_parameters = [best_ellipse, pnts, norm_err, inliers]
            globalflags = [debug, displayImages]
            img_color = OverlayFitEllipse(img_edges, confidence_parameters, new_controls, globalflags)
            return params, img_color
def processImage(inputfilename, new_controls):
    global debug
    global displayImages
    #Read the image file
    img = ReadImage(inputfilename)
    #############
    #Find the ellipse fit#
    #############
    params, img_color = FindEllipseFitLeastSquares(img,nucleation_down,center_ROI,aspr_ROI, array_ROI, max_norm_err_sq, new_controls)
    if debug: print ("Best Ellipse params are " + str(params))
    if displayImages: plt.figure(),plt.title("Ellipse Fit with Inliers and Outliers"),plt.imshow(img_color)
    #TO DO : saving images in tree structure
    #if saveImages: 
     #   plt.figure(),plt.title("Ellipse Fit with Inliers and Outliers"),plt.imsave(img_color)
    return params
def processPulseSequence(datadir,controls_extracted):
    global debug
    global displayImages
    #datadir = 'TestData'
    os.chdir(datadir)#level 1
    if debug : print("Starting Sequence processing at point " + str(os.getcwd())) #level 2
    sequence_dir = os.listdir()[0]
    os.chdir(sequence_dir)  #level 0
    files = os.listdir()
    files.sort()
    images = files[0:2]
    if debug : print("List of Images are \n" + str(images))
    #create six 1D arrays separately for confidence and ellipse parameter channels of size 8. The size 8 corresponds to 8 channels or colors for a particular Hop and t_pulse
    n= len(images)
    confidence = np.zeros(n)
    cx = np.zeros(n)
    cy = np.zeros(n)
    a = np.zeros(n)
    b = np.zeros(n)
    o = np.zeros(n)
    for i in np.arange(n):
        #append the pulse sequence to the controls_extracted 
        new_controls = [controls_extracted, i]
        if debug: print("Currently processing Image:  " + str(images[i]))
        params = processImage(images[i], new_controls)
        #populate the channels for each of the 1D arrays
        confidence[i]   =   params[0]
        cx[i]           =   params[1]
        cy[i]           =   params[2]
        a[i]            =   params[3]
        b[i]            =   params[4]
        o[i]            =   params[5]
    #come out of the directory Tree to original
    os.chdir("../..")
    if debug: print("Parameters for the set of pulses in this experiment are :")
    if debug: print( confidence, cx, cy, a, b, o)
    return confidence, cx, cy, a, b, o
def processExperimentSequence(expdir):
    global debug
    global displayImages
    if debug :print("Starting Experiment processing at location " + str(os.getcwd())) #level 2
    os.chdir(expdir)#level 1
    files = os.listdir()
    dirs = [each for each in files if os.path.isdir(each)]
    n= len(dirs)  #number of experiments done
    if debug : print("Number of experiments done are: " + str(n))
    #create 2D arrays separately for confidence and ellipse parameter channels of size (nx8)
    experiments_confidence  =   np.zeros((n,2))
    experiments_cx          =   np.zeros((n,2))
    experiments_cy          =   np.zeros((n,2))
    experiments_a          =   np.zeros((n,2))
    experiments_b          =   np.zeros((n,2))
    experiments_o          =   np.zeros((n,2))
    #also create (nx3) 2D array for storing Hop, t_pulse, Hip. These values need to be extracted from the foldername.
    controls                =   np.zeros((n,3),dtype=np.int32)
    for i in np.arange(n):
        #find (Hop,t_pulse,Hip) point in control space
        control_space_point = extractControlsFromFolderName(dirs[i],debug)
        controls[i]                 =   control_space_point
        #enter the experiment
        result_confidence, result_cx, result_cy, result_a, result_b, result_o = processPulseSequence(dirs[i],controls[i])
        experiments_confidence[i]   =   result_confidence
        experiments_cx[i]           =   result_cx
        experiments_cy[i]           =   result_cy
        experiments_a[i]            =   result_a
        experiments_b[i]            =   result_b
        experiments_o[i]            =   result_o

    #come out of the experiment directory
    os.chdir("..")
    if debug: print (experiments_confidence, experiments_cx, experiments_cy, experiments_a, experiments_b, experiments_o)
    return (experiments_confidence, experiments_cx, experiments_cy, experiments_a, experiments_b, experiments_o), controls
def FindParameters(exp_seq_filename,rawdir):
    global debug
    global displayImages
    #exp_seq_filename = 'TestSequence.csv' # .csv file which sets up the experiment sequence for DMI measurement
    #setup controls array from the experiment sequence generator file
    if debug : print("Starting parameter extraction for Raw Data at " + str(rawdir) + "using Experiment Sequence file " + str(exp_seq_filename))
    control_knobs = extractControlsFromSequence(exp_seq_filename,debug)
    #extract the parameters from experiments
    os.chdir(rawdir)
    experiment_dir = os.listdir()[0]
    experiments, controls_extracted = processExperimentSequence(experiment_dir)
    if debug: print("Confidence of fit for the experiments is \n" + str(experiments[0]))
    #return to the parent dir
    os.chdir("..")
    #extract the parameters from experiments
    return experiments, controls_extracted, control_knobs
def main():
    global experiments
    global controls_extracted
    global control_knobs
    global debug
    global displayImages
    global filtered_experiments
    global min_confidence
    global exp_seq_filename
    global raw_dir
    global velocity_center
    global velocity_rel
    experiments, controls_extracted, control_knobs = FindParameters(exp_seq_filename,raw_dir)
    print("Control knobs of the experiements are [Hip Hop tp] :")
    print(control_knobs)
    print("Controls extracted are :")
    print(controls_extracted)
    print("Experiments are :")
    print(str(experiments))
    #Find the goodfits_indices and badfits_indices based on a chosen confidence in the fit
    goodfits, badfits, goodfits_indices, badfits_indices = FitAnalysis(experiments, min_confidence)
    #Filter the experiments
    #filtered_experiments = FilterOutExperiments(experiments, goodfits_indices, badfits_indices, min_confidence)
    velocity_center, velocity_rel = FindVelocityOnExperiments(experiments, goodfits, controls_extracted, debug=True)
    #subspace_experiments,subspace_experiments_indices = extractTimeSequenceExperiments(filtered_experiments, controls_extracted, Hip, Hop,debug=True)
    #PlotTimeSequence(subspace_experiments,debug=True)
    #AverageTimeSequenceData()

###################################
#############Main##################
###################################

####################
##global variables##
####################
######training parameters##
nucleation_down =   1 # 0 for nucleation up
center_ROI      =   (511,645)  #center of the object to be identified
aspr_ROI        =   8/9   # x_width/y_width for ROI. This is found by TRAINING
array_ROI       =   np.array([100,200,300,400,500,600,700,800,902])
min_confidence = 40.0   #minimum confidence for fit to be considered OK
max_norm_err_sq =   20.0    # maximum allowed normalized error allowed for a point to be an inlier
#####################
##raw data parameters
#####################
exp_seq_filename = '-59.csv'
raw_dir = 'G:\Banibrato\domain_wall_dynamics\PostProcessing\-59'
#########################
######debug parameters###
#########################
debug = False
displayImages = 0
doNotDisplayImagesLowLevel = 1
#parameters to be extracted from analysis
experiments             = []
controls_extracted    = []
control_knobs           = []
filtered_experiments = []
velocity_center = []
velocity_rel = []
main()
#processImage('img4.png',displayImages=1,debug=True)

