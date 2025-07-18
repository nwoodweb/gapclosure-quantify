# WOOD-NT (contact@nwoodweb.xyz)
# quantifies cell migration by measuring changes in  unoccupied gap
# size using entropy filter
# MIT LICENSE

import os
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tifffile import imread
from glob import glob
from joblib import Parallel, delayed    

import skimage as sk
import skimage.io
from skimage import data,color,restoration
from skimage.util import img_as_ubyte
from skimage.filters.rank import entropy,otsu
from skimage.exposure import equalize_hist
from skimage.morphology import disk,remove_small_holes,remove_small_objects
from skimage.filters import threshold_otsu, gaussian

imageDirectory = os.path.expanduser("./tiffs/analyze/")
saveDirectory = os.path.expanduser("./data")
saveReadsCSV = os.path.join(saveDirectory,"scratchAssay.csv") 

# specifies number of CPU cores to run on
numberParallelJobs = 8

# for our 10X objective: um_width = um_length, 0.8850341
# for our 4X objective: um_width = um_length, 0.3540199

pixel_um_width = 0.3540199
pixel_um_length = pixel_um_width
pixel_um_square = pixel_um_width * pixel_um_length
dateStart = "12OCT2024"

data = []

def process_image(img):

    # strip pH from filename
    pH = img[18:19]
#    print(pH)   
    # cariporide or vehicle
    drug = img[21]
    # strip iteration from filename
    iteration = img[23:24]
#    print(iteration)    
    # strip timepoint from filename 
    timeHour = img[33:35]
#    print(timeHour)    
    # load image and convert to 8 bit unsigned grayscale
    image = imread(img)
    image = img_as_ubyte(image) 
    
    # extract pixel area of image
    image_size_pixel = image.size

    # convert pixel size to um^2
    image_size_um = image_size_pixel * pixel_um_square
    
    # 1. entropy filter, 2. otsu segment, 3. binarize
    image = gaussian(image,sigma=2)
    entropyFilteredImage = entropy(image,disk(25))
 #   otsuImage = otsu(entropyFilteredImage,40)
    otsuImage = threshold_otsu(entropyFilteredImage) 
    otsuBinary = entropyFilteredImage <= otsuImage
    otsuBinary = remove_small_holes(otsuBinary,area_threshold=750)
    otsuBinary = remove_small_objects(otsuBinary, min_size = 64)
    # extract pixel area of scratch area
    areaScratch_pixel = np.sum(otsuBinary == True)
    # convert pixel area of scratch to um^2
    areaScratch_um = areaScratch_pixel * pixel_um_square
    # convert to percent closure
    percentClosure = 100 * (areaScratch_um / image_size_um)
    areaCells_um = image_size_um - areaScratch_um
    # organize output
    data = [dateStart,img,pH,drug,iteration,timeHour,areaScratch_um,areaCells_um,percentClosure]
    '''
    plt.imshow(otsuBinary)
    plt.title(img)
    plt.show()
    '''
    return data 

# EXECUTE IN PARALLEL, CONSOLIDATE RETURNED DATA INTO DATAFRAME IMMEDIATELY !!!
dataFrame = pd.DataFrame(Parallel(n_jobs=numberParallelJobs)(delayed(process_image)(img) for img in sorted(glob(os.path.join(imageDirectory,"*.tif")))))

# WRITE TO CSV
dataFrame.to_csv(saveReadsCSV, sep = '\t', encoding = 'utf-8') 
