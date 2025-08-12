'''
2025 WOOD-NT contact@nwoodweb.xyz
MIT LICENSE 

This script is meant to automate the quantification of changes
in gap size during a gap closure/wound healing/scratch assay.
It uses image processing provided by scikit image and 
multithreading by JobLib.

RETURNS
-------

gapclosure-quantify-<date_experiment>.csv : CSV ASCII text file
    CSV file containing processed data

USER DEFINED PARAMETERS
-----------------------

data_directory: string, default="./"
    directory of which the output CSV file will be placed

date_experiment: string, default="11-11-1111"
    date of when experiment started
    recommended format: DD-MM-YYYY 

disk_size: u_int, default=25
    neighborhood radius for entropy filter

hole_size: u_int, default=500
    otsu segmentation may have holes inside the "true" region
    representing migrating cells, hole_size is the maximum
    hole size that the scikit-image parameter fill_hole()
    should fill

input_directory: string, default="./tiffs"
    file directory where images to be analyzed will occur

iteration:
    defines the replicate that the image file is from
    This is dependent on the naming system used by the user,
    so will need to be determined based on position. 
    example: iteration = img[2:6]

number_parallel_jobs: u_int, default=4
    defines number of CPU cores requested for JobLib multithreading
    genuinely not sure what happens if you specify something silly 
    like 32768, but you can try :)
    
object_size: u_int, default=128
    defines minimum object size for scikit-image parameter
    remove_small_object()

ph: sliced string
    defines the experimental condition (in my case pH) of the image
    This is dependent on the naming system used by the user,
    so will need to be determined based on position. 
    example: ph = img[2:6]

time_hour: sliced string
    defines the time point of the image
    This is dependent on the naming system used by the user,
    so will need to be determined based on position. 
    example: time_hour = img[2:6]

'''

import os
import pandas as pd
import numpy as np
from tifffile import imread
from glob import glob
from joblib import Parallel, delayed
from skimage import data
from skimage.util import img_as_ubyte
from skimage.filters.rank import entropy
from skimage.filters import threshold_otsu, gaussian
from skimage.morphology import disk, remove_small_holes, remove_small_objects

# user defined parameters: file io
input_directory = os.path.expanduser("./tiffs/")
data_directory = os.path.expanduser("./")
date_experiment = "11-11-1111"

# user defined parameters: image analysis
disk_size = 25
hole_size = 1000
numberParallelJobs = 4
object_size = 500

output_file = "gapclosure-quantify-" + date_experiment + ".csv"
output_file = os.path.join(data_directory, output_file)

# for our 10X objective: um_width = um_length, 0.8850341
# for our 4X objective: um_width = um_length, 0.3540199

pixel_um_width = 0.3540199
pixel_um_length = pixel_um_width
pixel_um_square = pixel_um_width * pixel_um_length

data = []


def process_image(img):

    '''
    MORE USER DEFINED PARAMETERS
    these three parameters are derived from the filename, you will
    need to specify them based on how you do you file organization
    and nomenclature
    '''
    # strip pH from filename
    ph = img[18:19]

    # strip iteration from filename
    iteration = img[23:24]

    # strip timepoint from filename
    time_hour = img[32:34]

    # load image and convert to 8 bit unsigned grayscale
    image = imread(img)
    image = img_as_ubyte(image)

    # extract pixel area of image
    image_size_pixel = image.size

    # convert pixel size to um^2
    image_size_um = image_size_pixel * pixel_um_square

    # gaussian blur improves segmentation quality
    # maybe experiment with low pass filters in future
    image_gaussian_blur = gaussian(image, sigma=2)
    
    # entropy filter
    entropy_filter_image = entropy(image_gaussian_blur,
                                   disk(disk_size))

    # otsu segmentation
    otsu_thresh = threshold_otsu(entropy_filter_image)
    otsu_binary = entropy_filter_image <= otsu_thresh

    # remove holes 
    otsu_binary = remove_small_holes(otsu_binary,
                                     area_threshold=hole_size)

    # remove small objects, not very effective
    otsu_binary = remove_small_objects(otsu_binary,
                                       min_size=object_size)

    # extract pixel area of scratch area
    area_gap_pixel = np.sum(otsu_binary == True)

    # convert pixel area of scratch to um^2
    area_gap_um = areaScratch_pixel * pixel_um_square

    # convert to percent closure
    percent_closure = 100 * (area_gap_um / image_size_um)

    area_cells_um = image_size_um - area_gap_um

    # organize output
    data = [date_experiment, img, pH, iteration, time_hour,
            area_gap_um, area_cells_um, percent_closure]

    return data


'''
This monstrosity is the for joblib multithreading.

Thanks to: https://stackoverflow.com/questions/42220458/what-does-the-delayed-function-do-when-used-with-joblib-in-python
'''
dataframe = pd.DataFrame(Parallel(n_jobs=numberParallelJobs)(delayed(process_image)(img)for img in sorted(glob(os.path.join(input_directory, "*.tif")))))

# add headers to output dataframe
dataframe_headers = ['date','file','pH','iteration','timePoint',
                     'area_scratch_um','area_cells_um',
                     'percentClosure']
dataframe.columns = dataframe_headers

# write to csv
dataframe.to_csv(output_file, sep=',', encoding='utf-8')
