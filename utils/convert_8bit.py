'''
WOOD-NT 20JULY2026 contact@nwoodweb.xyz
MIT LICENSE

This script takes input directory of presumably TIF images, 
crops out some bizarre misalignment found on our Echo Revolution,
then normalizes, before converting to 8 bit via CV2 and saving
as 8bit tif

'''

import glob
import os
import cv2

input_directory = os.path.expanduser("./training/images/")
input_directory = os.path.join(input_directory, "*.tif")

output_directory = os.path.expanduser("./training/images-8bit/")

crop_px = 10

for image in glob.glob(input_directory):
    img = cv2.imread(image, cv2.IMREAD_UNCHANGED)

    h, w = img.shape[:2]
    img_cropped = img[crop_px : h - crop_px, crop_px : w - crop_px]

    img_cropped = cv2.normalize(img_cropped, None, 0, 255,
                                cv2.NORM_MINMAX)
    
    img_cropped = img_cropped.astype('uint8')

    output_filename = image + "-8bit.tif"

    cv2.imwrite(output_filename, img_cropped)
