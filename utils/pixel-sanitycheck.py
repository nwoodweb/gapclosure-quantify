'''
WOOD-NT 27JULY2026 contact@nwoodweb.xyz
MIT LICENSE

This script is meant to diagnose what
pixel value is the segmented image,
for whichever reason my masks were at 109

==========
USER INPUT
==========

mask_path: string
    filepath to image var validation 

'''

import cv2
import numpy as np

# Path to one of your real mask images
mask_path = "/scratch/user/woodn/gapclosure-quantify/neuralnet/unet/dataset/val/masks/ph7-c-01_TRANS_TL0002_41-1x1.tif-validation-8bit.png"
raw_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

print(f"Mask Shape: {raw_mask.shape}")
print(f"Unique pixel values in mask: {np.unique(raw_mask)}")
print(f"Ratio of pixels < 128: {(raw_mask < 128).mean():.4f}")
print(f"Ratio of pixels > 127: {(raw_mask > 127).mean():.4f}")
