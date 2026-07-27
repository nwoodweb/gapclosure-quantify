
'''
WOOD-NT 25JULY2026 contact@nwoodweb.xyz
MIT LICENSE

This script is meant to train yolov11 (nano-seg or small-seg) to
quantify scratch assay gaps. 

What we found is that yolo has some severe accuracy issues for 
scratch assay identification. In particular, it would not segment
the entire gap in some validation images, and it would mark overlapping
regions as multiple gaps. Furthermore, map50 never really got past
0.8 in the best case.

==========
USER INPUT
==========

model : string

    select your yolov11 segmentation model of choise (nano,small,
    medium)

'''
import torch
import cv2
from ultralytics import YOLO

def custom_imread(filename, flags=cv2.IMREAD_COLOR):
    return cv2.imread(filename, cv2.IMREAD_COLOR)

cv2.imread = custom_imread

model = YOLO("yolo11s-seg.pt")

device_type = 0 if torch.cuda.is_available() else "cpu"
print(f"Training on device: {device_type} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

# https://docs.ultralytics.com/modes/train#train-settings
model.train(
        data = data,
        epochs = 500,
        patience = 100,
        imgsz = 1280,
        rect = False,
        optimizer = "AdamW", # https://www.sciencedirect.com/science/article/pii/S277252862500064
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.01,
        batch = 8,
        device = device_type,
        overlap_mask = True,
        mask_ratio=1,
        single_cls=True,
        hsv_v=0.2,  
        hsv_h=0.0,  
        hsv_s=0.0,
        translate = 0.1,
        fliplr=0.5,  # flip horizontally
        flipud=0.5,  # flip vertically
        degrees=0.0,
        scale = 0.0,
        shear = 0.0,
        perspective=0.0,
        mosaic=0.0,  
        mixup=0.0,  
        copy_paste=0.0,
        project="scratch_assay_training-25JULY2026-adamw-1280",
        name="yolov11-small-25JULY2026-adamw-1280",
    )
