
'''
WOOD-NT 26JULY2026 contact@nwoodweb.xyz
MIT LICENSE

This script is meant to train a UNET architecture
on scratch assay data. I presume that while annoying
to get started, will yield better accuracy than traditional
otsu segmentation, especially against poor illumination,
cell debris, and partically closed gaps.

https://www.youtube.com/watch?v=azM57JuQpQI&list=PLZsOBAyNTZwbR08R959iCvYT3qzhxvGOE&index=1

===========
USER INPUTS
===========

epoch_number : integer

    sets number of training iterations, or epochs.
    3 epochs is a good number for debugging.

patience_number : integer

    sets number of epochs to persist before cancelling
    when there are diminishing returns on training loop,
    saves valuble SU on a cluster

batch_size : integer

worker_number : integer


'''


import os
import glob
import numpy as np
import cv2
import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import albumentations
from albumentations.pytorch import ToTensorV2


epoch_number = 250
patience_number = 25
batch_size = 16
worker_number = 2

# cuda autodetection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class scratchassaydataset(Dataset):
    def __init__(self, image_directory, mask_directory, transform  = None):
        self.image_directory = image_directory
        self.mask_directory = mask_directory
        self.transform = transform
        self.image_names = sorted([f for f in os.listdir(image_directory) if f.endswith(('.jpg', '.png', '.tif'))])

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        base = os.path.splitext(image_name)[0]

        image_path = os.path.join(self.image_directory, image_name)
        mask_path = os.path.join(self.mask_directory, f"{base}.png")

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        mask = (mask > 50).astype(np.float32)

        if self.transform:
            augmented = self.transform(image = image, mask = mask)
            image = augmented['image']
            mask = augmented['mask']

        return image, mask

desired_image_width, desired_image_height = 1280, 1024

train_transform = albumentations.Compose([
    albumentations.Resize(height = desired_image_height,
        width = desired_image_width),
    albumentations.HorizontalFlip(p = 0.5),
    albumentations.VerticalFlip(p = 0.5),
    albumentations.RandomBrightnessContrast(p = 0.2),
    albumentations.Normalize(mean = (0.485, 0.456, 0.406),
        std = (0.229, 0.224, 0.225)),
    ToTensorV2(),
    ]
    )

validation_transform = albumentations.Compose([
    albumentations.Resize(height = desired_image_height,
        width = desired_image_width),
    albumentations.Normalize(mean = (0.485, 0.456, 0.406),
        std = (0.229, 0.224, 0.225)),
    ToTensorV2(),
    ]
    )

def train():
    training_dataset = scratchassaydataset("/scratch/user/woodn/gapclosure-quantify/neuralnet/unet/dataset/train/images",
            "/scratch/user/woodn/gapclosure-quantify/neuralnet/unet/dataset/train/masks",
            transform = train_transform)
    validation_dataset = scratchassaydataset("/scratch/user/woodn/gapclosure-quantify/neuralnet/unet/dataset/val/images",
            "/scratch/user/woodn/gapclosure-quantify/neuralnet/unet/dataset/val/masks",
            transform = validation_transform)

    training_loader = DataLoader(training_dataset,
            batch_size = batch_size,
            shuffle = True,
            num_workers = worker_number,
            pin_memory = True)
    validation_loader = DataLoader(validation_dataset,
            batch_size = batch_size,
            shuffle = False,
            num_workers = worker_number)

    model = smp.Unet(encoder_name = "efficientnet-b3",
            encoder_weights = "imagenet",
            in_channels = 3,
            classes = 1).to(device)

    dice_loss = smp.losses.DiceLoss(mode = "binary",
            from_logits = True)
    bce_loss = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(model.parameters(),
            lr = 0.0001,
            weight_decay = 0.01)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
            mode = 'max',
            patience = patience_number,
            factor = 0.5)

    best_iou = 0.0

    scaler = torch.amp.GradScaler("cuda")

    for epoch in range(1, epoch_number + 1):
        model.train()
        training_loss = 0.0


        for images, masks in training_loader:
            images, masks = images.to(device), masks.to(device)
            if masks.ndim == 3:
                masks = masks.unsqueeze(1)

            optimizer.zero_grad()
            
            with torch.amp.autocast("cuda", enabled = (device.type == "cuda")):
                outputs = model(images)


            loss = dice_loss(outputs.float(),
                        masks.float()) + bce_loss(outputs.float(),
                                masks.float())

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            #outputs = model(images)

            '''
            loss = dice_loss(outputs, masks) + bce_loss(outputs, masks)
            loss.backward()
            optimizer.step()
            ''' 

            training_loss += loss.item()
        model.eval()
        validation_loss, total_iou = 0.0, 0.0
        tp_list, fp_list, fn_list, tn_list = [], [], [], []       

        with torch.no_grad():
            for images, masks in validation_loader:
                images, masks = images.to(device), masks.to(device)

                if masks.ndim == 3:
                    masks = masks.unsqueeze(1)

                with torch.amp.autocast("cuda", enabled = (device.type == "cuda")):
                    outputs = model(images)
                    loss = dice_loss(outputs.float(), masks.float()) + bce_loss(outputs.float(), masks.float())

                validation_loss += loss.item()

                predictions = torch.sigmoid(outputs)

                tp, fp, fn, tn = smp.metrics.get_stats(predictions.float(),
                        masks.long(),
                        mode = 'binary',
                        threshold = 0.5)
                
                tp_list.append(tp)
                fp_list.append(fp)
                fn_list.append(fn)
                tn_list.append(tn)

                '''
                pred = (predictions > 0.5)           
                pmask = (pred[0].squeeze()*255).byte().cpu().numpy()
                cv2.imwrite("pred.png",pmask)
                '''

        average_training_loss = training_loss / len(training_loader)
        average_validation_loss = validation_loss / len(validation_loader)
        

        total_tp = torch.cat(tp_list).sum()
        total_fp = torch.cat(fp_list).sum()
        total_fn = torch.cat(fn_list).sum()
        
        average_iou = (total_tp / (total_tp + total_fp + total_fn + 1e-7)).item()
        scheduler.step(average_iou)

        print(
                f"Epoch {epoch:03d} | Train Loss: {average_training_loss:.4f} | "
                f"Val Loss: {average_validation_loss:.4f} | Val IoU: {average_iou:.4f}" 
        )

        if average_iou > best_iou:
            best_iou = average_iou
            torch.save(model.state_dict(),
                "best-unet-efficient-26july26.pth")
            print(f"IoU: {best_iou:.4f})")

if __name__ == "__main__":
    train()
