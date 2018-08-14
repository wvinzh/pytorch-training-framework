from __future__ import print_function, division
import os
import torch
# import pandas as pd
from skimage import io, transform
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from imgaug import augmenters as iaa


def argument_image(input_np):
    seq = iaa.Sequential([
        # iaa.Fliplr(0.5), # horizontal flips
        # iaa.Crop(percent=(0, 0.1)), # random crops
        # Small gaussian blur with random sigma between 0 and 0.5.
        # But we only blur about 50% of all images.
        iaa.Sometimes(0.5,
            iaa.GaussianBlur(sigma=(0, 0.5))
        ),
        # Strengthen or weaken the contrast in each image.
        iaa.ContrastNormalization((0.75, 1.5)),
        # iaa.Dropout((0,0.02)),
        # Add gaussian noise.
        # For 50% of all images, we sample the noise once per pixel.
        # For the other 50% of all images, we sample the noise per pixel AND
        # channel. This can change the color (not only brightness) of the
        # pixels.
        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
        # Make some images brighter and some darker.
        # In 20% of all cases, we sample the multiplier once per channel,
        # which can end up changing the color of the images.
        iaa.Multiply((0.8, 1.2), per_channel=0.2),
        # Apply affine transformations to each image.
        # Scale/zoom them, translate/move them, rotate them and shear them.
        iaa.Affine(
            rotate=(-25, 25),
            shear=(-8, 8)
        )
    ], random_order=True)
    
    if len(input_np.shape) < 3:  #for gray images
        h,w = input_np.shape
        c=1
    else:
        h,w,c = input_np.shape

    img = input_np.reshape((1,h,w,c))
    images_aug = seq.augment_images(img)
    images_aug = images_aug.reshape((h,w,c))
    pil_img = transforms.ToPILImage()(images_aug).convert('RGB')
    return images_aug

class SensitiveCl3Dataset(Dataset):
    def __init__(self, txt_file, root_dir, transform=None, isTrain=False):
        """
        Args:
            pkl_file (string): Path to the pickle file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        with open(txt_file, 'r') as f:
            self.datas = f.readlines()
        self.root_dir = root_dir
        self.transform = transform
        self.train = isTrain
        # print(len(self.datas))

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        # print(self.datas[idx])
        line = self.datas[idx].strip()
        img_name = line.split()[0]
        label = line.split()[1]
        label = int(label)
        img_name = os.path.join(self.root_dir,img_name)

        if self.train:
            image = io.imread(img_name)
            image = argument_image(image)
            image = transforms.ToPILImage()(image).convert('RGB')
        else:
            image = Image.open(img_name).convert('RGB')
        if self.transform:
            image = self.transform(image)
            # pass
        # print(image.shape)
        return img_name,image,label


class SensitiveCl2Dataset(Dataset):
    def __init__(self, txt_file, root_dir, transform=None, isTrain=False):
        """
        Args:
            pkl_file (string): Path to the pickle file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        with open(txt_file, 'r') as f:
            self.datas = f.readlines()
        self.root_dir = root_dir
        self.transform = transform
        self.train = isTrain
        # print(len(self.datas))

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        # print(self.datas[idx])
        line = self.datas[idx].strip()
        img_name = line.split()[0]
        label = line.split()[1]
        label = int(label)
        img_name = os.path.join(self.root_dir,img_name)
        if self.train:
            image = io.imread(img_name)
            try:
                image = argument_image(image)
            except Exception as e:
                print(e,img_name)
                exit(0)
            image = transforms.ToPILImage()(image).convert('RGB')
        else:
            image = Image.open(img_name).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image,label