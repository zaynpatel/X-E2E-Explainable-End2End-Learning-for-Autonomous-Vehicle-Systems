import torch
import torchvision
from torch.utils.data import Dataset
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import utils
import os

class CommaPreprocess(Dataset):

    def __init__(self, width, height, transform, target_transform):
        super(CommaPreprocess, self).__init__()
        self.width = width
        self.height = height
        self.transform = transform
        self.target_transform = target_transform
        self.image = glob.glob("comma10k/imgs/*.png")
        self.masks = glob.glob("comma10k/masks/*.png")

    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):
        img = cv2.resize(cv2.cvtColor(cv2.imread(self.image[idx]),cv2.COLOR_BGR2RGB),(self.width, self.height))
        mask = cv2.resize(cv2.cvtColor(cv2.imread(self.masks[idx]),cv2.COLOR_BGR2RGB),(self.width, self.height))
        
        if self.transform:
            img = self.transform(img)

        mask = torch.from_numpy(mask).permute(2,0,1)
        mask = utils.encodeMask(mask)

        if self.target_transform:
            mask = self.target_transform(mask)

        img = utils.pad_image(img)
        mask = utils.pad_image(mask)
        return img, mask
