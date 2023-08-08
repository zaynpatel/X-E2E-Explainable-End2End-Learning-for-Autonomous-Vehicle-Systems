import torch
import torchvision
from torch.utils.data import Dataset
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import utils
import os
import random

class CommaPreprocess(Dataset):

    def __init__(self, width, height, transform, target_transform):
        super(CommaPreprocess, self).__init__()
        self.width = width
        self.height = height
        self.transform = transform
        self.target_transform = target_transform
        image_folder = ["imgs", "imgs2"]#, "imgsd"]
        mask_folder = ["masks", "masks2"]#, "masksd"]
        image = []
        masks = []

        for (img, mask) in zip(image_folder, mask_folder):
            image.extend(glob.glob(f"comma10k/{img}/*.png"))
            masks.extend(glob.glob(f"comma10k/{mask}/*.png"))

        self.image = image
        self.masks = masks


    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):
        img = cv2.resize(cv2.cvtColor(cv2.imread(self.image[idx]),cv2.COLOR_BGR2RGB),(self.width, self.height))
        mask = cv2.resize(cv2.cvtColor(cv2.imread(self.masks[idx]),cv2.COLOR_BGR2RGB),(self.width, self.height))
        
        r = random.random()

        if r > 0.5:
            img = cv2.flip(img, 1)
            mask = cv2.flip(mask, 1)

        if self.transform:
            img = self.transform(img)

        mask = torch.from_numpy(mask).permute(2,0,1)
        mask = utils.encodeMask(mask)

        if self.target_transform:
            mask = self.target_transform(mask)

        img = utils.pad_image(img)
        mask = utils.pad_image(mask)
        return img, mask
"""

import torch
import torchvision
from torch.utils.data import Dataset
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import utils
import os
import random

class CommaPreprocess(Dataset):

    def __init__(self, width, height, transform, target_transform):
        super(CommaPreprocess, self).__init__()
        self.width = width
        self.height = height
        self.transform = transform
        self.target_transform = target_transform
        self.image = glob.glob("comma10k/imgs/*.png")[:20]
        self.masks = glob.glob("comma10k/masks/*.png")[:20]

    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):
        img = cv2.resize(cv2.cvtColor(cv2.imread(self.image[idx]),cv2.COLOR_BGR2RGB),(self.width, self.height))
        mask = cv2.resize(cv2.cvtColor(cv2.imread(self.masks[idx]),cv2.COLOR_BGR2RGB),(self.width, self.height))

        r = random.random()

        if r > 0.5:
            img = cv2.flip(img, 1)
            mask = cv2.flip(mask, 1)
        
        if self.transform:
            img = self.transform(img)

        mask = torch.from_numpy(mask).permute(2,0,1)
        mask = utils.encodeMask(mask)

        if self.target_transform:
            mask = self.target_transform(mask)

        img = utils.pad_image(img)
        mask = utils.pad_image(mask)
        return img, mask"""