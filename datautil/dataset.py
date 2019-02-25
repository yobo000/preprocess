# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import sys

import pandas as pd
from PIL import Image


import warnings
warnings.filterwarnings("ignore")

try:
    from torch.utils.data import Dataset
except ImportError as e:
    Dataset = object

if sys.version_info[0] == 3:
    xrange = range
else:
    pass


class CategoriesDataset(object):
    """Categories Landmarks datase for beauty, fashion, mobile"""

    def __init__(self, type, stage):
        self.type = type
        self.stage = stage

    def __call__(self, *args):
        args = list(args)
        if len(args) == 2:
            args.extend([None, self.stage])
        else:
            args.append(self.stage)
        if self.type == "image":
            return ImagesDataset(*args)
        elif self.type == "text":
            return TextDataset(*args)

    def __repr__(self):
        fmt_str = "This class will return a {0} dataset of {1}".format(
            self.stage, self.type)
        return fmt_str


class ImagesDataset(Dataset):
    """Categories Image Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None, stage="train"):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.train = True if stage == "train" else False

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if self.train:
            img_path = os.path.join(self.root_dir,
                                    self.landmarks_frame.iloc[idx, 3])
            with open(img_path, 'rb') as f:
                image = Image.open(f)
            # image = Image.fromarray(img_path)
            landmarks = self.landmarks_frame.iloc[idx, 2].astype('int')
            # landmarks = [1 if landmarks == i else 0 for i in range(1, 58)]
            sample = {'image': image, 'landmarks': landmarks}
        else:
            img_path = os.path.join(self.root_dir,
                                    self.landmarks_frame.iloc[idx, 2])
            image = Image.fromarray(img_path)
            landmarks = landmarks
            sample = {'image': image}

        if self.transform:
            sample = self.transform(sample)

        return sample


class TextDataset(Dataset):
    """Categories Text Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None, stage="train"):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.stage = stage

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 3])
        image = Image.fromarray(img_name)
        if self.stage == "train":
            landmarks = self.landmarks_frame.iloc[idx, 2].astype('int')
            landmarks = [1 if landmarks == i else 0 for i in range(1, 58)]
        else:
            landmarks = self.landmarks_frame.iloc[idx, 1]
            landmarks = landmarks
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample
