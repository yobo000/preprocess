# -*- coding: utf-8 -*-

import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


class CategoriesDataset(object):
    """Categories Landmarks datase for beauty, fashion, mobile"""

    def __init__(self, category):
        self.category = category

    def __call__(self, *args):
        return BaseDataset(*args)


class BaseDataset(Dataset):
    """Categories Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
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
        self.category = "image"

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 3])
        image = Image.fromarray(img_name)
        if self.category == "image":
            landmarks = self.landmarks_frame.iloc[idx, 2].astype('int')
            landmarks = [0 if landmarks == i else 1 for i in xrange(1, 56)]
        else:
            landmarks = self.landmarks_frame.iloc[idx, 1]
            landmarks = landmarks
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample
