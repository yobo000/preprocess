# -*- coding: utf-8 -*-

import os
import pandas as pd
from skimage import io
from torch.utils.data import Dataset

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


class CategoriesDataset(object):
    """Categories Landmarks datase for beauty, fashion, mobile"""

    def __init__(self, cag):
        self.category = cag

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

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 3])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 2]
        landmarks = landmarks.astype('int')
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample
