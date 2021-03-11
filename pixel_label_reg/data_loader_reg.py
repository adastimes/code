'''
Prepares the dataset loader from pytorch. The loader needs 3 functions : init, len and getitem.
In init we read the csv file and take out the frame list and then in getitem we read the image, put it
in an np.array, same for the reference.

The labels are also replaced such that road is marked as 1 and everything else is zero.

Assumption: gtFine and  leftImg8bit from cityscapes are in the same directory.

Author: John ADAS Doe
Email: john.adas.doe@gmail.com
License: Apache-2.0
'''

import torch
import pandas as pd
from PIL import Image
import numpy as np


class CityscapeDatasetReg(torch.utils.data.Dataset):

    def __init__(self, csv_file, root_dir):
        """
        Initialize the config file with images and the root directory for tha data.
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
        """
        self.frames = pd.read_csv(csv_file)
        self.root_dir = root_dir

    def __len__(self):
        """
        Return the length of tha dataset.
        :return: number of samples available.
        """
        return len(self.frames)

    def __getitem__(self, idx):
        """
        Get an image and label. Image is prepared to color channel first and then width and height.
        :param idx: index of the item.
        :return:
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.frames.iloc[idx, 1]
        label_name = img_name.replace('/leftImg8bit/', '/gtFine/').replace('_leftImg8bit', '_gtFine_labelIds') + '.bin'

        try:
            image = np.array(Image.open(img_name))
            labels = np.loadtxt(label_name, dtype=np.float)

            image = np.rollaxis(image, 2, 0)  # need to change to color channel first, as this is the way the network
            # wants it

            sample = {'image': image, 'labels': labels}

        except:
            return None

        return sample


class CityscapeTestDatasetReg(torch.utils.data.Dataset):

    def __init__(self, csv_file, root_dir):
        """
        Initialize the config file with images and the root directory for tha data.
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
        """
        self.frames = pd.read_csv(csv_file)
        self.root_dir = root_dir

    def __len__(self):
        """
        Return the length of tha dataset.
        :return: number of samples available.
        """
        return len(self.frames)

    def __getitem__(self, idx):
        """
        Get an image and label. Image is prepared to color channel first and then width and height.
        :param idx: index of the item.
        :return:
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.frames.iloc[idx, 1]
        label_name = img_name.replace('/leftImg8bit/', '/gtFine/').replace('_leftImg8bit', '_gtFine_labelIds') + '.bin'

        # if there is no road we need to jump over this image
        orig_label = img_name.replace('/leftImg8bit/', '/gtFine/').replace('_leftImg8bit', '_gtFine_labelIds')
        orig_labels_np = np.array(Image.open(orig_label))
        if len(np.where(orig_labels_np == 7)) == 0:
            return None

        try:
            image = np.array(Image.open(img_name))
            labels = np.loadtxt(label_name, dtype=np.float)
            image = np.rollaxis(image, 2, 0)  # need to change to color channel first, as this is the way the network

            sample = {'image': image, 'name': img_name, 'labels': labels}
            # sample = {'image': image, 'name': img_name}

        except:
            return None

        return sample
