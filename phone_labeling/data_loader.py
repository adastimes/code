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
from common import *


class CityscapeDataset(torch.utils.data.Dataset):

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
        label_name = img_name.replace('/img_raw/', '/ref/').replace('.bin', '.png')

        try:

            f = open(img_name, "rb")
            data = list(f.read())
            f.close()
            data_int = np.array(data, dtype='float')
            image_y = data_int.reshape(480, 640)
            image = np.zeros(shape=(1, 480, 640))
            image[0, :, :] = image_y

            labels = np.array(Image.open(label_name),dtype='float')
            labels = np.where(labels != 7, 0, labels)  # all other labels are zero
            labels = np.where(labels == 7, 1, labels)  # replace road label with one
            labels = np.flip(np.transpose(labels), 0).copy()

            '''
            for i in range(labels.shape[0]):
                for j in range(labels.shape[1]):
                    if(labels[i,j]==1):
                        image[0,i,j] = 255

            img = Image.fromarray(image)
            img.show()
            '''

            sample = {'image': image, 'labels': labels}

        except:
            return None

        return sample


'''
CityscapeTestDataset is used for parsing the testing data where we do not need any reference/label.   
'''


class CityscapeTestDataset(torch.utils.data.Dataset):

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

        try:
            f = open(img_name, "rb")
            data = list(f.read())
            f.close()
            data_int = np.array(data, dtype='float')
            image_y = data_int.reshape(480, 640)
            image = np.zeros(shape=(1, 480, 640))
            image[0, :, :] = image_y

            sample = {'image': image, 'name': img_name}

        except:
            return None

        return sample
