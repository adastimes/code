'''
Configuration file for the pixel labeling project.Will run automatically with the other scripts.
Author: John ADAS Doe
Email: john.adas.doe@gmail.com
License: Apache-2.0
'''

import torch
import torch.optim as optim
from network_reg import NetworkReg, NetworkLabeling,NetworkRegSimple, NetworkRegR
import numpy as np

'''
    Configuration Parameters. Here you need to change for your setup.
'''
reference_dir = '/mnt/nvme/work/Cityscape/gtFine'  # here we have the annotated data
dataset_dir = '/mnt/nvme/work/Cityscape/leftImg8bit'  # here are the images
checkpoints_dir = '/mnt/nvme/adas/code/pixel_label_reg/networks/'
checkpoints_dir_labeling = '/mnt/nvme/adas/code/pixel_label/networks/'

extension_added = '_small'  # this is the extension appended to all files after downsampling
df = 2  # downsampling factor on both x and y direction just to fit the GPU and train for less time
road_id = 7  # this is the id of the road
prep_dataset = 1  # resize images and produce a csv image


def collate_func(batch):
    """
    This function is used to filter out batches that have bad images. The DatasetLoaders are returning None
    if images are corrupted (in __getitem__).
    :param batch: The batch to look into if there is any None inside.
    :return:
    """
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


def progress(percent=0, width=30):
    """
    Print a progress bar. We use this around the place to see where we are.
    :param percent: percent done
    :param width: width of the progress bar
    """
    left = width * percent // 100
    right = width - left
    print('\r[', '#' * left, ' ' * right, ']',
          f' {percent:.0f}%',
          sep='', end='', flush=True)


class NetworkToolReg(object):
    def __init__(self, path):
        """
        :param path: This is the path where to save the file. Files saved are model, params and checkpoint_xx.pt
        """
        self.path_model = path + 'model.pt'
        self.path_params = path + 'params.pt'
        self.path_checkpoint = path + 'checkpoint_'

    def save_checkpoint(self, model, epoch, loss, optimizer):
        """
        Save a checkpoint.
        :param model: model to be saved
        :param epoch: epoch where we are with training
        :param loss: the loss where we are
        :param optimizer: optimizer setup
        """
        file_name = self.path_checkpoint + str(epoch) + '.pt'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss}, file_name)

    def load_checkpoint(self, epoch):
        """
        Load a checkpoint.
        :param epoch: epoch checkpoint that we should load
        :return: model, optimizer , epoch and loss are returned
        """
        self.path_checkpoint = self.path_checkpoint + str(epoch) + '.pt'
        model = NetworkRegR()
        optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01,
                               amsgrad=False)

        checkpoint = torch.load(self.path_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']

        return model, optimizer, epoch, loss

    def load_checkpoint_labeling(self, epoch):
        """
        Load a checkpoint.
        :param epoch: epoch checkpoint that we should load
        :return: model, optimizer , epoch and loss are returned
        """
        self.path_checkpoint = self.path_checkpoint + str(epoch) + '.pt'
        model = NetworkLabeling()
        optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01,
                               amsgrad=False)

        checkpoint = torch.load(self.path_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']

        return model, optimizer, epoch, loss