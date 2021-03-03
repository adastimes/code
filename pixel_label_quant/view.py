'''
This is used to view the data. It also runs the network over a dataset and dumps images with street overlay
to visualise the result.

Author: John ADAS Doe
Email: john.adas.doe@gmail.com
License: Apache-2.0
'''

import os
import numpy as np
from PIL import Image
from data_loader import CityscapeTestDataset
from network import Network
from common import *
import torch
from skimage import io
import torch.nn as nn
import torch.optim as optim


def view_image(im, overlay):
    """
    Adds the overlay on the image and then displays it. The overlay is shifted to green.
    :param im: np.array with the image as (3,x, y)
    :param overlay: np.array with the mask as (x,y)
    """
    over = np.stack((overlay, overlay * 100, overlay))
    vec = np.moveaxis(im + over, 0, 2)
    img = Image.fromarray(vec, 'RGB')
    img.show()


def save_overlay(im, overlay, file_name):
    """
    Adds the overlay on the image and then displays it. The overlay is shifted to green.
    :param im: np.array with the image as (3,x, y)
    :param overlay: np.array with the mask as (x,y)
    :param file_name: file name to save
    """
    over = np.stack((overlay, overlay * 100, overlay))
    vec = np.moveaxis(im + over, 0, 2)
    img = Image.fromarray(vec, 'RGB')
    img.save(file_name, "PNG")


def run_test(directory, checkpoint=8):
    """
    This function parses a csv file and applyies the network over all files specified in there. The it
    overlays the network output and writes the images in a specified directory.

    :param directory: This is the directory where we save the images after overlaying the network result.
    :param checkpoint: The number of the checkpoint to load, default is 8.
    """
    if not os.path.exists(directory):  # make the directory
        os.makedirs(directory)

    batch_size = 4
    test_dataset = CityscapeTestDataset(csv_file='/mnt/nvme/work/Cityscape/leftImg8bit/test.csv',
                                        root_dir='/mnt/nvme/work/Cityscape/leftImg8bit')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=2,
                                              collate_fn=collate_func)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    tool = NetworkTool(path=checkpoints_dir)
    net, optimizer, epoch, loss = tool.load_checkpoint(checkpoint)
    net.eval()

    nr_samples = len(test_dataset)
    procent = 0
    for i, data in enumerate(test_loader, 0):
        procent +=4
        local_batch = data['image'].float()
        file_names = data['name']

        local_batch = local_batch.to(device)
        net.to(device)
        outputs = net(local_batch)
        pred = torch.argmax(outputs, dim=1)  # we get the predictions
        pred = np.array(pred.cpu().numpy(), dtype='uint8')  # take things back in cpu space and make them uint8

        j = 0
        for fn in file_names:
            file = file_names[j][file_names[j].rfind('/') + 1: file_names[j].__len__()]
            im = np.array(data['image'][j, :].numpy(), dtype='uint8')
            save_overlay(im, pred[j, :], directory + '/' + file)
            j +=1
        progress(np.int(100*procent / nr_samples))


run_test(checkpoints_dir + '/view', 18)
