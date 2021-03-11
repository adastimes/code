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
from data_loader_reg import CityscapeTestDatasetReg
from network_reg import NetworkReg, NetworkLabeling,NetworkRegSimple,NetworkRegR
from common_reg import *
import torch
from skimage import io
import torch.nn as nn
import torch.optim as optim


def save_overlay(im, overlay, overlay_ref, file_name):
    """
    Adds the overlay on the image and then displays it. The overlay is shifted to green.
    :param im: np.array with the image as (3,x, y)
    :param overlay: np.array with the mask as (x,y)
    :param file_name: file name to save
    """
    overlay_cliped = np.round(overlay).clip(0, 255)
    overlay_ref = np.round(overlay_ref).clip(0, 255)
    for j in range(len(overlay)):
        im[1, np.int(overlay_cliped[j]), j] = 255
        im[0, np.int(overlay_ref[j]), j] = 255

    im = np.moveaxis(im, 0, 2)
    img = Image.fromarray(im)
    img.save(file_name, "PNG")


def run_test_labeling(directory, checkpoint=8):
    """
    This function parses a csv file and applyies the network over all files specified in there. The it
    overlays the network output and writes the images in a specified directory.

    :param directory: This is the directory where we save the images after overlaying the network result.
    :param checkpoint: The number of the checkpoint to load, default is 8.
    """
    if not os.path.exists(directory):  # make the directory
        os.makedirs(directory)

    batch_size = 4
    test_dataset = CityscapeTestDatasetReg(csv_file='/mnt/nvme/work/Cityscape/leftImg8bit/val.csv',
                                        root_dir='/mnt/nvme/work/Cityscape/leftImg8bit')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=2,
                                              collate_fn=collate_func)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #device = "cpu"
    print(device)

    tool = NetworkToolReg(path=checkpoints_dir_labeling)
    net, optimizer, epoch, loss = tool.load_checkpoint_labeling(checkpoint)
    criterion = nn.MSELoss()
    net.eval()

    nr_samples = len(test_dataset)
    procent = 0
    running_loss = 0
    for i, data in enumerate(test_loader, 0):
        procent += batch_size
        local_batch = data['image'].float()
        local_labels = data['labels'].float()
        ref = local_labels
        file_names = data['name']

        local_batch, local_labels = local_batch.to(device), local_labels.to(device)
        net.to(device)
        outputs = net(local_batch)

        pred = torch.argmax(outputs, dim=1)  # we get the predictions
        pred = np.array(pred.cpu().numpy(), dtype='uint8')

        freespace = torch.zeros(size=(pred.shape[0], pred.shape[2]), dtype=torch.int)
        for batch_idx in range(pred.shape[0]):
            for col_idx in range(pred.shape[2]):
                col = pred[batch_idx, :, col_idx]

                found_first = False
                freespace[batch_idx, col_idx] = 255
                for line_idx in range(len(col) - 1, 0, -1):
                    if found_first == False and col[line_idx] == 1:
                        freespace[batch_idx, col_idx] = line_idx
                        found_first = True

                    if found_first and col[line_idx] != 1:
                        freespace[batch_idx, col_idx] = line_idx
                        break

        start_dev = freespace.to(device)

        ind = torch.where(local_labels != 255)
        loss = criterion(start_dev[ind[0]], local_labels[ind[0]])

        running_loss += loss.item()

        freespace = freespace.numpy()

        # pred = np.array(outputs.cpu().detach().numpy(), dtype='float')  # take things back in cpu space

        j = 0
        for fn in file_names:
            file = file_names[j][file_names[j].rfind('/') + 1: file_names[j].__len__()]
            im = np.array(data['image'][j, :].numpy(), dtype='uint8')
            save_overlay(im, freespace[j, :], ref[j, :], directory + '/' + file)
            j += 1
        progress(np.int(100 * procent / nr_samples))

    print('\n LOSS: %.3f' % (running_loss / (nr_samples)))


def run_test_reg(directory, checkpoint=8):
    """
    This function parses a csv file and applyies the network over all files specified in there. The it
    overlays the network output and writes the images in a specified directory.

    :param directory: This is the directory where we save the images after overlaying the network result.
    :param checkpoint: The number of the checkpoint to load, default is 8.
    """
    if not os.path.exists(directory):  # make the directory
        os.makedirs(directory)

    batch_size = 16
    test_dataset = CityscapeTestDatasetReg(csv_file='/mnt/nvme/work/Cityscape/leftImg8bit/val_reg.csv',
                                        root_dir='/mnt/nvme/work/Cityscape/leftImg8bit')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=2,
                                              collate_fn=collate_func)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    tool = NetworkToolReg(path=checkpoints_dir)
    net, optimizer, epoch, loss = tool.load_checkpoint(checkpoint)
    criterion = nn.MSELoss()
    net.eval()

    nr_samples = len(test_dataset)
    procent = 0
    running_loss = 0
    for i, data in enumerate(test_loader, 0):
        procent += batch_size
        local_batch = data['image'].float()
        local_labels = data['labels'].float()
        #local_labels = local_labels[:, 100:400]

        ref = local_labels
        file_names = data['name']

        local_batch, local_labels = local_batch.to(device), local_labels.to(device)
        net.to(device)
        outputs,y = net(local_batch)

        outputs = torch.round(outputs*256.0)

        loss = criterion(outputs, local_labels)
        running_loss += loss.item()

        pred = np.array(outputs.cpu().detach().numpy(), dtype='float')  # take things back in cpu space
        j = 0
        for fn in file_names:
            file = file_names[j][file_names[j].rfind('/') + 1: file_names[j].__len__()]
            im = np.array(data['image'][j, :].numpy(), dtype='uint8')
            save_overlay(im, pred[j, :], ref[j, :], directory + '/' + file)
            j += 1
        progress(np.int(100 * procent / nr_samples))

    print('\n LOSS: %.3f' % (running_loss / (nr_samples)))


run_test_reg(checkpoints_dir + '/view_reg_norm', 58)
#run_test_labeling(checkpoints_dir + '/view_labeling', 78)
