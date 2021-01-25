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
import ctypes
import numpy
import glob

libfile = '/mnt/nvme/adas/code/yuv2rgb/build/lib.linux-x86_64-3.8/yuv2rgb.cpython-38-x86_64-linux-gnu.so'

# 1. open the shared library
yuvlib = ctypes.CDLL(libfile)

# 2. tell Python the argument and result types of function mysum
yuvlib.yuv2rgb.argtypes = [ctypes.c_int, ctypes.c_int,
                           np.ctypeslib.ndpointer(dtype=np.int32),
                           np.ctypeslib.ndpointer(dtype=np.int32),
                           np.ctypeslib.ndpointer(dtype=np.int32),
                           np.ctypeslib.ndpointer(dtype=np.int32),
                           np.ctypeslib.ndpointer(dtype=np.int32),
                           np.ctypeslib.ndpointer(dtype=np.int32)]


def save_overlay(file_in_y, file_in_u, file_in_v, file_out, overlay):
    """
    Read the y,u,v files from kotlin and make an rgb png file to be used to to some annotation.
    :param file_in_y: y file path
    :param file_in_u: u file path
    :param file_in_v: v file path
    :param file_out: output png file
    """
    w = 480
    h = 640

    f = open(file_in_u, "rb")
    u = list(f.read())
    f.close()

    f = open(file_in_v, "rb")
    v = list(f.read())
    f.close()

    f = open(file_in_y, "rb")
    y = list(f.read())
    f.close()

    u = np.array(u, dtype=np.int32)
    v = np.array(v, dtype=np.int32)
    y = np.array(y, dtype=np.int32)

    index = np.arange(0, np.int(w * h / 2), 2, dtype=np.int)
    v = v[index]
    u = u[index]

    r = np.ndarray(w * h, dtype=np.int32)
    g = np.ndarray(w * h, dtype=np.int32)
    b = np.ndarray(w * h, dtype=np.int32)

    yuvlib.yuv2rgb(w, h, y, u, v, r, g, b)

    r = np.flip(np.transpose(r.reshape((w, h))), 1)
    g = np.flip(np.transpose(g.reshape((w, h))), 1)
    b = np.flip(np.transpose(b.reshape((w, h))), 1)

    rgb = np.zeros(shape=(h, w, 3))
    rgb[:, :, 0] = r
    rgb[:, :, 1] = g + np.flip(np.transpose(overlay), 1) * 100
    rgb[:, :, 2] = b

    img = Image.fromarray(np.uint8(rgb.clip(0, 255)))
    img.save(file_out)


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
    test_dataset = CityscapeTestDataset(csv_file='/mnt/nvme/work/phone_labeling/test.csv',
                                        root_dir='/mnt/nvme/work/phone_labeling/img_raw')
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
        procent += 4
        local_batch = data['image'].float()
        file_names = data['name']

        local_batch = local_batch.to(device)
        net.to(device)
        outputs = net(local_batch)
        pred = torch.argmax(outputs, dim=1)  # we get the predictions
        pred = np.array(pred.cpu().numpy(), dtype='uint8')  # take things back in cpu space and make them uint8

        j = 0
        for fn in file_names:
            file = file_names[j][file_names[j].rfind('/') + 1: file_names[j].__len__()].replace('.bin', '.png')
            file_out = checkpoints_dir + '/view/' + file
            files_y = file_names[j]
            files_u = file_names[j].replace('_y', '_u')
            files_v = file_names[j].replace('_y', '_v')

            save_overlay(files_y, files_u, files_v, file_out, pred[j, :])
            j += 1
        progress(np.int(100 * procent / nr_samples))


run_test(checkpoints_dir + '/view', 356)
