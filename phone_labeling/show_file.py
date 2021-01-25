'''
This is used to view a file produced by kotlin and check that the network is outputting the right thing.

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


def get_example_input(image_file):
    """
    Loads image from disk and converts to compatible shape.
    :param image_file: Path to single image file
    :return:  torch.Tensor image
    """

    f = open(image_file, "rb")
    data = list(f.read())
    f.close()
    data_int = np.array(data, dtype='float')
    image_y = data_int.reshape(480, 640)
    image = np.zeros(shape=(1, 480, 640))
    image[0, :, :] = image_y

    torch_img = torch.from_numpy(image).float()
    torch_img = torch_img.unsqueeze(0)
    torch_img = torch_img.to(torch.device("cpu"))

    return torch_img


def predict_torch(model, image):
    """
    Torch model prediction (forward propagate)
    :param model: PyTorch model
    :param image: Input image
    :return: Numpy array with logits
    """
    image = image.to("cpu")
    outputs = model(image)
    pred = torch.argmax(outputs, dim=1)

    return pred.data.cpu().numpy()


def get_torch_model(model_path):
    """
    Loads state-dict into model and creates an instance
    :param model_path: State-dict path to load PyTorch model with pre-trained weights
    :return: PyTorch model instance
    """
    model = torch.load(model_path, map_location='cpu')
    return model


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

    #index = np.arange(0, np.int(w * h / 2), 2, dtype=np.int)
    #v = v[index]
    #u = u[index]

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


def show_overlay(file_in_y, file_in_u, file_in_v, file_overlay):
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

    f = open(file_overlay, "rb")
    overlay = list(f.read())
    f.close()

    u = np.array(u, dtype=np.int32)
    v = np.array(v, dtype=np.int32)
    y = np.array(y, dtype=np.int32)
    overlay = np.array(overlay, dtype=np.int32)

    print(y[0:10])

    #index = np.arange(0, np.int(w * h / 2), 2, dtype=np.int)
    #v = v[index]
    #u = u[index]

    r = np.ndarray(w * h, dtype=np.int32)
    g = np.ndarray(w * h, dtype=np.int32)
    b = np.ndarray(w * h, dtype=np.int32)

    yuvlib.yuv2rgb(w, h, y, u, v, r, g, b)

    r = np.flip(np.transpose(r.reshape((w, h))), 1)
    g = np.flip(np.transpose(g.reshape((w, h))), 1)
    b = np.flip(np.transpose(b.reshape((w, h))), 1)

    overlay = np.flip(np.transpose(overlay.reshape((w, h))), 1)

    rgb = np.zeros(shape=(h, w, 3))
    rgb[:, :, 0] = r
    rgb[:, :, 1] = g + overlay * 100
    rgb[:, :, 2] = b

    img = Image.fromarray(np.uint8(rgb.clip(0, 255)))
    img.show()


file_overlay = '/mnt/nvme/mask.bin'
file_out = '/mnt/nvme/out.png'
torch_model_path = '/mnt/nvme/adas/code/phone_labeling/networks/checkpoint__net174.pt'

file_y = '/mnt/nvme/2_110_334_2021-01-25-06-04-13-743_y.bin'
file_u = file_y.replace('_y', '_u')
file_v = file_y.replace('_y', '_v')

show_overlay(file_y, file_u, file_v, file_overlay)

torch_model = get_torch_model(torch_model_path)
torch_image = get_example_input(file_y)
torch_output = predict_torch(torch_model, torch_image)
torch_output = np.squeeze(torch_output, axis=0)
save_overlay(file_y, file_u, file_v, file_out, torch_output)
