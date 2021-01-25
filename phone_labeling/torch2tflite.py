'''
Script that has all the functions to transform the network from pytorch to tflite, load models, run models etc.

Author: John ADAS Doe
Email: john.adas.doe@gmail.com
License: Apache-2.0
'''
import onnx
import torch
import numpy as np
import tensorflow as tf
from PIL import Image
from torchvision import transforms
from onnx_tf.backend import prepare
import os
import shutil
import ctypes

# pip install onnx
# pip instal onnx-tf


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


def get_example_input(image_file):
    """
    Loads image from disk and converts to compatible shape.
    :param image_file: Path to single image file
    :return: Original image, numpy.ndarray instance image, torch.Tensor image
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

    return image, torch_img.numpy(), torch_img


def get_torch_model(model_path):
    """
    Loads state-dict into model and creates an instance
    :param model_path: State-dict path to load PyTorch model with pre-trained weights
    :return: PyTorch model instance
    """
    model = torch.load(model_path, map_location='cpu')
    return model


def get_tf_lite_model(model_path):
    """
    Creates an instance of TFLite CPU interpreter
    :param model_path: TFLite model path to initialize
    :return: TFLite interpreter
    """
    interpret = tf.lite.Interpreter(model_path)
    interpret.allocate_tensors()
    return interpret


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


def predict_tf_lite(model, image):
    """
    TFLite model prediction (forward propagate)
    :param model: TFLite interpreter
    :param image: Input image
    :return: Numpy array with logits
    """
    input_details = model.get_input_details()
    output_details = model.get_output_details()
    model.set_tensor(input_details[0]['index'], image)
    model.invoke()
    tf_lite_output = model.get_tensor(output_details[0]['index'])
    tf_lite_output2 = tf.math.argmax(tf_lite_output,axis=1)
    tf_lite_output = tf_lite_output2.numpy()
    return tf_lite_output


def calc_error(res1, res2, img_in_path, img_out_path):
    """
    Calculates specified error between two results. In here Mean-Square-Error and Mean-Absolute-Error calculated"
    :param res1: First result
    :param res2: Second result
    :param verbose: Print loss results
    :return: Loss metrics as a dictionary
    """

    res1 = np.squeeze(res1, axis=0)
    res2 = np.squeeze(res2, axis=0)

    mse = ((res1 - res2) ** 2).mean(axis=None)
    mae = np.abs(res1 - res2).mean(axis=None)
    metrics = {'mse': mse, 'mae': mae}
    print(f"\n\nMean-Square-Error between predictions:\t{metrics['mse']}")
    print(f"Mean-Abs-Error between predictions:\t{metrics['mae']}\n\n")

    file_y = img_in_path
    file_u = file_y.replace('_y','_u')
    file_v = file_y.replace('_y', '_v')

    save_overlay(file_y, file_u, file_v, img_out_path + '1.png', res1)
    save_overlay(file_y, file_u, file_v, img_out_path + '2.png', res2)

    return metrics


# ------------------ Main Convert Function ------------------#
def convert(torch_model_path, tf_lite_model_path, image_path):
    onnx_path = "/mnt/nvme/adas/code/phone_labeling/networks/model.onnx"
    tf_path = "/mnt/nvme/adas/code/phone_labeling/networks/model.tf"

    try:
        os.remove(onnx_path)
        shutil.rmtree(tf_path)
    except:
        pass

    try:
        pytorch_model = get_torch_model(torch_model_path)
        image, tf_lite_image, torch_image = get_example_input(image_path)

        torch.onnx.export(
            model=pytorch_model,
            args=torch_image,
            f=onnx_path,
            verbose=False,
            export_params=True,
            do_constant_folding=False,  # fold constant values for optimization
            input_names=['input'],
            opset_version=10,
            output_names=['output'])

        print('\n\nTorch to ONNX converted!\n\n')

        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)  # Checks signature
        tf_rep = prepare(onnx_model)  # Prepare TF representation
        tf_rep.export_graph(tf_path)  # Export the model

        print('\n\nONNX to TF converted!\n\n')

        converter = tf.lite.TFLiteConverter.from_saved_model(tf_path)  # Path to the SavedModel directory
        tflite_model = converter.convert()  # Creates converter instance
        with open(tf_lite_model_path, 'wb') as f:
            f.write(tflite_model)

        print('\n\nTF to TFLite converted!\n\n')
    except:
        pass