'''
From torch to tf.

Author: John ADAS Doe
Email: john.adas.doe@gmail.com
License: Apache-2.0
'''

import argparse
from torch2tflite import *


def init_models(torch_model_path, tf_lite_model_path):
    """
    Initialize the Torch and TFLite models
    :param torch_model_path: Path to Torch model
    :param tf_lite_model_path: Path to TFLite model
    :return: CPU initialized models
    """
    torch_model = get_torch_model(torch_model_path)
    tf_lite_model = get_tf_lite_model(tf_lite_model_path)
    return torch_model, tf_lite_model

def main():
    """
    Converts PyTorch model into TFLite with using ONNX and main TensorFlow between them.
    """

    tf_lite_path = '/mnt/nvme/adas/code/phone_labeling/networks/PixelLabeling.tflite'
    torch_path = '/mnt/nvme/adas/code/phone_labeling/networks/checkpoint__net356.pt'
    img_path = '/mnt/nvme/work/phone_labeling/img_raw/2_120_339_y.bin'
    img_out_path = '/mnt/nvme/adas/code/phone_labeling/networks/tmp'

    convert(torch_model_path=torch_path,
            tf_lite_model_path=tf_lite_path,
            image_path=img_path)
    original_image, tf_lite_image, torch_image = get_example_input(img_path)
    torch_model, tf_lite_model = init_models(torch_path, tf_lite_path)
    tf_lite_output = predict_tf_lite(tf_lite_model, tf_lite_image)
    torch_output = predict_torch(torch_model, torch_image)

    _ = calc_error(torch_output, tf_lite_output, img_path, img_out_path)

if __name__ == '__main__':
    main()
