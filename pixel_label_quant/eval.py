'''
Run the network as save in a checkpoint and then compute metrics like IU and accuracy.

Author: John ADAS Doe
Email: john.adas.doe@gmail.com
License: Apache-2.0
'''

import os
import numpy as np
from PIL import Image
from data_loader import CityscapeDataset
from network import Network
from common import *
import torch
from skimage import io
import torch.nn as nn
import torch.optim as optim


def compute_metrics(checkpoint=8):
    """
    This function parses a csv file and applies the network over all files specified in there. It computes
    after the mIU and accuracy by using the Metric class from common.py

    :param checkpoint: The number of the checkpoint to load, default is 8.
    """
    batch_size = 4
    test_dataset = CityscapeDataset(csv_file='/mnt/nvme/work/Cityscape/leftImg8bit/val.csv',
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
    m = Metric()
    for i, data in enumerate(test_loader, 0):
        procent += 4

        local_labels = data['labels'].long()
        local_batch = data['image'].float()
        local_batch, local_labels = local_batch.to(device), local_labels.to(device)
        net.to(device)
        outputs = net(local_batch)

        pred = torch.argmax(outputs, dim=1)  # we get the predictions
        pred = np.array(pred.cpu().numpy(), dtype='uint8')  # take things back in cpu space and make them uint8
        ref = np.array(data['labels'], dtype='uint8')

        m.get_metrics(pred,ref)
        progress(np.int(100*procent / nr_samples))

    print('\n=======================================')
    print('Accuracy = ', m.accuracy)
    print('mIU = ', m.IU)


compute_metrics(8)
