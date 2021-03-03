'''
Script to train the neural network for road labeling.

Author: John ADAS Doe
Email: john.adas.doe@gmail.com
License: Apache-2.0
'''

import numpy as np
from PIL import Image

from data_loader import CityscapeDataset
from network import Network
from common import *

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import csv


def plot_stats(v, t, fn, label):
    fig, ax = plt.subplots()
    ax.plot(v, 'r')
    ax.plot(t, 'b')
    ax.set(xlabel='Epochs', ylabel=label,
           title='Loss function.')
    ax.grid()
    fig.savefig(checkpoints_dir + fn)


def write_net(file_path, model, in_shape):
    # Works only for convolution, transpose conv and for relu
    file_name = file_path + "/model.txt"
    f = open(file_name, 'w', newline='')
    id = 0
    f.write("-------------------------------------------\n")
    f.write('Layer_Type = Image\n')
    f.write('Channels = %d\n' % in_shape[1])
    f.write('Height = %d\n' % in_shape[2])
    f.write('Width = %d\n' % in_shape[3])

    for layer in net.children():  # write the just the convolution and relu
        if layer._get_name() == "Conv2d" or layer._get_name() == "ConvTranspose2d":
            f.write("-------------------------------------------\n")
            f.write('Layer_Type = %s\n' % layer._get_name())
            f.write('ID = %d\n' % id)
            f.write('in_channels = %d\n' % layer.in_channels)
            f.write('out_channels = %d\n' % layer.out_channels)
            f.write('kernel_size = (%d,%d)\n' % (layer.kernel_size[0], layer.kernel_size[1]))
            f.write('padding = (%d,%d)\n' % (layer.padding[0], layer.padding[1]))
            f.write('stride = (%d,%d)\n' % (layer.stride[0], layer.stride[1]))

            fl_name = file_path + "/" + str(id) + ".txt"
            weights = layer.weight.data.numpy()
            weights = weights.reshape(weights.shape[0], -1)
            np.savetxt(fl_name, weights)

            fl_name = file_path + "/" + str(id) + "_bias.txt"
            bias = layer.bias.data.numpy()
            np.savetxt(fl_name, bias)

            id += 1
        if layer._get_name() == "ReLU":
            f.write("-------------------------------------------\n")
            f.write('Layer_Type = %s\n' % layer._get_name())
            f.write('ID = %d\n' % id)
            id += 1

    f.close()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    train_dataset = CityscapeDataset(csv_file='/mnt/nvme/work/Cityscape/leftImg8bit/train.csv',
                                     root_dir='/mnt/nvme/work/Cityscape/leftImg8bit')

    valid_dataset = CityscapeDataset(csv_file='/mnt/nvme/work/Cityscape/leftImg8bit/train.csv',
                                     root_dir='/mnt/nvme/work/Cityscape/leftImg8bit')

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=2)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1, shuffle=True, num_workers=2,
                                               collate_fn=collate_func)

    # net = Network()
    tool = NetworkTool(path=checkpoints_dir)
    net, opt, ep, l = tool.load_checkpoint(74)

    #write_net('/mnt/nvme', net,[1, 3, 256, 512])

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)

    '''
    #betas - this are the values provided in the Adam paper
    #eps - 1e-4 to 1e-8 is suggested in the paper
    #weight decay - it cannot be too much as then we prioratize small weights to the goal, fastai puts 0.01
    '''

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print(device)
    device = torch.device("cpu")

    net.train()
    torch.backends.quantized.engine = 'qnnpack'
    net.qconfig = torch.quantization.get_default_qat_qconfig('qnnpack')

    net_fused = torch.quantization.fuse_modules(net, [['conv1', 'relu1'], ['conv2', 'relu2'], ['conv3', 'relu3'],
                                                      ['conv4', 'relu4'], ['conv5', 'relu5']])
    net_prepared = torch.quantization.prepare_qat(net_fused)

    train_loss = []
    valid_loss = []
    accuracy = []
    IU = []
    # Loop over epochs
    torch.manual_seed(0)
    for epoch in range(80):
        # Training
        running_loss = 0.0
        net_prepared.train()
        for i, data in enumerate(train_loader, 0):
            local_labels = data['labels'].long()
            local_batch = data['image'].float()

            x = net(local_batch)

            local_batch, local_labels = local_batch.to(device), local_labels.to(device)
            net_prepared.to(device)
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net_prepared(local_batch)
            loss = criterion(outputs, local_labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 1000 == 0:
                print('TRAINING: epoch: %d loss: %.3f' % (epoch + 1, running_loss / (i + 1)))

        train_loss.append(running_loss / float(i + 1))
        if epoch % 2 == 0:
            tool.save_checkpoint(net_prepared, epoch, loss, optimizer)

        # validation
        running_loss = 0.0
        net_prepared.eval()
        net_int8 = torch.quantization.convert(net_prepared)
        m = Metric()
        for i, data in enumerate(valid_loader, 0):
            local_labels = data['labels'].long()
            local_batch = data['image'].float()
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)
            net_int8.to(device)
            outputs = net_int8(local_batch)
            loss = criterion(outputs, local_labels)
            running_loss += loss.item()

            pred = torch.argmax(outputs, dim=1)  # we get the predictions
            pred = np.array(pred.cpu().numpy(), dtype='uint8')  # take things back in cpu space and make them uint8
            ref = np.array(data['labels'], dtype='uint8')
            m.get_metrics(pred, ref)

        valid_loss.append(running_loss / float(i + 1))
        accuracy.append(m.accuracy)
        IU.append(m.IU)
        print('VALIDATION: epoch: %d loss: %.3f accuracy: %f mIU: %f' % (
            epoch + 1, running_loss / (i + 1), m.accuracy, m.IU))

    plot_stats(valid_loss, train_loss, 'loss.png', 'r- validation loss/b - training loss')
    plot_stats(accuracy, IU, 'acc_iu.png', 'r- accuracy/b - IU')
