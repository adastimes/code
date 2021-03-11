'''
Script to train the neural network for road labeling.

Author: John ADAS Doe
Email: john.adas.doe@gmail.com
License: Apache-2.0
'''

import numpy as np
from PIL import Image

from data_loader_reg import CityscapeDatasetReg
from network_reg import NetworkReg,NetworkRegSimple,NetworkRegR
from common_reg import *

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


def plot_stats(v, t, fn, label):
    fig, ax = plt.subplots()
    ax.plot(v, 'r')
    ax.plot(t, 'b')
    ax.set(xlabel='Epochs', ylabel=label,
           title='Loss function.')
    ax.grid()
    fig.savefig(checkpoints_dir + fn)


#sum(abs(out-target)) - L1
def loss_func_L1(output, target):
    l = torch.mean( torch.abs(output - target) )
    return l

#max(out-target)
def loss_func_max(output, target):
    l = torch.max(torch.abs(output - target))
    return l


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    train_dataset = CityscapeDatasetReg(csv_file='/mnt/nvme/work/Cityscape/leftImg8bit/train_reg.csv',
                                        root_dir='/mnt/nvme/work/Cityscape/leftImg8bit')

    valid_dataset = CityscapeDatasetReg(csv_file='/mnt/nvme/work/Cityscape/leftImg8bit/val_reg.csv',
                                        root_dir='/mnt/nvme/work/Cityscape/leftImg8bit')

    bs = 8
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=2)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=bs, shuffle=True, num_workers=2,
                                               collate_fn=collate_func)

    net = NetworkRegR()
    tool = NetworkToolReg(path=checkpoints_dir)
    net, opt, ep, l = tool.load_checkpoint(99)


    criterion = nn.MSELoss()
    smoth_L1 = nn.SmoothL1Loss()
    optimizer = optim.Adam(net.parameters(), lr=0.00001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)

    '''
    #betas - this are the values provided in the Adam paper
    #eps - 1e-4 to 1e-8 is suggested in the paper
    #weight decay - it cannot be too much as then we prioratize small weights to the goal, fastai puts 0.01
    '''

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    train_loss = []
    train_loss_L1 = []
    train_loss_max = []
    train_loss_sL1 = []

    valid_loss = []
    valid_loss_L1 = []
    valid_loss_max = []
    valid_loss_sL1 = []

    accuracy = []
    IU = []
    # Loop over epochs



    for epoch in range(100):
        # Training
        running_loss = 0.0
        running_loss_L1 = 0.0
        running_loss_max = 0.0
        running_loss_sL1 = 0.0
        running_smoothness = 0.0


        net.train()
        for i, data in enumerate(train_loader, 0):
            local_labels = data['labels'].float()
            local_batch = data['image'].float()
            #local_labels = local_labels[:,100:400]
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)
            net.to(device)
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs, y = net(local_batch)
            loss = criterion(outputs, local_labels/256.0)
            loss_L1 = loss_func_L1(outputs, local_labels/256.0)
            loss_max = loss_func_max(outputs, local_labels / 256.0)
            loss_sL1 = smoth_L1(outputs, local_labels / 256.0)

            smoothness = torch.mean(torch.abs(y))

            loss_sum = (10*loss + smoothness)/11
            loss_sum.backward()
            optimizer.step()
            running_loss += loss.item()
            running_loss_max += loss_max.item()
            running_loss_L1 += loss_L1.item()
            running_loss_sL1 += loss_sL1.item()
            running_smoothness += smoothness.item()

            if i % 1000 == 0:
                print('TRAIN: epoch: %d loss MSE: %.6f \t loss L1: %.6f \t loss max: %.6f \t loss sL1: %.6f \t loss smooth %.6f' % (epoch + 1, running_loss / (i + 1) / bs, running_loss_L1 / (i + 1) / bs, running_loss_max / (i + 1) / bs, running_loss_sL1 / (i + 1) / bs,running_smoothness / (i + 1) / bs))

        train_loss.append(running_loss / float(i + 1))
        train_loss_L1.append(running_loss_L1 / float(i + 1))
        train_loss_max.append(running_loss_max / float(i + 1))
        train_loss_sL1.append(running_loss_sL1 / float(i + 1))

        tool.save_checkpoint(net, epoch, loss, optimizer)

        # validation
        running_loss = 0.0
        running_loss_L1 = 0.0
        running_loss_max = 0.0
        running_loss_sL1 = 0.0
        running_smoothness = 0.0
        net.eval()
        for i, data in enumerate(valid_loader, 0):
            local_labels = data['labels'].float()
            local_batch = data['image'].float()
            #local_labels = local_labels[:,100:400]
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)
            net.to(device)
            outputs,y = net(local_batch)

            smoothness = torch.mean(torch.abs(y))

            loss = criterion(outputs, local_labels/256.0)
            loss_L1 = loss_func_L1(outputs, local_labels/256.0)
            loss_max = loss_func_max(outputs, local_labels / 256.0)
            loss_sL1 = smoth_L1(outputs, local_labels / 256.0)
            running_loss += loss.item()
            running_loss_max += loss_max.item()
            running_loss_L1 += loss_L1.item()
            running_loss_sL1 += loss_sL1.item()
            running_smoothness += smoothness.item()

        valid_loss.append(running_loss / (i + 1) / bs)
        valid_loss_L1.append(running_loss_L1 / (i + 1) / bs)
        valid_loss_max.append(running_loss_max / (i + 1) / bs)
        valid_loss_sL1.append(running_loss_sL1 / (i + 1) / bs)

        print('VALID: epoch: %d loss MSE: %.6f \t loss L1: %.6f \t loss max: %.6f \t loss sL1: %.6f \t loss ssmooth: %.6f' % (
            epoch + 1, running_loss / (i + 1) / bs, running_loss_L1 / (i + 1) / bs, running_loss_max / (i + 1) / bs,
            running_loss_sL1 / (i + 1) / bs, running_smoothness / (i + 1) / bs))

        #print('VALIDATION: epoch: %d loss MSE: %.6f' % (epoch + 1, running_loss / (i + 1) / bs))
        #print('VALIDATION: epoch: %d loss L1: %.6f' % (epoch + 1, running_loss_L1 / (i + 1) / bs))
        #print('VALIDATION: epoch: %d loss max: %.6f' % (epoch + 1, running_loss_max / (i + 1) / bs))
        #print('VALIDATION: epoch: %d loss sL1: %.6f' % (epoch + 1, running_loss_sL1 / (i + 1) / bs))

    plot_stats(valid_loss, train_loss, 'loss.png', 'r- validation loss/b - training loss')
    plot_stats(valid_loss_L1, train_loss_L1, 'loss_L1.png', 'r- validation loss/b - training loss')
    plot_stats(valid_loss_max, train_loss_max, 'loss_max.png', 'r- validation loss/b - training loss')
    plot_stats(valid_loss_sL1, train_loss_sL1, 'loss_sL1.png', 'r- validation loss/b - training loss')

    # plot_stats(accuracy, IU, 'acc_iu.png', 'r- accuracy/b - IU')
