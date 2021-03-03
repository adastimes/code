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


def plot_stats(v, t, fn, label):
    fig, ax = plt.subplots()
    ax.plot(v, 'r')
    ax.plot(t, 'b')
    ax.set(xlabel='Epochs', ylabel=label,
           title='Loss function.')
    ax.grid()
    fig.savefig(checkpoints_dir + fn)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    train_dataset = CityscapeDataset(csv_file='/mnt/nvme/work/Cityscape/leftImg8bit/train.csv',
                                     root_dir='/mnt/nvme/work/Cityscape/leftImg8bit')

    valid_dataset = CityscapeDataset(csv_file='/mnt/nvme/work/Cityscape/leftImg8bit/train.csv',
                                     root_dir='/mnt/nvme/work/Cityscape/leftImg8bit')

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=128, shuffle=True, num_workers=2,
                                               collate_fn=collate_func)

    net = Network()
    tool = NetworkTool(path=checkpoints_dir)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)

    '''
    #betas - this are the values provided in the Adam paper
    #eps - 1e-4 to 1e-8 is suggested in the paper
    #weight decay - it cannot be too much as then we prioratize small weights to the goal, fastai puts 0.01
    '''

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    train_loss = []
    valid_loss = []
    accuracy = []
    IU = []
    # Loop over epochs
    for epoch in range(80):
        # Training
        running_loss = 0.0
        net.train()
        for i, data in enumerate(train_loader, 0):
            local_labels = data['labels'].long()
            local_batch = data['image'].float()
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)
            net.to(device)
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(local_batch)
            loss = criterion(outputs, local_labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 1000 == 0:
                print('TRAINING: epoch: %d loss: %.3f' % (epoch + 1, running_loss / (i + 1)))

        train_loss.append(running_loss / float(i + 1))
        if epoch % 2 == 0:
            tool.save_checkpoint(net, epoch, loss, optimizer)

        # validation
        running_loss = 0.0
        net.eval()
        m = Metric()
        for i, data in enumerate(valid_loader, 0):
            local_labels = data['labels'].long()
            local_batch = data['image'].float()
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)
            net.to(device)
            outputs = net(local_batch)
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
