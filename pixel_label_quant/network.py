'''
Defines the NNs. One big network and one smaller network.

Author: John ADAS Doe
Email: john.adas.doe@gmail.com
License: Apache-2.0
'''

import torch
import torch.nn as nn


class Network(nn.Module):
    def __init__(self):
        """
        Define the NN layers and the initialization for the layers with parameters.
        """
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=2, padding=2)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=2, padding=2)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=2, padding=2)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=2, padding=2)
        self.relu4 = nn.ReLU()
        self.conv5 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=2, padding=2)
        self.relu5 = nn.ReLU()

        self.convt1 = nn.ConvTranspose2d(in_channels=32, out_channels=8, kernel_size=4, stride=2, padding=1)
        self.convt2 = nn.ConvTranspose2d(in_channels=8, out_channels=8, kernel_size=4, stride=2, padding=1)
        self.convt3 = nn.ConvTranspose2d(in_channels=8, out_channels=8, kernel_size=4, stride=2, padding=1)
        self.convt4 = nn.ConvTranspose2d(in_channels=8, out_channels=8, kernel_size=4, stride=2, padding=1)
        self.convt5 = nn.ConvTranspose2d(in_channels=8, out_channels=2, kernel_size=4, stride=2, padding=1)

        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()


        self.drop = nn.Dropout(p=0.05)

        # init with xavier distribution as in the old days
        '''
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.conv3.weight)
        nn.init.xavier_uniform_(self.conv4.weight)
        nn.init.xavier_uniform_(self.conv5.weight)

        nn.init.xavier_uniform_(self.convt1.weight)
        nn.init.xavier_uniform_(self.convt2.weight)
        nn.init.xavier_uniform_(self.convt3.weight)
        nn.init.xavier_uniform_(self.convt4.weight)
        nn.init.xavier_uniform_(self.convt5.weight)
        '''

    def forward(self, x):
        """
        The forward pass of the network.
        :param x: Input to the NN
        :return: return the output of the network
        """
        x = self.quant(x)
        x= self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        #x = self.drop(x)
        x = self.conv3(x)
        x = self.relu3(x)
        #x = self.drop(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.conv5(x)
        x = self.relu5(x)

        x = self.convt1(x)
        x = self.convt2(x)
        #x = self.drop(x)
        x = self.convt3(x)
        x = self.convt4(x)
        x = self.convt5(x)
        x = self.dequant(x)
        #print(x.shape)

        return x
