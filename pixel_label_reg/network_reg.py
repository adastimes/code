'''
Defines the NNs. One big network and one smaller network.

Author: John ADAS Doe
Email: john.adas.doe@gmail.com
License: Apache-2.0
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


class NetworkReg(nn.Module):
    def __init__(self):
        super(NetworkReg, self).__init__()

        self.conv0 = nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2)  # s1
        self.bn0 = nn.BatchNorm2d(16)
        self.relu0 = nn.ReLU()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2)  # s2
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(3, 16, kernel_size=5, stride=4, padding=2)  # s3
        self.bn2 = nn.BatchNorm2d(16)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(3, 32, kernel_size=5, stride=8, padding=2)  # s4
        self.bn3 = nn.BatchNorm2d(32)
        self.relu3 = nn.ReLU()

        self.conv3_1 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)  # s4
        self.bn3_1 = nn.BatchNorm2d(32)
        self.relu3_1 = nn.ReLU()

        self.conv3_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)  # s4
        self.bn3_2 = nn.BatchNorm2d(32)
        self.relu3_2 = nn.ReLU()

        self.convt3_4 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=4, stride=2, padding=1)

        # scale 3
        self.conv2_1 = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1)  # s3
        self.bn2_1 = nn.BatchNorm2d(16)
        self.relu2_1 = nn.ReLU()

        self.convt2_2 = nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=4, stride=2, padding=1)

        # scale 2
        self.conv1_1 = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1)  # s3
        self.bn1_1 = nn.BatchNorm2d(16)
        self.relu1_1 = nn.ReLU()

        self.convt1_2 = nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=4, stride=2, padding=1)

        self.conv0_1 = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1)

        self.conv_final = nn.Conv2d(in_channels=16,out_channels=1,kernel_size=(256,5), stride=(256,1), padding=(0,2), padding_mode='replicate')
        self.drop = nn.Dropout(p=0.01)

        nn.init.xavier_uniform_(self.conv0.weight)
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.conv3.weight)
        nn.init.xavier_uniform_(self.conv3_1.weight)
        nn.init.xavier_uniform_(self.conv3_2.weight)
        nn.init.xavier_uniform_(self.convt3_4.weight)
        nn.init.xavier_uniform_(self.conv2_1.weight)
        nn.init.xavier_uniform_(self.convt2_2.weight)
        nn.init.xavier_uniform_(self.conv1_1.weight)
        nn.init.xavier_uniform_(self.convt1_2.weight)
        nn.init.xavier_uniform_(self.conv0_1.weight)
        nn.init.xavier_uniform_(self.conv_final.weight)

    def forward(self, x):
        inp = x
        x = self.conv0(x)
        x = self.bn0(x)
        x1 = self.relu0(x)

        x = self.conv1(inp)
        x = self.bn1(x)
        x = self.drop(x)
        x2 = self.relu1(x)

        x = self.conv2(inp)
        x = self.bn2(x)
        x3 = self.relu2(x)

        # scale 4
        x = self.conv3(inp)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.drop(x)

        x = self.conv3_1(x)
        x = self.bn3_1(x)
        x = self.relu3_1(x)

        x = self.conv3_2(x)
        x = self.bn3_2(x)
        x = self.relu3_2(x)

        x = self.convt3_4(x)
        x = torch.cat((x3, x), 1)
        x = self.drop(x)

        # scale 3
        x = self.conv2_1(x)
        x = self.bn2_1(x)
        x = self.relu2_1(x)

        x = self.convt2_2(x)
        x = torch.cat((x2, x), 1)

        # scale2

        x = self.conv1_1(x)
        x = self.bn1_1(x)
        x = self.relu1_1(x)

        x = self.convt1_2(x)
        x = torch.cat((x1, x), 1)

        x = self.conv0_1(x)
        x = self.conv_final(x)
        x = torch.squeeze(x)

        return x


class NetworkRegR(nn.Module):
    def __init__(self):
        super(NetworkRegR, self).__init__()

        in_channels = 32
        out_channels = 32

        # starting of backbone
        self.conv0 = nn.Conv2d(3, out_channels, kernel_size=3, stride=2, padding=1)
        self.bn0 = nn.BatchNorm2d(out_channels)
        self.relu0 = nn.ReLU()

        # First block
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()

        self.maxpool0 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Second block
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn4 = nn.BatchNorm2d(out_channels)
        self.relu4 = nn.ReLU()

        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 3rd block
        self.conv5 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(out_channels)
        self.relu5 = nn.ReLU()

        self.conv6 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn6 = nn.BatchNorm2d(out_channels)
        self.relu6 = nn.ReLU()

        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # block 4
        self.conv7 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn7 = nn.BatchNorm2d(out_channels)
        self.relu7 = nn.ReLU()

        self.conv8 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn8 = nn.BatchNorm2d(out_channels)
        self.relu8 = nn.ReLU()

        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.lin1 = nn.Linear(in_features=32 * 16 * 32, out_features=512, bias=False)

        self.drop = nn.Dropout(p=0.01)

        self.cl = nn.Conv2d(1, 1, kernel_size=(1, 2), stride=1, padding=0)
        self.cl.weight.data[0, 0, 0, 0] = 1.0
        self.cl.weight.data[0, 0, 0, 1] = -1.0
        self.cl.bias.data[0] = 0
        self.cl.requires_grad = False


        nn.init.xavier_uniform_(self.conv0.weight)
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.conv3.weight)
        nn.init.xavier_uniform_(self.conv4.weight)
        nn.init.xavier_uniform_(self.conv5.weight)
        nn.init.xavier_uniform_(self.conv6.weight)
        nn.init.xavier_uniform_(self.conv7.weight)
        nn.init.xavier_uniform_(self.conv8.weight)

    def forward(self, x):
        x = self.conv0(x)
        x = self.bn0(x)
        x = self.relu0(x)

        # block 1
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x += identity
        x = self.relu2(x)
        x = self.maxpool0(x)
        x = self.drop(x)

        # block 2
        identity = x
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x += identity
        x = self.relu4(x)
        x = self.maxpool1(x)
        x = self.drop(x)

        # block 3
        identity = x
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu5(x)
        x = self.conv6(x)
        x = self.bn6(x)
        x += identity
        x = self.relu6(x)
        x = self.maxpool2(x)
        #x = self.drop(x)

        '''
        # block 4
        identity = x
        x = self.conv7(x)
        x = self.bn7(x)
        x = self.relu7(x)
        x = self.conv8(x)
        x = self.bn8(x)
        x += identity
        x = self.relu8(x)
        x = self.maxpool3(x)
        '''

        x = x.view(-1, 32 * 16 * 32)  # this "reshapes" the vector such that we can apply the linear layer
        x = self.lin1(x)

        y = torch.reshape(x,[x.shape[0],1,1,512])
        y = self.cl(y)

        return x,y


class NetworkRegSimple(nn.Module):
    def __init__(self):
        """
        Define the NN layers and the initialization for the layers with parameters.
        """
        super(NetworkRegSimple, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=2, padding=2)

        self.conv2_0 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=2, padding=2)

        self.conv3_0 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=2)

        self.conv4_0 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=2)

        self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=2)
        self.lin1 = nn.Linear(in_features=128 * 8 * 16, out_features=300, bias=False)
        self.drop = nn.Dropout(p=0.05)

        # init with xavier distribution as in the old days
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.conv3.weight)
        nn.init.xavier_uniform_(self.conv4.weight)
        nn.init.xavier_uniform_(self.conv5.weight)
        nn.init.xavier_uniform_(self.lin1.weight)

    def forward(self, x):
        """
        The forward pass of the network.
        :param x: Input to the NN
        :return: return the output of the network
        """
        x = F.relu(self.conv1(x))
        x = self.conv2_0(x)
        x = F.relu(self.conv2(x))
        x = self.drop(x)
        x = self.conv3_0(x)
        x = F.relu(self.conv3(x))
        x = self.drop(x)
        x = self.conv4_0(x)
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = x.view(-1, 128 * 8 * 16)  # this "reshapes" the vector such that we can apply the linear layer
        x = self.lin1(x)

        return x


class NetworkLabeling(nn.Module):
    def __init__(self):
        """
        Define the NN layers and the initialization for the layers with parameters.
        """
        super(NetworkLabeling, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=2)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=2, padding=2)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=2)

        self.convt1 = nn.ConvTranspose2d(in_channels=128, out_channels=16, kernel_size=4, stride=2, padding=1)
        self.convt2 = nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=4, stride=2, padding=1)
        self.convt3 = nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=4, stride=2, padding=1)
        self.convt4 = nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=4, stride=2, padding=1)
        self.convt5 = nn.ConvTranspose2d(in_channels=16, out_channels=2, kernel_size=4, stride=2, padding=1)

        self.drop = nn.Dropout(p=0.05)

        # init with xavier distribution as in the old days
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

    def forward(self, x):
        """
        The forward pass of the network.
        :param x: Input to the NN
        :return: return the output of the network
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.drop(x)
        x = F.relu(self.conv3(x))
        x = self.drop(x)
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))

        x = F.relu(self.convt1(x))
        x = F.relu(self.convt2(x))
        x = self.drop(x)
        x = F.relu(self.convt3(x))
        x = F.relu(self.convt4(x))
        x = self.convt5(x)

        return x


#inp = torch.rand(2, 3, 256, 512)
#net = NetworkReg()
#out = net(inp)
#print(out.shape)

