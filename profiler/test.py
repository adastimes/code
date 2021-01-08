import torch
import torch.nn as nn
import torch.nn.functional as F
from profiler import Profiler


class Network(nn.Module):
    def __init__(self):
        """
        Define the NN layers and the initialization for the layers with parameters.
        """
        super(Network, self).__init__()
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


def main():
    model = Network()
    checkpoint = torch.load('/mnt/nvme/adas/code/custom_layer/checkpoint_8.pt')
    model.load_state_dict(checkpoint['model_state_dict'])

    P = Profiler(model, (4, 3, 512, 256))
    total_ops, total_params, total_mult, total_add = P.profile()

    print("=====================================================")
    print("Mult Ops: %f GOps" % (total_mult / 1e9))
    print("Add Ops: %f GOps" % (total_add / 1e9))
    print("Total Ops: %f GOps" % (total_ops / 1e9))
    print("Total Parameters: %f * 10^6 " % (total_params / 1e6))


if __name__ == "__main__":
    main()
