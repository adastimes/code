import torch
from torch.autograd import Function
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from PIL import Image
import numpy as np
import torch.optim as optim


def collate_func(batch):
    """
    This function is used to filter out batches that have bad images. The DatasetLoaders are returning None
    if images are corrupted (in __getitem__).
    :param batch: The batch to look into if there is any None inside.
    :return:
    """
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


class SetSample(Function):
    @staticmethod
    def forward(ctx, input, pos):
        """
            We set one element.
        :param ctx: unused
        :param input: the input tensor
        :param poz: position where we want to make things zero
        :return:
        """
        out = input
        out[pos] = 0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        """
        Passing along the gradient.
        :param ctx: not used
        :param grad_output: previous gradient; with respect to the output
        :return:
        """
        return grad_output


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
        self.zero_out = SetSample.apply

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

    def forward(self, x, pos):
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
        x = self.zero_out(x, pos)
        return x


class CityscapeTestDataset(torch.utils.data.Dataset):

    def __init__(self, csv_file, root_dir):
        """
        Initialize the config file with images and the root directory for tha data.
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
        """
        self.frames = pd.read_csv(csv_file)
        self.root_dir = root_dir

    def __len__(self):
        """
        Return the length of tha dataset.
        :return: number of samples available.
        """
        return len(self.frames)

    def __getitem__(self, idx):
        """
        Get an image and label. Image is prepared to color channel first and then width and height.
        :param idx: index of the item.
        :return:
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.frames.iloc[idx, 1]

        try:
            image = np.array(Image.open(img_name))
            image = np.rollaxis(image, 2, 0)  # need to change to color channel first, as this is the way the network

            sample = {'image': image, 'name': img_name}

        except:
            return None

        return sample


class NetworkTool(object):
    def __init__(self, path):
        """
        :param path: This is the path where to save the file. Files saved are model, params and checkpoint_xx.pt
        """
        self.path_model = path + 'model.pt'
        self.path_params = path + 'params.pt'
        self.path_checkpoint = path + 'checkpoint_'

    def save_checkpoint(self, model, epoch, loss, optimizer):
        """
        Save a checkpoint.
        :param model: model to be saved
        :param epoch: epoch where we are with training
        :param loss: the loss where we are
        :param optimizer: optimizer setup
        """
        file_name = self.path_checkpoint + str(epoch) + '.pt'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss}, file_name)

    def load_checkpoint(self, epoch):
        """
        Load a checkpoint.
        :param epoch: epoch checkpoint that we should load
        :return: model, optimizer , epoch and loss are returned
        """
        self.path_checkpoint = self.path_checkpoint + str(epoch) + '.pt'
        model = Network()
        optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01,
                               amsgrad=False)

        checkpoint = torch.load(self.path_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']

        return model, optimizer, epoch, loss


def view_image(im, overlay):
    """
    Adds the overlay on the image and then displays it. The overlay is shifted to green.
    :param im: np.array with the image as (3,x, y)
    :param overlay: np.array with the mask as (x,y)
    """
    over = np.stack((overlay, overlay * 100, overlay))
    vec = np.moveaxis(im + over, 0, 2)
    img = Image.fromarray(vec, 'RGB')
    img.show()


def run_test(checkpoint=8):
    """
    This function parses a csv file and applyies the network over all files specified in there. For simlicity
    we just take the first image and visualize it. We also inspect if the custom layer did what is expected.

    :param checkpoint: The number of the checkpoint to load, default is 8.
    """

    batch_size = 1
    test_dataset = CityscapeTestDataset(csv_file='/mnt/nvme/work/Cityscape/leftImg8bit/test.csv',
                                        root_dir='/mnt/nvme/work/Cityscape/leftImg8bit')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=2,
                                              collate_fn=collate_func)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    tool = NetworkTool(path='')
    net, optimizer, epoch, loss = tool.load_checkpoint(checkpoint)
    net.eval()

    for i, data in enumerate(test_loader, 0):
        local_batch = data['image'].float()
        file_names = data['name']

        local_batch = local_batch.to(device)
        net.to(device)
        outputs = net(local_batch,(0,0,0,0))
        pred = torch.argmax(outputs, dim=1)  # we get the predictions
        pred = np.array(pred.cpu().numpy(), dtype='uint8')  # take things back in cpu space and make them uint8

        file = file_names[0][file_names[0].rfind('/') + 1: file_names[0].__len__()]
        im = np.array(data['image'][0, :].numpy(), dtype='uint8')

        print(outputs)
        view_image(im, pred[0, :])
        break


run_test(8)

