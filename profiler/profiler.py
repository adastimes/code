'''
This code computes the number of operations and the number of parameters for a NN in pytorch.

Author: John ADAS Doe
Email: john.adas.doe@gmail.com
License: Apache-2.0
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

'''
TODO: 
Need to check operation cound for conv and convtranspose

'''


class Profiler:
    def __init__(self, model, input_size):
        """
            Initialize the profile with the model and input size.
        :param model: This is the NN model
        :param input_size: The input size of following size (batch_size, channels, width, height)
        """
        self.model = model
        self.input_size = input_size

    @staticmethod
    def count_convtranspose2d(m, x, y):
        """
            Compute the number of OPs for convtranspose2d.
        :param m: The ConvTranspose2d layer object
        :param x: input tensor
        :param y: output tensor
        """
        cin = m.in_channels // m.groups
        kh, kw = m.kernel_size

        # ops per output element
        kernel_mul = kh * kw * cin
        kernel_add = kh * kw * cin - 1
        bias_ops = 1 if m.bias is not None else 0
        ops = kernel_mul + kernel_add + bias_ops

        # total ops
        num_out_elements = y.numel()
        total_ops = num_out_elements * ops

        # incase same conv is used multiple times
        m.total_ops += torch.Tensor([int(total_ops)])
        m.total_mult += torch.Tensor([int(kernel_mul * num_out_elements)])
        m.total_add += torch.Tensor([int((kernel_add + bias_ops) * num_out_elements)])
        print("ConvTranspose2D  - %f GOPs" % (float(total_ops) / 10 ** 9))

    @staticmethod
    def count_conv2d(m, x, y):
        """
            Compute the number of OPs for conv2d.
        :param m: The Conv2d layer object
        :param x: input tensor
        :param y: output tensor
        """
        x = x[0]

        cin = m.in_channels // m.groups
        kh, kw = m.kernel_size

        # ops per output element
        kernel_mul = kh * kw * cin
        kernel_add = kh * kw * cin - 1
        bias_ops = 1 if m.bias is not None else 0
        ops = kernel_mul + kernel_add + bias_ops

        # total ops
        num_out_elements = y.numel()
        total_ops = num_out_elements * ops

        # incase same conv is used multiple times
        m.total_ops += torch.Tensor([int(total_ops)])
        m.total_mult += torch.Tensor([int(kernel_mul * num_out_elements)])
        m.total_add += torch.Tensor([int((kernel_add + bias_ops) * num_out_elements)])
        print("Conv2D  - %f GOPs" % (float(total_ops) / 10 ** 9))

    @staticmethod
    def count_bn2d(m, x, y):
        """
            Counte operations for a bn2d layer
        :param m: Layer class
        :param x: input tensor
        :param y: output tensor
        """
        x = x[0]

        nelements = x.numel()
        total_sub = nelements
        total_div = nelements
        total_ops = total_sub + total_div

        m.total_ops += torch.Tensor([int(total_ops)])
        m.total_add += torch.Tensor([int(total_sub)])
        print("BN2D  - %f GOPs" % (float(total_ops) / 10 ** 9))

    @staticmethod
    def count_relu(m, x, y):
        """
            Operations in the relu layer
        :param m: Relu layer class
        :param x: input tensor
        :param y: output tensor
        """
        x = x[0]

        nelements = x.numel()
        total_ops = nelements

        m.total_ops += torch.Tensor([int(total_ops)])
        print("Relu  - %f GOPs" % (float(total_ops) / 10 ** 9))

    @staticmethod
    def count_softmax(m, x, y):
        """
            Operations in the softmax layer
        :param m: Softmax layer class
        :param x: input tensor
        :param y: output tensor
        """
        x = x[0]

        batch_size, nfeatures = x.size()

        total_exp = nfeatures
        total_add = nfeatures - 1
        total_div = nfeatures
        total_ops = batch_size * (total_exp + total_add + total_div)

        m.total_ops += torch.Tensor([int(total_ops)])
        m.total_add += torch.Tensor([int(total_add * batch_size)])
        print("SoftMax  - %f GOPs" % (float(total_ops) / 10 ** 9))

    @staticmethod
    def count_maxpool(m, x, y):
        """
            Operations in the maxpool layer
        :param m: maxpool layer class
        :param x: input tensor
        :param y: output tensor
        """
        kernel_ops = torch.prod(torch.Tensor([m.kernel_size])) - 1
        num_elements = y.numel()
        total_ops = kernel_ops * num_elements

        m.total_ops += torch.Tensor([int(total_ops)])
        print("MaxPool  - %f GOPs" % (float(total_ops) / 10 ** 9))

    @staticmethod
    def count_avgpool(m, x, y):
        """
            Operations in the average pooling layer
        :param m: avg pooling layer class
        :param x: input tensor
        :param y: output tensor
        """
        total_add = torch.prod(torch.Tensor([m.kernel_size])) - 1
        total_div = 1
        kernel_ops = total_add + total_div
        num_elements = y.numel()
        total_ops = kernel_ops * num_elements

        m.total_ops += torch.Tensor([int(total_ops)])
        m.total_add += torch.Tensor([int(total_add * num_elements)])
        print("AvgPool  - %f GOPs" % (float(total_ops) / 10 ** 9))

    @staticmethod
    def count_linear(m, x, y):
        """
            Operations in the Linear layer
        :param m: Linear layer class
        :param x: input tensor
        :param y: output tensor
        """
        # per output element
        total_mul = m.in_features
        total_add = m.in_features - 1
        num_elements = y.numel()
        total_ops = (total_mul + total_add) * num_elements

        m.total_ops += torch.Tensor([int(total_ops)])
        m.total_mult += torch.Tensor([int(total_mul * num_elements)])
        m.total_add += torch.Tensor([int(total_add * num_elements)])
        print("Linear  - %f GOPs" % (float(total_ops) / 10 ** 9))

    def profile(self):
        """
            Profile the network.
        :return: the total operations , numbar of parameters, number of multiply and number of add
        """
        self.model.eval()

        def add_hooks(m):
            """
                This function acts like a lambda and is passed to apply.It checks the layer type and then
                registers the proper hook function to that layer. The hook function computes the number
                of parameters once the model forward is done.
            :param m: Layer object
            :return: Nothing
            """
            if len(list(m.children())) > 0: return
            m.register_buffer('total_ops', torch.zeros(1))
            m.register_buffer('total_params', torch.zeros(1))
            m.register_buffer('total_mult', torch.zeros(1))
            m.register_buffer('total_add', torch.zeros(1))

            for p in m.parameters():
                m.total_params += torch.Tensor([p.numel()])

            if isinstance(m, nn.Conv2d):
                m.register_forward_hook(self.count_conv2d)
            elif isinstance(m, nn.ConvTranspose2d):
                m.register_forward_hook(self.count_convtranspose2d)
            elif isinstance(m, nn.BatchNorm2d):
                m.register_forward_hook(self.count_bn2d)
            elif isinstance(m, nn.ReLU):
                m.register_forward_hook(self.count_relu)
            elif isinstance(m, (nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d)):
                m.register_forward_hook(self.count_maxpool)
            elif isinstance(m, (nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d)):
                m.register_forward_hook(self.count_avgpool)
            elif isinstance(m, nn.Linear):
                m.register_forward_hook(self.count_linear)
            elif isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
                pass
            else:
                print("Not implemented for ", m)

        self.model.apply(add_hooks)

        x = torch.zeros(self.input_size)
        self.model(x)

        total_ops = 0
        total_params = 0
        total_mult = 0
        total_add = 0
        for m in self.model.modules():
            if len(list(m.children())) > 0: continue
            total_ops += m.total_ops
            total_params += m.total_params
            total_mult += m.total_mult
            total_add += m.total_add
        total_ops = total_ops
        total_params = total_params

        return total_ops, total_params, total_mult, total_add
