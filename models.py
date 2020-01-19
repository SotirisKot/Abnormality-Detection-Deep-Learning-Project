seed = 1997

import random
random.seed(seed)

import numpy as np
np.random.seed(seed)

import torch.nn as nn
import torch
from torch.nn.modules.module import Module
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import models

use_cuda = torch.cuda.is_available()
torch.manual_seed(seed)
if use_cuda:
    torch.cuda.manual_seed(seed)


from tqdm import tqdm

import numpy as np
np.random.seed(seed)


#  ==================================== MODELS ==================================== #


# FOR THE MLP MAYBE WE JUST NEED TO FLATTEN THE VIEWS OF A STUDY AND OUTPUT 1 NUMBER
# THE PROBA OF BEING NORMAL OR ABNORMAL (THIS IS STUPID BUT WE MUST START FROM ZERO TO HERO)

# OR ELSE WE NEED FOR EACH IMAGE TO OUTPUT A NUMBER AND TAKE THE AVERAGE
# THAT'S WHY THE AVERAGE POOLING
class MLP_With_Average_Pooling(Module):
    def __init__(self, input_dim, n_classes, hidden_1, hidden_2, hidden_3, dropout=None):
        super(MLP_With_Average_Pooling, self).__init__()

        self.input_dim   = input_dim
        self.n_classes   = n_classes
        self.hidden_1    = hidden_1
        self.hidden_2    = hidden_2
        self.hidden_3    = hidden_3
        self.leaky_relu  = F.leaky_relu
        self.dropout     = dropout

        # affine = True means it has learnable parameters

        # batchnorm does not work for input of dim: [1, x]
        # the 1 is the problem -- this happens when a patient has only 1 x-ray
        # https://discuss.pytorch.org/t/error-expected-more-than-1-value-per-channel-when-training/26274

        # self.batchnorm_1    = nn.BatchNorm1d(self.hidden_1, affine=True)
        # self.batchnorm_3    = nn.BatchNorm1d(self.hidden_3, affine=True)

        # self.batchnorm_2 = nn.BatchNorm1d(self.hidden_2, affine=True)

        self.layer_1     = nn.Linear(self.input_dim, self.hidden_1, bias=True)
        self.layer_2     = nn.Linear(self.hidden_1, self.hidden_2, bias=True)
        self.layer_3     = nn.Linear(self.hidden_2, self.hidden_3, bias=True)

        self.final_layer = nn.Linear(self.hidden_3, self.n_classes, bias=True)

    def init_weights(self):
        torch.nn.init.xavier_uniform_(self.layer_1.weight)
        torch.nn.init.xavier_uniform_(self.layer_2.weight)
        torch.nn.init.xavier_uniform_(self.layer_3.weight)
        torch.nn.init.xavier_uniform_(self.final_layer.weight)

    def forward(self, images):

        # flatten the images

        # squeeze the first dim...it is the batch_size = 1
        images  = images.squeeze(0)
        images  = images.view(images.shape[0], -1)

        output  = self.layer_1(images)

        # output  = self.batchnorm_1(output)
        output  = F.dropout(output, self.dropout, self.training)

        output  = self.leaky_relu(output)

        output  = self.layer_2(output)
        output  = self.leaky_relu(output)

        output  = self.layer_3(output)

        # output = self.batchnorm_3(output)
        output = F.dropout(output, self.dropout, self.training)

        output  = self.leaky_relu(output)

        output  = self.final_layer(output)
        output  = torch.mean(output)

        # do not add sigmoid...BCEWithLogitsLoss does this
        return output.unsqueeze(-1)


# =========================== CNNs MODELS =========================== #
class CNN_With_Average_Pooling(Module):
    def __init__(self, input_channels, n_classes, n_filters_1, n_filters_2, dropout=None):
        super(CNN_With_Average_Pooling, self).__init__()

        self.input_channels   = input_channels
        self.n_classes        = n_classes
        self.n_filters_1      = n_filters_1
        self.n_filters_2      = n_filters_2

        self.maxpool          = nn.MaxPool2d(kernel_size=3)
        self.leaky_relu       = F.leaky_relu
        self.dropout          = dropout

        self.conv1            = nn.Conv2d(3, n_filters_1, kernel_size=5)
        self.conv2            = nn.Conv2d(n_filters_1, n_filters_2, kernel_size=5)

        self.final_layer      = nn.Linear(1620, self.n_classes, bias=True)

    def init_weights(self, scale=1e-4):

        for param in self.conv1.parameters():
            nn.init.uniform_(param, a=-scale, b=scale)

        for param in self.conv2.parameters():
            nn.init.uniform_(param, a=-scale, b=scale)

        torch.nn.init.xavier_uniform_(self.final_layer.weight)

    def forward(self, images):

        # flatten the images

        # squeeze the first dim...it is the batch_size = 1
        images  = images.squeeze(0)

        # images  = images.view(images.shape[0], -1)

        output  = self.leaky_relu(self.maxpool(self.conv1(images)))

        # TODO add it later
        # output  = F.dropout(output, self.dropout, self.training)

        output = self.leaky_relu(self.maxpool(self.conv2(output)))

        # flatten the output

        # TODO add it later
        # output = F.dropout(output, self.dropout, self.training)

        output  = output.view(images.shape[0], -1)

        output  = self.final_layer(output)

        output  = torch.mean(output)

        # do not add sigmoid...BCEWithLogitsLoss does this
        return output.unsqueeze(-1)


# =========================== FROM TORCHVISION =========================== #
# https://github.com/pytorch/vision/blob/master/torchvision/models/densenet.py

model_urls = {
    'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',
    'densenet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth',
    'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth',
    'densenet161': 'https://download.pytorch.org/models/densenet161-8d451a50.pth',
}


class PretrainedDensenet(nn.Module):
    def __init__(self, model_name, num_class=1):
        super().__init__()

        self.channels = 1664
        densenet_169 = models.densenet169(pretrained=True)

        # densenet_121 = models.densenet121(pretrained=True)
        # how_many = 0
        # for params in densenet_169.parameters():
        #     params.requires_grad_(False)
        #     how_many += 1
        #     if how_many == 400:
        #         break

        # here we get the part of the model where the feature extraction is happening
        # in that way we can add on top of the feature extractor our own classifier as in the MURA paper
        # if we used densenet169.classifier we would get the final linear layer used for classification
        self.features = nn.Sequential(*list(densenet_169.features.children()))

        # freeze certain blocks
        for mod in self.features[:9]:
            mod.requires_grad_(False)

        self.relu = nn.ReLU(inplace=True)
        self.leaky_relu = F.leaky_relu
        self.fc1 = nn.Linear(self.channels, num_class)

        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        # we must squeeze the first dimension ---> it is the batch_size
        x        = x.squeeze(0)

        # TODO for later...upsample the image to 224x224

        features = self.features(x)
        out      = self.leaky_relu(features)

        # applies average_pooling but it is adaptive..because it can reduce the dimensions
        # to whatever we like
        # out has dimensions of [#views, 1664, 10, 10]
        # using adaptive_avg_pool2d we can make it:
        # [#views, 1664, 1, 1] --> we do not specify a kernel_size
        # --> pytorch infers it (this is why it is called adaptive)
        out      = nn.functional.adaptive_avg_pool2d(out, (1, 1))

        out      = out.view(-1, self.channels)
        out      = self.fc1(out)

        return torch.mean(torch.sigmoid(out)).unsqueeze(0)

