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

        # BATCHNORM DOES NOT WORK FOR INPUT OF DIM: [1, X]
        # THE 1 IS THE PROBLEM -- THIS HAPPENS WHEN A PATIENT HAS ONLY 1 X-RAY
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

        self.hidden_layer     = nn.Linear(1620, 500, bias=True)
        # self.hidden_layer2    = nn.Linear(500, 100, bias=True)
        self.final_layer      = nn.Linear(500, self.n_classes, bias=True)  # for 100x100
        # self.final_layer = nn.Linear(20*23*23, self.n_classes, bias=True)

    def init_weights(self, scale=1e-4):

        for param in self.conv1.parameters():
            nn.init.uniform_(param, a=-scale, b=scale)

        for param in self.conv2.parameters():
            nn.init.uniform_(param, a=-scale, b=scale)

        torch.nn.init.xavier_uniform_(self.hidden_layer.weight)
        # torch.nn.init.xavier_uniform_(self.hidden_layer2.weight)
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

        # MLP
        output = self.leaky_relu(self.hidden_layer(output))

        # output = self.leaky_relu(self.hidden_layer2(output))

        # Perform max pooling amongst the views
        output = torch.max(output, dim=0)[0].view(-1)

        output  = self.final_layer(output)

        # ALT METHOD
        # output  = torch.mean(output)

        # do not add sigmoid...BCEWithLogitsLoss does this
        # ALT METHOD
        # return output.unsqueeze(-1)
        return output


# =========================== FROM TORCHVISION =========================== #
# https://github.com/pytorch/vision/blob/master/torchvision/models/densenet.py


class PretrainedDensenet(nn.Module):
    def __init__(self, num_class=1):
        super().__init__()

        self.channels = 81536
        densenet_169 = models.densenet169(pretrained=True)

        # HERE WE GET THE PART OF THE MODEL WHERE THE FEATURE EXTRACTION IS HAPPENING
        # IN THAT WAY WE CAN ADD ON TOP OF THE FEATURE EXTRACTOR OUR OWN CLASSIFIER AS IN THE MURA PAPER
        # IF WE USED DENSENET169.CLASSIFIER WE WOULD GET THE FINAL LINEAR LAYER USED FOR CLASSIFICATION
        self.features = nn.Sequential(*list(densenet_169.features.children()))

        # freeze certain blocks
        for mod in self.features[:9]:
            mod.requires_grad_(False)

        self.leaky_relu = F.leaky_relu

        self.hidden_layer = nn.Linear(self.channels, 500, bias=True)
        self.final_layer = nn.Linear(500, num_class, bias=True)

        # self.sigmoid = nn.Sigmoid()

    def init_weights(self, scale=1e-4):

        torch.nn.init.xavier_uniform_(self.hidden_layer.weight)
        torch.nn.init.xavier_uniform_(self.final_layer.weight)

    def forward(self, x):

        # we must squeeze the first dimension ---> it is the batch_size
        x        = x.squeeze(0)

        # TODO for later...upsample the image to 224x224

        features = self.features(x)
        out      = self.leaky_relu(features)

        # APPLIES AVERAGE_POOLING BUT IT IS ADAPTIVE..BECAUSE IT CAN REDUCE THE DIMENSIONS
        # TO WHATEVER WE LIKE
        # OUT HAS DIMENSIONS OF [#VIEWS, 1664, 10, 10]
        # USING ADAPTIVE_AVG_POOL2D WE CAN MAKE IT:
        # [#VIEWS, 1664, 1, 1] --> WE DO NOT SPECIFY A KERNEL_SIZE
        # --> PYTORCH INFERS IT (THIS IS WHY IT IS CALLED ADAPTIVE)
        # out      = nn.functional.adaptive_avg_pool2d(out, (1, 1))

        # out      = out.view(-1, self.channels)
        # out      = self.fc1(out)

        # NEW STUFF

        out = out.view(out.shape[0], -1)

        # MLP
        out = self.leaky_relu(self.hidden_layer(out))

        # output = self.leaky_relu(self.hidden_layer2(output))

        # Perform max pooling amongst the views
        out = torch.max(out, dim=0)[0].view(-1)

        out = self.final_layer(out)

        # return torch.mean(torch.sigmoid(out)).unsqueeze(0)

        return out



class PretrainedResnet(nn.Module):
    def __init__(self, num_class=1):
        super().__init__()

        self.channels = 2048
        resnet101 = models.resnet101(pretrained=True)

        # HERE WE GET ALL THE LAYERS EXCEPT FROM THE FC
        self.features = nn.Sequential(*list(resnet101.children())[:-1])

        # freeze certain blocks
        for mod in self.features[:7]:
            mod.requires_grad_(False)

        self.leaky_relu = F.leaky_relu
        self.fc1 = nn.Linear(self.channels, num_class)

        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # we must squeeze the first dimension ---> it is the batch_size
        x = x.squeeze(0)

        # TODO for later...upsample the image to 224x224 or maybe just use resize?? ask makis

        features = self.features(x)

        out = self.leaky_relu(features)

        out = out.view(-1, self.channels)
        out = self.fc1(out)

        return torch.mean(torch.sigmoid(out)).unsqueeze(0)
