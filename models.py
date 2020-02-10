"""

In this file we define all the models tha twe use

"""
import random
import numpy as np
import torch.nn as nn
import torch
from torch.nn.modules.module import Module
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import models
from tqdm import tqdm

seed = 1997

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

use_cuda = torch.cuda.is_available()
if use_cuda:
    torch.cuda.manual_seed(seed)


#  ==================================== MODELS ==================================== #


# FOR THE MLP MAYBE WE JUST NEED TO FLATTEN THE VIEWS OF A STUDY AND OUTPUT 1 NUMBER
# THE PROBA OF BEING NORMAL OR ABNORMAL (THIS IS STUPID BUT WE MUST START FROM ZERO TO HERO)

# OR ELSE WE NEED FOR EACH IMAGE TO OUTPUT A NUMBER AND TAKE THE AVERAGE
# THAT'S WHY THE AVERAGE POOLING

"""
MLP model:
 Flattened pixels as input
 2 hidden layers
 Leaky relu activation 
 Average Voting on output of each View
"""
class MLP_With_Average_Voting(Module):
    def __init__(self, input_dim, n_classes, hidden_1, hidden_2, hidden_3, dropout=None):
        super(MLP_With_Average_Voting, self).__init__()

        self.input_dim   = input_dim
        self.n_classes   = n_classes
        self.hidden_1    = hidden_1
        self.hidden_2    = hidden_2
        self.hidden_3    = hidden_3
        self.leaky_relu  = F.leaky_relu
        # self.dropout     = dropout

        # affine = True means it has learnable parameters

        # BATCHNORM DOES NOT WORK FOR INPUT OF DIM: [1, X]
        # THE 1 IS THE PROBLEM -- THIS HAPPENS WHEN A PATIENT HAS ONLY 1 X-RAY
        # https://discuss.pytorch.org/t/error-expected-more-than-1-value-per-channel-when-training/26274

        # self.batchnorm_1    = nn.BatchNorm1d(self.hidden_1, affine=True)
        # self.batchnorm_3    = nn.BatchNorm1d(self.hidden_3, affine=True)

        # self.batchnorm_2 = nn.BatchNorm1d(self.hidden_2, affine=True)

        # Initialize the layers of the MLP
        self.layer_1     = nn.Linear(self.input_dim, self.hidden_1, bias=True)
        self.layer_2     = nn.Linear(self.hidden_1, self.hidden_2, bias=True)
        self.layer_3     = nn.Linear(self.hidden_2, self.hidden_3, bias=True)

        self.final_layer = nn.Linear(self.hidden_3, self.n_classes, bias=True)

    def init_weights(self):
        # Initialize the weights of the model
        torch.nn.init.xavier_uniform_(self.layer_1.weight)
        torch.nn.init.xavier_uniform_(self.layer_2.weight)
        torch.nn.init.xavier_uniform_(self.layer_3.weight)
        torch.nn.init.xavier_uniform_(self.final_layer.weight)

    def forward(self, images):

        # Squeeze the first dim...it is the batch_size = 1
        images  = images.squeeze(0)
        # Flatten the images
        images  = images.view(images.shape[0], -1)

        # Pass the input through the layers

        output  = self.layer_1(images)

        # output  = self.batchnorm_1(output)
        # output  = F.dropout(output, self.dropout, self.training)

        output  = self.leaky_relu(output)

        output  = self.layer_2(output)
        output  = self.leaky_relu(output)

        output  = self.layer_3(output)

        # output = self.batchnorm_3(output)
        # output = F.dropout(output, self.dropout, self.training)

        output  = self.leaky_relu(output)

        output  = self.final_layer(output)

        # Perform average voting on views
        output  = torch.mean(output)

        # do not add sigmoid...BCEWithLogitsLoss does this
        return output.unsqueeze(-1)


"""
MLP model:
 Flattened pixels as input
 2 hidden layers
 Leaky relu activation 
 Max Pooling over Views and then final layer to get one output
"""
class MLP_With_Max_Pooling(Module):
    def __init__(self, input_dim, n_classes, hidden_1, hidden_2, hidden_3, dropout=None):
        super(MLP_With_Max_Pooling, self).__init__()

        self.input_dim   = input_dim
        self.n_classes   = n_classes
        self.hidden_1    = hidden_1
        self.hidden_2    = hidden_2
        self.hidden_3    = hidden_3
        self.leaky_relu  = F.leaky_relu
        # self.dropout     = dropout

        # affine = True means it has learnable parameters

        # BATCHNORM DOES NOT WORK FOR INPUT OF DIM: [1, X]
        # THE 1 IS THE PROBLEM -- THIS HAPPENS WHEN A PATIENT HAS ONLY 1 X-RAY
        # https://discuss.pytorch.org/t/error-expected-more-than-1-value-per-channel-when-training/26274

        # self.batchnorm_1    = nn.BatchNorm1d(self.hidden_1, affine=True)
        # self.batchnorm_3    = nn.BatchNorm1d(self.hidden_3, affine=True)

        # self.batchnorm_2 = nn.BatchNorm1d(self.hidden_2, affine=True)

        # Initialize layers of model

        self.layer_1     = nn.Linear(self.input_dim, self.hidden_1, bias=True)
        self.layer_2     = nn.Linear(self.hidden_1, self.hidden_2, bias=True)
        self.layer_3     = nn.Linear(self.hidden_2, self.hidden_3, bias=True)

        self.final_layer = nn.Linear(self.hidden_3, self.n_classes, bias=True)

    def init_weights(self):
        # Initialize the weights of the model
        torch.nn.init.xavier_uniform_(self.layer_1.weight)
        torch.nn.init.xavier_uniform_(self.layer_2.weight)
        torch.nn.init.xavier_uniform_(self.layer_3.weight)
        torch.nn.init.xavier_uniform_(self.final_layer.weight)

    def forward(self, images):

        # Squeeze the first dim...it is the batch_size = 1
        images  = images.squeeze(0)
        # Flatten the images
        images  = images.view(images.shape[0], -1)

        output  = self.layer_1(images)

        # output  = self.batchnorm_1(output)
        # output  = F.dropout(output, self.dropout, self.training)

        # Pass the input through the model

        output  = self.leaky_relu(output)

        output  = self.layer_2(output)
        output  = self.leaky_relu(output)

        output  = self.layer_3(output)

        # output = self.batchnorm_3(output)
        # output = F.dropout(output, self.dropout, self.training)

        output  = self.leaky_relu(output)

        # Perform max pooling over the views
        output = torch.max(output, dim=0)[0].view(-1)

        # Pass through the final layer
        output  = self.final_layer(output)

        # do not add sigmoid...BCEWithLogitsLoss does this
        return output


# =========================== CNNs MODELS =========================== #
"""
CNN Model
    2 Conv Layers
    2 Max Pooling Layers
    Leaky Relu for activation function
    Linear for output
    Average Voting on output of each View
"""
class CNN_With_Average_Voting(Module):
    def __init__(self, input_channels, input_shape, n_classes, n_filters_1, n_filters_2, dropout=None):
        super(CNN_With_Average_Voting, self).__init__()

        self.input_channels = input_channels
        self.input_shape   = input_shape
        self.n_classes        = n_classes
        self.n_filters_1      = n_filters_1
        self.n_filters_2      = n_filters_2

        self.leaky_relu       = F.leaky_relu
        self.dropout          = dropout

        # Initialize Convolution Layers
        self.conv1            = nn.Conv2d(self.input_channels, n_filters_1, kernel_size=5)
        self.conv2            = nn.Conv2d(n_filters_1, n_filters_2, kernel_size=5)

        # Initialize Max Pooling Layers
        self.maxpool = nn.MaxPool2d(kernel_size=3)

        # intermediate_size1 = (input_shape[0] - self.conv1.kernel_size[0] + 1) // self.conv1.stride[0]
        # intermediate_size1 = intermediate_size1 // self.maxpool.kernel_size
        # intermediate_size1 = (intermediate_size1 - self.conv2.kernel_size[0] + 1) // self.conv2.stride[0]
        # intermediate_size1 = intermediate_size1 // self.maxpool.kernel_size
        # intermediate_size2 = (input_shape[1] - self.conv1.kernel_size[1] + 1) // self.conv1.stride[1]
        # intermediate_size2 = intermediate_size2 // self.maxpool.kernel_size
        # intermediate_size2 = (intermediate_size2 - self.conv2.kernel_size[1] + 1) // self.conv2.stride[1]
        # intermediate_size2 = intermediate_size2 // self.maxpool.kernel_size

        # Initialize Linear layer
        self.final_layer      = nn.Linear(n_filters_2 * 23 * 23, self.n_classes, bias=True)  # ONLY TRUE FOR 224x224

    def init_weights(self, scale=1e-4):
        # Initialize weights of model
        for param in self.conv1.parameters():
            nn.init.uniform_(param, a=-scale, b=scale)

        for param in self.conv2.parameters():
            nn.init.uniform_(param, a=-scale, b=scale)

        torch.nn.init.xavier_uniform_(self.final_layer.weight)

    def forward(self, images):
        # Squeeze the first dim...it is the batch_size = 1
        images  = images.squeeze(0)

        # images  = images.view(images.shape[0], -1)

        # Pass the input from the conv and max pool layers

        output  = self.leaky_relu(self.maxpool(self.conv1(images)))

        # TODO add it later
        # output  = F.dropout(output, self.dropout, self.training)

        output = self.leaky_relu(self.maxpool(self.conv2(output)))

        # flatten the output

        # TODO add it later
        # output = F.dropout(output, self.dropout, self.training)

        # Flatten the output
        output  = output.view(images.shape[0], -1)

        # Pass though linear layer
        output  = self.final_layer(output)

        # Perform average voting
        output  = torch.mean(output)

        # do not add sigmoid...BCEWithLogitsLoss does this
        return output.unsqueeze(-1)


"""
CNN Model
    2 Conv Layers
    2 Max Pooling Layers
    Leaky Relu for activation function
    Linear for output
    Max Pooling over Views and then final layer to get one output 
"""
class CNN_With_Max_Pooling(Module):
    def __init__(self, input_channels, input_shape, n_classes, n_filters_1, n_filters_2, dropout=None):
        super(CNN_With_Max_Pooling, self).__init__()

        self.input_channels = input_channels
        self.input_shape   = input_shape
        self.n_classes        = n_classes
        self.n_filters_1      = n_filters_1
        self.n_filters_2      = n_filters_2

        self.leaky_relu       = F.leaky_relu
        self.dropout          = dropout

        # Initialize convolution layers
        self.conv1            = nn.Conv2d(self.input_channels, n_filters_1, kernel_size=5)
        self.conv2            = nn.Conv2d(n_filters_1, n_filters_2, kernel_size=5)

        # Initialize max pooling layers
        self.maxpool          = nn.MaxPool2d(kernel_size=3)

        # intermediate_size1 = (input_shape[0] - self.conv1.kernel_size[0] + 1) // self.conv1.stride[0]
        # intermediate_size1 = intermediate_size1 // self.maxpool.kernel_size
        # intermediate_size1 = (intermediate_size1 - self.conv2.kernel_size[0] + 1) // self.conv2.stride[0]
        # intermediate_size1 = intermediate_size1 // self.maxpool.kernel_size
        # intermediate_size2 = (input_shape[1] - self.conv1.kernel_size[1] + 1) // self.conv1.stride[1]
        # intermediate_size2 = intermediate_size2 // self.maxpool.kernel_size
        # intermediate_size2 = (intermediate_size2 - self.conv2.kernel_size[1] + 1) // self.conv2.stride[1]
        # intermediate_size2 = intermediate_size2 // self.maxpool.kernel_size

        # Initialize linear layer
        self.final_layer      = nn.Linear(n_filters_2 * 23 * 23, self.n_classes, bias=True)  # ONLY TRUE FOR 224x224

    def init_weights(self, scale=1e-4):
        # Initialize weights of the model
        for param in self.conv1.parameters():
            nn.init.uniform_(param, a=-scale, b=scale)

        for param in self.conv2.parameters():
            nn.init.uniform_(param, a=-scale, b=scale)

        torch.nn.init.xavier_uniform_(self.final_layer.weight)

    def forward(self, images):
        # Squeeze the first dim...it is the batch_size = 1
        images  = images.squeeze(0)

        # images  = images.view(images.shape[0], -1)

        # Pass the input from the conv and max pool layers

        output  = self.leaky_relu(self.maxpool(self.conv1(images)))

        # TODO add it later
        # output  = F.dropout(output, self.dropout, self.training)

        output = self.leaky_relu(self.maxpool(self.conv2(output)))

        # TODO add it later
        # output = F.dropout(output, self.dropout, self.training)

        # Flatten the output
        output  = output.view(images.shape[0], -1)

        # Perform max pooling over the Views
        output  = torch.max(output, dim=0)[0].view(-1)

        # Pass though the linear layer
        output  = self.final_layer(output)

        # do not add sigmoid...BCEWithLogitsLoss does this
        return output


"""
CNN Model
    2 Conv Layers
    2 Max Pooling Layers
    Leaky Relu for activation function
    MPL with 1 hidden for output
    Average Voting on output of each View
"""
class CNN_MLP_Average_Voting(Module):
    def __init__(self, input_channels, input_shape, n_classes, n_filters_1, n_filters_2, hidden_size, dropout=None):
        super(CNN_MLP_Average_Voting, self).__init__()

        self.input_channels = input_channels
        self.input_shape      = input_shape
        self.n_classes        = n_classes
        self.n_filters_1      = n_filters_1
        self.n_filters_2      = n_filters_2
        self.hidden_size      = hidden_size

        self.leaky_relu       = F.leaky_relu
        self.dropout          = dropout

        # Initialize convolution layers
        self.conv1            = nn.Conv2d(self.input_channels, n_filters_1, kernel_size=5)
        self.conv2            = nn.Conv2d(n_filters_1, n_filters_2, kernel_size=5)

        # Initialize max poooling layers
        self.maxpool          = nn.MaxPool2d(kernel_size=3)

        # Place holder for gradients for cam
        self.cam_gradients = None

        # intermediate_size1 = (input_shape[0] - self.conv1.kernel_size[0] + 1) // self.conv1.stride[0]
        # intermediate_size1 = intermediate_size1 // self.maxpool.kernel_size
        # intermediate_size1 = (intermediate_size1 - self.conv2.kernel_size[0] + 1) // self.conv2.stride[0]
        # intermediate_size1 = intermediate_size1 // self.maxpool.kernel_size
        # intermediate_size2 = (input_shape[1] - self.conv1.kernel_size[1] + 1) // self.conv1.stride[1]
        # intermediate_size2 = intermediate_size2 // self.maxpool.kernel_size
        # intermediate_size2 = (intermediate_size2 - self.conv2.kernel_size[1] + 1) // self.conv2.stride[1]
        # intermediate_size2 = intermediate_size2 // self.maxpool.kernel_size

        # Initialize MLP
        self.hidden_layer     = nn.Linear(n_filters_2*23*23, self.hidden_size, bias=True)  # ONLY TRUE FOR 224x224
        self.final_layer      = nn.Linear(self.hidden_size, self.n_classes, bias=True)

    def init_weights(self, scale=1e-4):
        # Initialize weights for model
        for param in self.conv1.parameters():
            nn.init.uniform_(param, a=-scale, b=scale)

        for param in self.conv2.parameters():
            nn.init.uniform_(param, a=-scale, b=scale)

        torch.nn.init.xavier_uniform_(self.hidden_layer.weight)
        torch.nn.init.xavier_uniform_(self.final_layer.weight)

    def forward(self, images):
        # Squeeze the first dim...it is the batch_size = 1
        images  = images.squeeze(0)

        # images  = images.view(images.shape[0], -1)

        # Pass the input from the conv and max pool layers

        output  = self.leaky_relu(self.maxpool(self.conv1(images)))

        # TODO add it later
        # output  = F.dropout(output, self.dropout, self.training)

        output = self.leaky_relu(self.maxpool(self.conv2(output)))

        # TODO add it later
        # output = F.dropout(output, self.dropout, self.training)

        # Flatten the output

        output  = output.view(images.shape[0], -1)

        # Pass through MLP
        output = self.leaky_relu(self.hidden_layer(output))

        # output = self.leaky_relu(self.hidden_layer2(output))

        output  = self.final_layer(output)

        # Perform average pooling
        output  = torch.mean(output)

        # do not add sigmoid...BCEWithLogitsLoss does this
        return output.unsqueeze(-1)


    def forward_cam(self, images):
        # Squeeze the first dim...it is the batch_size = 1
        images = images.squeeze(0)

        # images  = images.view(images.shape[0], -1)

        # Pass the input from the conv and max pool layers

        output = self.leaky_relu(self.maxpool(self.conv1(images)))

        # TODO add it later
        # output  = F.dropout(output, self.dropout, self.training)

        output = self.conv2(output)

        # Register hook for gradients
        h = output.register_hook(self.activations_hook)

        output = self.leaky_relu(self.maxpool(output))

        # TODO add it later
        # output = F.dropout(output, self.dropout, self.training)

        # Flatten the output
        output = output.view(images.shape[0], -1)

        # Pass through MLP

        output = self.leaky_relu(self.hidden_layer(output))

        # output = self.leaky_relu(self.hidden_layer2(output))

        output = self.final_layer(output)

        # Perform average voting
        output = torch.mean(output)

        # do not add sigmoid...BCEWithLogitsLoss does this
        return output.unsqueeze(-1)

    def activations_hook(self, grad):
        self.cam_gradients = grad

    def get_activations_gradient(self):
        return self.cam_gradients

    def get_activations(self, x):
        # flatten the images

        # squeeze the first dim...it is the batch_size = 1
        x = x.squeeze(0)

        # images  = images.view(images.shape[0], -1)

        output = self.leaky_relu(self.maxpool(self.conv1(x)))

        # TODO add it later
        # output  = F.dropout(output, self.dropout, self.training)

        output = self.conv2(output)

        return output


"""
CNN Model
    2 Conv Layers
    2 Max Pooling Layers
    Leaky Relu for activation function
    MPL with 1 hidden + Linear for output
    Max Pooling over Views and then final layer to get one output 
"""
class CNN_MLP_Max_Pooling(Module):
    def __init__(self, input_channels, input_shape, n_classes, n_filters_1, n_filters_2, hidden_size, dropout=None):
        super(CNN_MLP_Max_Pooling, self).__init__()

        self.input_channels = input_channels
        self.input_shape   = input_shape
        self.n_classes        = n_classes
        self.n_filters_1      = n_filters_1
        self.n_filters_2      = n_filters_2
        self.hidden_size      = hidden_size

        self.leaky_relu       = F.leaky_relu
        self.dropout          = dropout

        # Initialize convolution layers
        self.conv1            = nn.Conv2d(self.input_channels, n_filters_1, kernel_size=5)
        self.conv2            = nn.Conv2d(n_filters_1, n_filters_2, kernel_size=5)

        # Initialize max pooling layers
        self.maxpool          = nn.MaxPool2d(kernel_size=3)

        # Place holder for gradients for cam
        self.cam_gradients = None

        # intermediate_size1 = (input_shape[0] - self.conv1.kernel_size[0] + 1) // self.conv1.stride[0]
        # intermediate_size1 = intermediate_size1 // self.maxpool.kernel_size
        # intermediate_size1 = (intermediate_size1 - self.conv2.kernel_size[0] + 1) // self.conv2.stride[0]
        # intermediate_size1 = intermediate_size1 // self.maxpool.kernel_size
        # intermediate_size2 = (input_shape[1] - self.conv1.kernel_size[1] + 1) // self.conv1.stride[1]
        # intermediate_size2 = intermediate_size2 // self.maxpool.kernel_size
        # intermediate_size2 = (intermediate_size2 - self.conv2.kernel_size[1] + 1) // self.conv2.stride[1]
        # intermediate_size2 = intermediate_size2 // self.maxpool.kernel_size

        # Initialize MLP
        self.hidden_layer     = nn.Linear(n_filters_2*23*23, self.hidden_size, bias=True)  # ONLY TRUE FOR 224x224
        self.final_layer      = nn.Linear(self.hidden_size, self.n_classes, bias=True)

    def init_weights(self, scale=1e-4):
        # Initialize weights of the model
        for param in self.conv1.parameters():
            nn.init.uniform_(param, a=-scale, b=scale)

        for param in self.conv2.parameters():
            nn.init.uniform_(param, a=-scale, b=scale)

        torch.nn.init.xavier_uniform_(self.hidden_layer.weight)
        torch.nn.init.xavier_uniform_(self.final_layer.weight)

    def forward(self, images):
        # Squeeze the first dim...it is the batch_size = 1
        images  = images.squeeze(0)

        # images  = images.view(images.shape[0], -1)

        # Pass the input from the conv and max pool layers

        output  = self.leaky_relu(self.maxpool(self.conv1(images)))

        # TODO add it later
        # output  = F.dropout(output, self.dropout, self.training)

        output = self.leaky_relu(self.maxpool(self.conv2(output)))

        # TODO add it later
        # output = F.dropout(output, self.dropout, self.training)

        # Flatten the output
        output  = output.view(images.shape[0], -1)

        # Pass through MLP

        output = self.leaky_relu(self.hidden_layer(output))

        # output = self.leaky_relu(self.hidden_layer2(output))

        # Perform max pooling over the Views
        output = torch.max(output, dim=0)[0].view(-1)

        output  = self.final_layer(output)

        # do not add sigmoid...BCEWithLogitsLoss does this
        return output

    def forward_cam(self, images):
        # Squeeze the first dim...it is the batch_size = 1
        images = images.squeeze(0)

        # images  = images.view(images.shape[0], -1)

        # Pass the input from the conv and max pool layers

        output = self.leaky_relu(self.maxpool(self.conv1(images)))

        # TODO add it later
        # output  = F.dropout(output, self.dropout, self.training)

        output = self.conv2(output)

        # Register hook for gradients
        h = output.register_hook(self.activations_hook)

        output = self.leaky_relu(self.maxpool(output))

        # TODO add it later
        # output = F.dropout(output, self.dropout, self.training)

        # Flatten the output
        output = output.view(images.shape[0], -1)

        # Pass through MLP

        output = self.leaky_relu(self.hidden_layer(output))

        # output = self.leaky_relu(self.hidden_layer2(output))

        # Perform max pooling over the Views
        output = torch.max(output, dim=0)[0].view(-1)

        output = self.final_layer(output)

        # do not add sigmoid...BCEWithLogitsLoss does this
        return output

    def activations_hook(self, grad):
        self.cam_gradients = grad

    def get_activations_gradient(self):
        return self.cam_gradients

    def get_activations(self, x):
        # flatten the images

        # squeeze the first dim...it is the batch_size = 1
        x = x.squeeze(0)

        # images  = images.view(images.shape[0], -1)

        output = self.leaky_relu(self.maxpool(self.conv1(x)))

        # TODO add it later
        # output  = F.dropout(output, self.dropout, self.training)

        output = self.conv2(output)

        return output


"""
CNN Model
    2 Conv Layers
    2 Max Pooling Layers
    Relu for activation function
    Linear for output
    Average Voting on output of each View
"""
class CNN_With_Average_VotingRELU(Module):
    def __init__(self, input_channels, input_shape, n_classes, n_filters_1, n_filters_2, dropout=None):
        super(CNN_With_Average_VotingRELU, self).__init__()

        self.input_channels = input_channels
        self.input_shape   = input_shape
        self.n_classes        = n_classes
        self.n_filters_1      = n_filters_1
        self.n_filters_2      = n_filters_2

        self.relu       = F.relu
        self.dropout          = dropout

        # Initialize convolution layers
        self.conv1            = nn.Conv2d(self.input_channels, n_filters_1, kernel_size=5)
        self.conv2            = nn.Conv2d(n_filters_1, n_filters_2, kernel_size=5)

        # Initialize max pooling layers
        self.maxpool          = nn.MaxPool2d(kernel_size=3)

        # intermediate_size1 = (input_shape[0] - self.conv1.kernel_size[0] + 1) // self.conv1.stride[0]
        # intermediate_size1 = intermediate_size1 // self.maxpool.kernel_size
        # intermediate_size1 = (intermediate_size1 - self.conv2.kernel_size[0] + 1) // self.conv2.stride[0]
        # intermediate_size1 = intermediate_size1 // self.maxpool.kernel_size
        # intermediate_size2 = (input_shape[1] - self.conv1.kernel_size[1] + 1) // self.conv1.stride[1]
        # intermediate_size2 = intermediate_size2 // self.maxpool.kernel_size
        # intermediate_size2 = (intermediate_size2 - self.conv2.kernel_size[1] + 1) // self.conv2.stride[1]
        # intermediate_size2 = intermediate_size2 // self.maxpool.kernel_size

        # Initialize linear layer
        self.final_layer      = nn.Linear(n_filters_2 * 23 * 23, self.n_classes, bias=True)  # ONLY TRUE FOR 224x224

    def init_weights(self, scale=1e-4):
        # Initialize weights of the model
        for param in self.conv1.parameters():
            nn.init.uniform_(param, a=-scale, b=scale)

        for param in self.conv2.parameters():
            nn.init.uniform_(param, a=-scale, b=scale)

        torch.nn.init.xavier_uniform_(self.final_layer.weight)

    def forward(self, images):
        # Squeeze the first dim...it is the batch_size = 1
        images  = images.squeeze(0)

        # images  = images.view(images.shape[0], -1)

        # Pass the input from the conv and max pool layers

        output  = self.relu(self.maxpool(self.conv1(images)))

        # TODO add it later
        # output  = F.dropout(output, self.dropout, self.training)

        output = self.relu(self.maxpool(self.conv2(output)))

        # TODO add it later
        # output = F.dropout(output, self.dropout, self.training)

        # Flatten the output
        output  = output.view(images.shape[0], -1)

        # Pass through Linear
        output  = self.final_layer(output)

        # Perform average voting
        output  = torch.mean(output)

        # do not add sigmoid...BCEWithLogitsLoss does this
        return output.unsqueeze(-1)


    # def forward_cam(self, images):
    #     # flatten the images
    #
    #     # squeeze the first dim...it is the batch_size = 1
    #     images = images.squeeze(0)
    #
    #     # images  = images.view(images.shape[0], -1)
    #
    #     output = self.relu(self.maxpool(self.conv1(images)))
    #
    #     # TODO add it later
    #     # output  = F.dropout(output, self.dropout, self.training)
    #
    #     output = self.relu(self.maxpool(self.conv2(output)))
    #
    #     # flatten the output
    #
    #     # TODO add it later
    #     # output = F.dropout(output, self.dropout, self.training)
    #
    #     output = output.view(images.shape[0], -1)
    #
    #     # MLP
    #     output = self.relu(self.hidden_layer(output))
    #
    #     return output


"""
CNN Model
    2 Conv Layers
    2 Max Pooling Layers
    Relu for activation function
    Linear for output
    Max Pooling over Views and then final layer to get one output 
"""
class CNN_With_Max_PoolingRELU(Module):
    def __init__(self, input_channels, input_shape, n_classes, n_filters_1, n_filters_2, dropout=None):
        super(CNN_With_Max_PoolingRELU, self).__init__()

        self.input_channels = input_channels
        self.input_shape   = input_shape
        self.n_classes        = n_classes
        self.n_filters_1      = n_filters_1
        self.n_filters_2      = n_filters_2

        self.relu       = F.relu
        self.dropout          = dropout

        # Initialize convolution layers
        self.conv1            = nn.Conv2d(self.input_channels, n_filters_1, kernel_size=5)
        self.conv2            = nn.Conv2d(n_filters_1, n_filters_2, kernel_size=5)

        # Initialize max pooling layers
        self.maxpool          = nn.MaxPool2d(kernel_size=3)

        # intermediate_size1 = (input_shape[0] - self.conv1.kernel_size[0] + 1) // self.conv1.stride[0]
        # intermediate_size1 = intermediate_size1 // self.maxpool.kernel_size
        # intermediate_size1 = (intermediate_size1 - self.conv2.kernel_size[0] + 1) // self.conv2.stride[0]
        # intermediate_size1 = intermediate_size1 // self.maxpool.kernel_size
        # intermediate_size2 = (input_shape[1] - self.conv1.kernel_size[1] + 1) // self.conv1.stride[1]
        # intermediate_size2 = intermediate_size2 // self.maxpool.kernel_size
        # intermediate_size2 = (intermediate_size2 - self.conv2.kernel_size[1] + 1) // self.conv2.stride[1]
        # intermediate_size2 = intermediate_size2 // self.maxpool.kernel_size

        # Initialize linear layer
        self.final_layer      = nn.Linear(n_filters_2 * 23 * 23, self.n_classes, bias=True)  # ONLY TRUE FOR 224x224

    def init_weights(self, scale=1e-4):
        # Initialize weights of the model
        for param in self.conv1.parameters():
            nn.init.uniform_(param, a=-scale, b=scale)

        for param in self.conv2.parameters():
            nn.init.uniform_(param, a=-scale, b=scale)

        torch.nn.init.xavier_uniform_(self.final_layer.weight)

    def forward(self, images):
        # Squeeze the first dim...it is the batch_size = 1
        images  = images.squeeze(0)

        # images  = images.view(images.shape[0], -1)

        # Pass the input from the conv and max pool layers

        output  = self.relu(self.maxpool(self.conv1(images)))

        # TODO add it later
        # output  = F.dropout(output, self.dropout, self.training)

        output = self.relu(self.maxpool(self.conv2(output)))

        # flatten the output

        # TODO add it later
        # output = F.dropout(output, self.dropout, self.training)

        # Flatten the output
        output  = output.view(images.shape[0], -1)

        # Perform max pooling over the Views
        output  = torch.max(output, dim=0)[0].view(-1)

        output  = self.final_layer(output)

        # do not add sigmoid...BCEWithLogitsLoss does this
        return output


"""
CNN Model
    2 Conv Layers
    2 Max Pooling Layers
    Leaky Relu for activation function
    MPL with 1 hidden for output
    Average Voting on output of each View
"""
class CNN_MLP_Average_VotingRELU(Module):
    def __init__(self, input_channels, input_shape, n_classes, n_filters_1, n_filters_2, hidden_size, dropout=None):
        super(CNN_MLP_Average_VotingRELU, self).__init__()

        self.input_channels = input_channels
        self.input_shape      = input_shape
        self.n_classes        = n_classes
        self.n_filters_1      = n_filters_1
        self.n_filters_2      = n_filters_2
        self.hidden_size      = hidden_size

        self.relu       = F.relu
        self.dropout          = dropout

        # Initialize convolution layers
        self.conv1            = nn.Conv2d(self.input_channels, n_filters_1, kernel_size=5)
        self.conv2            = nn.Conv2d(n_filters_1, n_filters_2, kernel_size=5)

        # Initialize max pooling layers
        self.maxpool          = nn.MaxPool2d(kernel_size=3)


        # Place holder for gradients for cam
        self.cam_gradients = None

        # intermediate_size1 = (input_shape[0] - self.conv1.kernel_size[0] + 1) // self.conv1.stride[0]
        # intermediate_size1 = intermediate_size1 // self.maxpool.kernel_size
        # intermediate_size1 = (intermediate_size1 - self.conv2.kernel_size[0] + 1) // self.conv2.stride[0]
        # intermediate_size1 = intermediate_size1 // self.maxpool.kernel_size
        # intermediate_size2 = (input_shape[1] - self.conv1.kernel_size[1] + 1) // self.conv1.stride[1]
        # intermediate_size2 = intermediate_size2 // self.maxpool.kernel_size
        # intermediate_size2 = (intermediate_size2 - self.conv2.kernel_size[1] + 1) // self.conv2.stride[1]
        # intermediate_size2 = intermediate_size2 // self.maxpool.kernel_size

        # Initialize MLP
        self.hidden_layer     = nn.Linear(n_filters_2*23*23, self.hidden_size, bias=True)  # ONLY TRUE FOR 224x224
        self.final_layer      = nn.Linear(self.hidden_size, self.n_classes, bias=True)

    def init_weights(self, scale=1e-4):
        # Initialize weights of the model
        for param in self.conv1.parameters():
            nn.init.uniform_(param, a=-scale, b=scale)

        for param in self.conv2.parameters():
            nn.init.uniform_(param, a=-scale, b=scale)

        torch.nn.init.xavier_uniform_(self.hidden_layer.weight)
        torch.nn.init.xavier_uniform_(self.final_layer.weight)

    def forward(self, images):
        # Squeeze the first dim...it is the batch_size = 1
        images  = images.squeeze(0)

        # images  = images.view(images.shape[0], -1)

        # Pass the input from the conv and max pool layers

        output  = self.relu(self.maxpool(self.conv1(images)))

        # TODO add it later
        # output  = F.dropout(output, self.dropout, self.training)

        output = self.relu(self.maxpool(self.conv2(output)))

        # TODO add it later
        # output = F.dropout(output, self.dropout, self.training)

        # Flatten the output

        output  = output.view(images.shape[0], -1)

        # Pass through MLP
        output = self.relu(self.hidden_layer(output))

        # output = self.relu(self.hidden_layer2(output))

        output  = self.final_layer(output)

        # Perform average voting
        output  = torch.mean(output)

        # do not add sigmoid...BCEWithLogitsLoss does this
        return output.unsqueeze(-1)


    def forward_cam(self, images):
        # Squeeze the first dim...it is the batch_size = 1
        images = images.squeeze(0)

        # images  = images.view(images.shape[0], -1)

        # Pass the input from the conv and max pool layers

        output = self.relu(self.maxpool(self.conv1(images)))

        # TODO add it later
        # output  = F.dropout(output, self.dropout, self.training)

        output = self.conv2(output)

        h = output.register_hook(self.activations_hook)

        output = self.relu(self.maxpool(output))

        # TODO add it later
        # output = F.dropout(output, self.dropout, self.training)

        # Flatten the output
        output = output.view(images.shape[0], -1)

        # Pass through MLP
        output = self.relu(self.hidden_layer(output))

        # output = self.relu(self.hidden_layer2(output))

        output = self.final_layer(output)

        # Perform average voting
        output = torch.mean(output)

        # do not add sigmoid...BCEWithLogitsLoss does this
        return output.unsqueeze(-1)

    def activations_hook(self, grad):
        self.cam_gradients = grad

    def get_activations_gradient(self):
        return self.cam_gradients

    def get_activations(self, x):
        # flatten the images

        # squeeze the first dim...it is the batch_size = 1
        x = x.squeeze(0)

        # images  = images.view(images.shape[0], -1)

        output = self.relu(self.maxpool(self.conv1(x)))

        # TODO add it later
        # output  = F.dropout(output, self.dropout, self.training)

        output = self.conv2(output)

        return output


"""
CNN Model
    2 Conv Layers
    2 Max Pooling Layers
    Relu for activation function
    MPL with 1 hidden for output
    Max Pooling over Views and then final layer to get one output 
"""
class CNN_MLP_Max_PoolingRELU(Module):
    def __init__(self, input_channels, input_shape, n_classes, n_filters_1, n_filters_2, hidden_size, dropout=None):
        super(CNN_MLP_Max_PoolingRELU, self).__init__()

        self.input_channels = input_channels
        self.input_shape   = input_shape
        self.n_classes        = n_classes
        self.n_filters_1      = n_filters_1
        self.n_filters_2      = n_filters_2
        self.hidden_size      = hidden_size

        self.relu       = F.relu
        self.dropout          = dropout

        # Initialize convolution layers
        self.conv1            = nn.Conv2d(self.input_channels, n_filters_1, kernel_size=5)
        self.conv2            = nn.Conv2d(n_filters_1, n_filters_2, kernel_size=5)

        # Initialize max pooling layers
        self.maxpool          = nn.MaxPool2d(kernel_size=3)

        # Place holder for gradients for cam
        self.cam_gradients = None

        # intermediate_size1 = (input_shape[0] - self.conv1.kernel_size[0] + 1) // self.conv1.stride[0]
        # intermediate_size1 = intermediate_size1 // self.maxpool.kernel_size
        # intermediate_size1 = (intermediate_size1 - self.conv2.kernel_size[0] + 1) // self.conv2.stride[0]
        # intermediate_size1 = intermediate_size1 // self.maxpool.kernel_size
        # intermediate_size2 = (input_shape[1] - self.conv1.kernel_size[1] + 1) // self.conv1.stride[1]
        # intermediate_size2 = intermediate_size2 // self.maxpool.kernel_size
        # intermediate_size2 = (intermediate_size2 - self.conv2.kernel_size[1] + 1) // self.conv2.stride[1]
        # intermediate_size2 = intermediate_size2 // self.maxpool.kernel_size

        # Initialize MLP
        self.hidden_layer     = nn.Linear(n_filters_2*23*23, self.hidden_size, bias=True)  # ONLY TRUE FOR 224x224
        self.final_layer      = nn.Linear(self.hidden_size, self.n_classes, bias=True)

    def init_weights(self, scale=1e-4):
        # Initialize weights of the model
        for param in self.conv1.parameters():
            nn.init.uniform_(param, a=-scale, b=scale)

        for param in self.conv2.parameters():
            nn.init.uniform_(param, a=-scale, b=scale)

        torch.nn.init.xavier_uniform_(self.hidden_layer.weight)
        torch.nn.init.xavier_uniform_(self.final_layer.weight)

    def forward(self, images):
        # Squeeze the first dim...it is the batch_size = 1
        images  = images.squeeze(0)

        # images  = images.view(images.shape[0], -1)

        # Pass the input from the conv and max pool layers

        output  = self.relu(self.maxpool(self.conv1(images)))

        # TODO add it later
        # output  = F.dropout(output, self.dropout, self.training)

        output = self.relu(self.maxpool(self.conv2(output)))

        # TODO add it later
        # output = F.dropout(output, self.dropout, self.training)

        # Flatten the output
        output  = output.view(images.shape[0], -1)

        # Pass through MLP
        output = self.relu(self.hidden_layer(output))

        # output = self.relu(self.hidden_layer2(output))

        # Perform max pooling over the Views
        output = torch.max(output, dim=0)[0].view(-1)

        output  = self.final_layer(output)

        # do not add sigmoid...BCEWithLogitsLoss does this
        return output

    def forward_cam(self, images):
        # Squeeze the first dim...it is the batch_size = 1
        images = images.squeeze(0)

        # images  = images.view(images.shape[0], -1)

        # Pass the input from the conv and max pool layers

        output = self.relu(self.maxpool(self.conv1(images)))

        # TODO add it later
        # output  = F.dropout(output, self.dropout, self.training)

        output = self.conv2(output)

        # Register hook for gradients
        h = output.register_hook(self.activations_hook)

        output = self.relu(self.maxpool(output))

        # TODO add it later
        # output = F.dropout(output, self.dropout, self.training)

        # Flatten the output
        output = output.view(images.shape[0], -1)

        # Pass through MLP
        output = self.relu(self.hidden_layer(output))

        # output = self.relu(self.hidden_layer2(output))

        # Perform max pooling over the Views
        output = torch.max(output, dim=0)[0].view(-1)

        output = self.final_layer(output)

        # do not add sigmoid...BCEWithLogitsLoss does this
        return output

    def activations_hook(self, grad):
        self.cam_gradients = grad

    def get_activations_gradient(self):
        return self.cam_gradients

    def get_activations(self, x):
        # flatten the images

        # squeeze the first dim...it is the batch_size = 1
        x = x.squeeze(0)

        # images  = images.view(images.shape[0], -1)

        output = self.relu(self.maxpool(self.conv1(x)))

        # TODO add it later
        # output  = F.dropout(output, self.dropout, self.training)

        output = self.conv2(output)

        return output


# =========================== FROM TORCHVISION =========================== #
# https://github.com/pytorch/vision/blob/master/torchvision/models/densenet.py

"""
Pretrained DenseNet Model
    Leaky Relu for activation function
    MPL with 1 hidden for output
    Average Voting on output of each View  
"""
class PretrainedDensenetAverageVoting(nn.Module):
    def __init__(self, hidden_size, num_class=1, frozen=False):
        super().__init__()

        self.channels = 81536
        self.hidden_size = hidden_size

        # Get pretrained densenet
        densenet_169 = models.densenet169(pretrained=True)

        # Here we get the part of the model where the feature extraction is happening
        # in that way we can add on top of the feature extractor our own classifier as in the mura paper
        # if we used densenet169.classifier we would get the final linear layer used for classification
        self.features = nn.Sequential(*list(densenet_169.features.children()))

        # Place holder for gradients for cam
        self.cam_gradients = None

        # Freeze certain blocks
        if not frozen:
            for mod in self.features[:9]:
                mod.requires_grad_(False)
        else:
            for mod in self.features:
                mod.requires_grad_(False)

        self.leaky_relu = F.leaky_relu

        # Initialize MLP
        self.hidden_layer = nn.Linear(self.channels, self.hidden_size, bias=True)
        self.final_layer = nn.Linear(self.hidden_size, num_class, bias=True)

        # self.sigmoid = nn.Sigmoid()

    def init_weights(self):
        # Initialize weights of the model
        torch.nn.init.xavier_uniform_(self.hidden_layer.weight)
        torch.nn.init.xavier_uniform_(self.final_layer.weight)

    def forward(self, x):

        # we must squeeze the first dimension ---> it is the batch_size
        x        = x.squeeze(0)

        # TODO for later...upsample the image to 224x224

        # Pass input though some layers of densenet to get features
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

        # Flatten output
        out = out.view(out.shape[0], -1)

        # Pass though MLP
        out = self.leaky_relu(self.hidden_layer(out))

        out = self.final_layer(out)

        # Perform average voting
        out = torch.mean(out)

        # do not add sigmoid...BCEWithLogitsLoss does this
        return out.unsqueeze(-1)

    def forward_cam(self, x):
        # we must squeeze the first dimension ---> it is the batch_size
        x = x.squeeze(0)

        # TODO for later...upsample the image to 224x224

        # Pass input though some layers of densenet to get features
        features = self.features(x)

        # Register hook for gradients
        h = features.register_hook(self.activations_hook)

        out = self.leaky_relu(features)

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

        # Flatten the output
        out = out.view(out.shape[0], -1)

        # Pass through MLP
        out = self.leaky_relu(self.hidden_layer(out))

        out = self.final_layer(out)

        # Perform average voting
        out = torch.mean(out)

        # do not add sigmoid...BCEWithLogitsLoss does this
        return out.unsqueeze(-1)

    def activations_hook(self, grad):
        self.cam_gradients = grad

    def get_activations_gradient(self):
        return self.cam_gradients

    def get_activations(self, x):
        # we must squeeze the first dimension ---> it is the batch_size
        x = x.squeeze(0)
        return self.features(x)


"""
Pretrained DenseNet Model
    Relu for activation function
    MPL with 1 hidden for output
    Average Voting on output of each View  
"""
class PretrainedDensenetAverageVotingRELU(nn.Module):
    def __init__(self, hidden_size, num_class=1, frozen=False):
        super().__init__()

        self.channels = 81536
        self.hidden_size = hidden_size

        # Get pretrained densenet
        densenet_169 = models.densenet169(pretrained=True)

        # Here we get the part of the model where the feature extraction is happening
        # in that way we can add on top of the feature extractor our own classifier as in the mura paper
        # if we used densenet169.classifier we would get the final linear layer used for classification
        self.features = nn.Sequential(*list(densenet_169.features.children()))

        # Place holder for gradients for cam
        self.cam_gradients = None

        # Freeze certain blocks
        if not frozen:
            for mod in self.features[:9]:
                mod.requires_grad_(False)
        else:
            for mod in self.features:
                mod.requires_grad_(False)

        self.relu = F.relu

        # Initialize MLP
        self.hidden_layer = nn.Linear(self.channels, self.hidden_size, bias=True)
        self.final_layer = nn.Linear(self.hidden_size, num_class, bias=True)

        # self.sigmoid = nn.Sigmoid()

    def init_weights(self):
        # Initialize weights of the model
        torch.nn.init.xavier_uniform_(self.hidden_layer.weight)
        torch.nn.init.xavier_uniform_(self.final_layer.weight)

    def forward(self, x):

        # we must squeeze the first dimension ---> it is the batch_size
        x        = x.squeeze(0)

        # TODO for later...upsample the image to 224x224

        # Pass input though some layers of densenet to get features
        features = self.features(x)
        out      = self.relu(features)

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

        # Flatten the output
        out = out.view(out.shape[0], -1)

        # Pass through MLP
        out = self.relu(self.hidden_layer(out))

        out = self.final_layer(out)

        # Perform average voting
        out = torch.mean(out)

        # do not add sigmoid...BCEWithLogitsLoss does this
        return out.unsqueeze(-1)

    def forward_cam(self, x):
        # we must squeeze the first dimension ---> it is the batch_size
        x = x.squeeze(0)

        # TODO for later...upsample the image to 224x224

        # Pass input though some layers of densenet to get features
        features = self.features(x)

        # Register hook for gradients
        h = features.register_hook(self.activations_hook)

        out = self.relu(features)

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

        # Flatten the output
        out = out.view(out.shape[0], -1)

        # Pass through MLP
        out = self.relu(self.hidden_layer(out))

        out = self.final_layer(out)

        # Perform average voting
        out = torch.mean(out)

        # do not add sigmoid...BCEWithLogitsLoss does this
        return out.unsqueeze(-1)

    def activations_hook(self, grad):
        self.cam_gradients = grad

    def get_activations_gradient(self):
        return self.cam_gradients

    def get_activations(self, x):
        # we must squeeze the first dimension ---> it is the batch_size
        x = x.squeeze(0)
        return self.features(x)


"""
Pretrained DenseNet Model
    Leaky Relu for activation function
    MPL with 1 hidden for output
    Max Pooling over Views and then final layer to get one output  
"""
class PretrainedDensenet(nn.Module):
    def __init__(self, hidden_size, num_class=1, frozen=False):
        super().__init__()

        self.channels = 81536
        self.hidden_size = hidden_size

        # Get pretrained densenet
        densenet_169 = models.densenet169(pretrained=True)

        # HERE WE GET THE PART OF THE MODEL WHERE THE FEATURE EXTRACTION IS HAPPENING
        # IN THAT WAY WE CAN ADD ON TOP OF THE FEATURE EXTRACTOR OUR OWN CLASSIFIER AS IN THE MURA PAPER
        # IF WE USED DENSENET169.CLASSIFIER WE WOULD GET THE FINAL LINEAR LAYER USED FOR CLASSIFICATION
        self.features = nn.Sequential(*list(densenet_169.features.children()))

        # Place holder for gradients for cam
        self.cam_gradients = None

        # Freeze certain blocks
        if not frozen:
            for mod in self.features[:9]:
                mod.requires_grad_(False)
        else:
            for mod in self.features:
                mod.requires_grad_(False)

        self.leaky_relu = F.leaky_relu

        # Initialize MLP
        self.hidden_layer = nn.Linear(self.channels, self.hidden_size, bias=True)
        self.final_layer = nn.Linear(self.hidden_size, num_class, bias=True)

        # self.sigmoid = nn.Sigmoid()

    def init_weights(self):
        # Initialize weights of the model
        torch.nn.init.xavier_uniform_(self.hidden_layer.weight)
        torch.nn.init.xavier_uniform_(self.final_layer.weight)

    def forward(self, x):

        # we must squeeze the first dimension ---> it is the batch_size
        x        = x.squeeze(0)

        # TODO for later...upsample the image to 224x224

        # Pass input though some layers of densenet to get features
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

        # Flatten the output
        out = out.view(out.shape[0], -1)

        # Pass through MLP
        out = self.leaky_relu(self.hidden_layer(out))

        # output = self.leaky_relu(self.hidden_layer2(output))

        # Perform max pooling over the Views
        out = torch.max(out, dim=0)[0].view(-1)

        out = self.final_layer(out)

        # return torch.mean(torch.sigmoid(out)).unsqueeze(0)

        return out


    def forward_cam(self, x):
        # we must squeeze the first dimension ---> it is the batch_size
        x = x.squeeze(0)

        # TODO for later...upsample the image to 224x224

        # Pass input though some layers of densenet to get features
        features = self.features(x)

        # Register hook for gradients
        h = features.register_hook(self.activations_hook)

        out = self.leaky_relu(features)

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

        # Flatten the output
        out = out.view(out.shape[0], -1)

        # Pass through MLP
        out = self.leaky_relu(self.hidden_layer(out))

        # output = self.leaky_relu(self.hidden_layer2(output))

        # Perform max pooling over the Views
        out = torch.max(out, dim=0)[0].view(-1)

        out = self.final_layer(out)

        # return torch.mean(torch.sigmoid(out)).unsqueeze(0)

        return out

    def activations_hook(self, grad):
        self.cam_gradients = grad

    def get_activations_gradient(self):
        return self.cam_gradients

    def get_activations(self, x):
        # we must squeeze the first dimension ---> it is the batch_size
        x = x.squeeze(0)
        return self.features(x)


"""
Pretrained DenseNet Model
    Leaky Relu for activation function
    MPL with 1 hidden for output
    Max Pooling over Views and then final layer to get one output  
"""
class PretrainedDensenetRELU(nn.Module):
    def __init__(self, hidden_size, num_class=1, frozen=False):
        super().__init__()

        self.channels = 81536
        self.hidden_size = hidden_size

        # Get pretrained densenet
        densenet_169 = models.densenet169(pretrained=True)

        # Here we get the part of the model where the feature extraction is happening
        # in that way we can add on top of the feature extractor our own classifier as in the mura paper
        # if we used densenet169.classifier we would get the final linear layer used for classification
        self.features = nn.Sequential(*list(densenet_169.features.children()))

        # Place holder for gradients for cam
        self.cam_gradients = None

        # Freeze certain blocks
        if not frozen:
            for mod in self.features[:9]:
                mod.requires_grad_(False)
        else:
            for mod in self.features:
                mod.requires_grad_(False)

        self.relu = F.relu

        # Initialize MLP
        self.hidden_layer = nn.Linear(self.channels, self.hidden_size, bias=True)
        self.final_layer = nn.Linear(self.hidden_size, num_class, bias=True)

        # self.sigmoid = nn.Sigmoid()

    def init_weights(self):
        # Initialize weights of the model
        torch.nn.init.xavier_uniform_(self.hidden_layer.weight)
        torch.nn.init.xavier_uniform_(self.final_layer.weight)

    def forward(self, x):

        # we must squeeze the first dimension ---> it is the batch_size
        x        = x.squeeze(0)

        # TODO for later...upsample the image to 224x224

        # Pass input though some layers of densenet to get features
        features = self.features(x)
        out      = self.relu(features)

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

        # Flatten the output
        out = out.view(out.shape[0], -1)

        # Pass through MLP
        out = self.relu(self.hidden_layer(out))

        # output = self.leaky_relu(self.hidden_layer2(output))

        # Perform max pooling over the Views
        out = torch.max(out, dim=0)[0].view(-1)

        out = self.final_layer(out)

        # return torch.mean(torch.sigmoid(out)).unsqueeze(0)

        return out


    def forward_cam(self, x):
        # we must squeeze the first dimension ---> it is the batch_size
        x = x.squeeze(0)

        # TODO for later...upsample the image to 224x224

        # Pass input though some layers of densenet to get features
        features = self.features(x)

        # Register hook for gradients
        h = features.register_hook(self.activations_hook)

        out = self.relu(features)

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

        # Flatten the output
        out = out.view(out.shape[0], -1)

        # Pass through MLP
        out = self.relu(self.hidden_layer(out))

        # output = self.leaky_relu(self.hidden_layer2(output))

        # Perform max pooling over the Views
        out = torch.max(out, dim=0)[0].view(-1)

        out = self.final_layer(out)

        # return torch.mean(torch.sigmoid(out)).unsqueeze(0)

        return out

    def activations_hook(self, grad):
        self.cam_gradients = grad

    def get_activations_gradient(self):
        return self.cam_gradients

    def get_activations(self, x):
        # we must squeeze the first dimension ---> it is the batch_size
        x = x.squeeze(0)
        return self.features(x)


class PretrainedResnet(nn.Module):
    def __init__(self, hidden_size, num_class=1):
        super().__init__()

        self.channels = 2048
        self.hidden_size = hidden_size
        resnet101 = models.resnet101(pretrained=True)

        # HERE WE GET ALL THE LAYERS EXCEPT FROM THE FC
        self.features = nn.Sequential(*list(resnet101.children())[:-1])

        # freeze certain blocks
        for mod in self.features[:7]:
            mod.requires_grad_(False)

        self.leaky_relu = F.leaky_relu

        self.hidden_layer = nn.Linear(self.channels, self.hidden_size, bias=True)
        self.final_layer = nn.Linear(self.hidden_size, num_class, bias=True)

    def init_weights(self):
        torch.nn.init.xavier_uniform_(self.hidden_layer.weight)
        torch.nn.init.xavier_uniform_(self.final_layer.weight)

    def forward(self, x):
        # we must squeeze the first dimension ---> it is the batch_size
        x = x.squeeze(0)

        features = self.features(x)

        out = self.leaky_relu(features)

        out = out.view(out.shape[0], -1)

        out = self.leaky_relu(self.hidden_layer(out))

        # Perform max pooling amongst the views
        out = torch.max(out, dim=0)[0].view(-1)

        out = self.final_layer(out)

        return out
