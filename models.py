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

use_cuda = torch.cuda.is_available()
torch.manual_seed(seed)
if use_cuda:
    torch.cuda.manual_seed(seed)


from tqdm import tqdm

import numpy as np
np.random.seed(seed)


#  ==================================== MODELS ==================================== #


# FOR THE MLP MAYBE WE JUST NEED TO FLATTEN THE VIEWS OF A STUDY AND OUTPUT 1 NUMBER
# THE PROBA OF BEING NORMAL OR ABNORMAL
# OR ELSE WE NEED FOR EACH IMAGE TO OUTPUT A NUMBER AND TAKE THE AVERAGE
# THAT'S WHY THE AVERAGE POOLING
class MLP_With_Average_Pooling(nn.Module):
    def __init__(self, input_dim, n_classes, hidden_1, hidden_2,):
        super(MLP_With_Average_Pooling, self).__init__()

        self.input_dim   = input_dim
        self.n_classes   = n_classes
        self.hidden_1    = hidden_1
        self.hidden_2    = hidden_2
        self.leaky_relu  = F.leaky_relu

        # affine = True means it has learnable parameters
        self.batchnorm_1    = nn.BatchNorm1d(self.hidden_1, affine=True)
        self.batchnorm_relu = F.relu

        # self.batchnorm_2 = nn.BatchNorm1d(self.hidden_2, affine=True)

        self.layer_1     = nn.Linear(self.input_dim, self.hidden_1, bias=True)
        self.layer_2     = nn.Linear(self.hidden_1, self.hidden_2, bias=True)
        self.final_layer = nn.Linear(self.hidden_2, self.n_classes, bias=True)

    def init_weights(self):
        torch.nn.init.xavier_uniform_(self.layer_1.weight)
        torch.nn.init.xavier_uniform_(self.layer_2.weight)
        torch.nn.init.xavier_uniform_(self.final_layer.weight)

    def forward(self):
        pass

