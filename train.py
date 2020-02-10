"""

In this file we run ours models one by one

"""

# Imports

import random
from random import shuffle
import numpy as np
import os
import scipy.sparse as sp
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import pickle
from torch.utils.data import DataLoader
from models import MLP_With_Average_Voting, PretrainedDensenet, PretrainedResnet, CNN_With_Average_Voting, \
    MLP_With_Max_Pooling, CNN_MLP_Average_Voting, CNN_MLP_Max_Pooling, PretrainedDensenetAverageVoting, \
    PretrainedDensenetRELU, PretrainedDensenetAverageVotingRELU, CNN_With_Average_VotingRELU, \
    CNN_MLP_Average_VotingRELU, CNN_MLP_Max_PoolingRELU, CNN_With_Max_Pooling, CNN_With_Max_PoolingRELU
from sklearn.metrics import roc_curve, auc, roc_auc_score, average_precision_score
import re
import argparse
import logging
import pandas as pd
import json

from dataloader import get_study_level_data, get_dataloaders

# Seed for our experiments
seed = 1997
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# Setting cuda for GPU if it is available
use_cuda = torch.cuda.is_available()
if use_cuda:
    torch.cuda.manual_seed(seed)

# Base directory for checkpoints
odir_checkpoint = '/mnt/data/sotiris/checkpoints/'
# odir_checkpoint = 'drive/My Drive/MURA Project/checkpoints/'

# Initialize the logger handle to None
hdlr = None

# Initialize names of the body parts for the MURA dataset
study_wrist = 'XR_WRIST'
study_elbow = 'XR_ELBOW'
study_finger = 'XR_FINGER'
study_forearm = 'XR_FOREARM'
study_hand = 'XR_HAND'
study_humerus = 'XR_HUMERUS'
study_shoulder = 'XR_SHOULDER'


# Set checkpoints for each model

# THIS IS FOR DENSENET PRETRAINED WITH MLP WITH 1 LAYER AND AVERAGE VOTING -- OUR LOSS
# best_checkpoint_name = 'densenet_mlp_averagevoting.pth.tar'
# progress_checkpoint = 'densenet_mlp_averagevoting_progress.pth.tar'

# THIS IS FOR DENSENET PRETRAINED WITH MLP WITH 1 LAYER AND AVERAGE VOTING WITH RELU -- OUR LOSS
# best_checkpoint_name = 'densenet_mlp_averagevoting_relu.pth.tar'
# progress_checkpoint = 'densenet_mlp_averagevoting_relu_progress.pth.tar'

# THIS IS FOR DENSENET PRETRAINED WITH MLP WITH 1 LAYER AND MAX POOLING OVER THE VIEWS -- OUR LOSS
# best_checkpoint_name = 'densenet_mlp_maxpooling.pth.tar'
# progress_checkpoint = 'densenet_mlp_maxpooling_progress.pth.tar'

# THIS IS FOR DENSENET PRETRAINED WITH MLP WITH 1 LAYER AND MAX POOLING OVER THE VIEWS WITH RELU -- OUR LOSS
# best_checkpoint_name = 'densenet_mlp_maxpooling_relu.pth.tar'
# progress_checkpoint = 'densenet_mlp_maxpooling_relu_progress.pth.tar'

# THIS IS FOR FROZEN DENSENET PRETRAINED WITH MLP WITH 1 LAYER AND AVERAGE VOTING -- OUR LOSS
# best_checkpoint_name = 'frozen_densenet_mlp_averagevoting.pth.tar'
# progress_checkpoint = 'frozen_densenet_mlp_averagevoting_progress.pth.tar'

# THIS IS FOR FROZEN DENSENET PRETRAINED WITH MLP WITH 1 LAYER AND AVERAGE VOTING WITH RELU -- OUR LOSS
# best_checkpoint_name = 'frozen_densenet_mlp_averagevoting_relu.pth.tar'
# progress_checkpoint = 'frozen_densenet_mlp_averagevoting_relu_progress.pth.tar'

# THIS IS FOR FROZEN DENSENET PRETRAINED WITH MLP WITH 1 LAYER AND MAX POOLING OVER THE VIEWS -- OUR LOSS
# best_checkpoint_name = 'frozen_densenet_mlp_maxpooling.pth.tar'
# progress_checkpoint = 'frozen_densenet_mlp_maxpooling_progress.pth.tar'

# THIS IS FOR FROZEN DENSENET PRETRAINED WITH MLP WITH 1 LAYER AND MAX POOLING OVER THE VIEWS WITH RELU -- OUR LOSS
# best_checkpoint_name = 'frozen_densenet_mlp_maxpooling_relu.pth.tar'
# progress_checkpoint = 'frozen_densenet_mlp_maxpooling_relu_progress.pth.tar'

# THIS IS FOR RESNET PRETRAINED WITH MLP WITH 1 LAYER AND MAX POOLING OVER THE VIEWS -- OUR LOSS
# best_checkpoint_name = 'resnet_mlp_maxpooling.pth.tar'
# progress_checkpoint = 'resnet_mlp_maxpooling_progress.pth.tar'

# THIS IS FOR CNN 2 LAYERS + AVERAGE VOTING -- OUR LOSS
# best_checkpoint_name = 'cnn_2layers_averagevoting.pth.tar'
# progress_checkpoint = 'cnn_2layers_averagevoting_progress.pth.tar'

# THIS IS FOR CNN 2 LAYERS + MAX POOLING -- OUR LOSS
# best_checkpoint_name = 'cnn_2layers_maxpooling.pth.tar'
# progress_checkpoint = 'cnn_2layers_maxpooling.pth.tar'

# THIS IS FOR CNN 2 LAYERS + MLP + AVERAGE VOTING -- OUR LOSS
# best_checkpoint_name = 'cnn_2layers_mlp_averagevoting.pth.tar'
# progress_checkpoint = 'cnn_2layers_mlp_averagevoting_progress.pth.tar'

# THIS IS FOR CNN 2 LAYERS + MLP + MAX POOLING OVER VIEWS -- OUR LOSS
# best_checkpoint_name = 'cnn_2layers_mpl_maxpooling.pth.tar'
# progress_checkpoint = 'cnn_2layers_mpl_maxpooling_progress.pth.tar'

# THIS IS FOR CNN 2 LAYERS + AVERAGE VOTING WITH RELU -- OUR LOSS
# best_checkpoint_name = 'cnn_2layers_averagevoting_relu.pth.tar'
# progress_checkpoint = 'cnn_2layers_averagevoting_relu_progress.pth.tar'

# THIS IS FOR CNN 2 LAYERS + MAX POOLING OVER VIEWS WITH RELU -- OUR LOSS
# best_checkpoint_name = 'cnn_2layers_maxpooling_relu.pth.tar'
# progress_checkpoint = 'cnn_2layers_maxpooling_relu_progress.pth.tar'

# THIS IS FOR CNN 2 LAYERS + MLP + AVERAGE VOTING WITH RELU-- OUR LOSS
# best_checkpoint_name = 'cnn_2layers_mlp_averagevoting_relu.pth.tar'
# progress_checkpoint = 'cnn_2layers_mlp_averagevoting_relu_progress.pth.tar'

# THIS IS FOR CNN 2 LAYERS + MLP + MAX POOLING OVER VIEWS WITH RELU-- OUR LOSS
# best_checkpoint_name = 'cnn_2layers_mpl_maxpooling_relu.pth.tar'
# progress_checkpoint = 'cnn_2layers_mpl_maxpooling_relu_progress.pth.tar'

# THIS IS FOR MLP + AVERAGE POOLING -- OUR LOSS
# best_checkpoint_name = 'mlp_averagevoting.pth.tar'
# progress_checkpoint = 'mlp_averagevoting_progress.pth.tar'
# best_checkpoint_name = 'mlp_averagevoting_nodropout.pth.tar'
# progress_checkpoint = 'mlp_averagevoting_nodropout_progress.pth.tar'

# THIS IS FOR MLP + MAX POOLING -- OUR LOSS
# best_checkpoint_name = 'mlp_maxpooling.pth.tar'
# progress_checkpoint = 'mlp_maxpooling_progress.pth.tar'
# best_checkpoint_name = 'mlp_maxpooling_nodropout.pth.tar'
# progress_checkpoint = 'mlp_maxpooling_nodropout_progress.pth.tar'

# FOR TESTING
# best_checkpoint_name = 'testing.pth.tar'
# progress_checkpoint = 'testing_progress.pth.tar'

# FOR BEST MODEL
best_checkpoint_name = 'densenet_maxpooling_relu/hyperopt_trial_0.pth.tar'
progress_checkpoint = None

# Create the checkpoints directory
if not os.path.exists(odir_checkpoint):
    os.makedirs(odir_checkpoint)


def print_params(model):
    '''
    It just prints the number of parameters in the model.
    :param model:   The pytorch model
    :return:        Nothing.
    '''
    print(40 * '=')
    print(model)
    print(40 * '=')
    logger.info(40 * '=')
    logger.info(model)
    logger.info(40 * '=')
    trainable = 0
    untrainable = 0
    for parameter in model.parameters():
        # print(parameter.size())
        v = 1
        for s in parameter.size():
            v *= s
        if parameter.requires_grad:
            trainable += v
        else:
            untrainable += v
    total_params = trainable + untrainable
    print(40 * '=')
    print('trainable:{} untrainable:{} total:{}'.format(trainable, untrainable, total_params))
    print(40 * '=')
    logger.info(40 * '=')
    logger.info('trainable:{} untrainable:{} total:{}'.format(trainable, untrainable, total_params))
    logger.info(40 * '=')
    logger.info('')
    logger.info('')


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    """
    Save the torch checkpoint
    :param state: The state/checkpoint to save
    :param filename: The path and filename
    :return: Nothing
    """
    torch.save(state, filename)


def init_the_logger(hdlr):
    """
    Initializes the logger
    :param hdlr: The handler for the logger
    :return: The logger and its handler
    """

    # Create the checkpoints folder
    if not os.path.exists(odir_checkpoint):
        os.makedirs(odir_checkpoint)

    # Set the logger base directory
    od = odir_checkpoint.split('/')[-1]
    logger = logging.getLogger(od)

    # Remove the previous handler
    if (hdlr is not None):
        logger.removeHandler(hdlr)

    # Create the handler for the logger for each experiment

    # THIS IS FOR DENSENET PRETRAINED WITH MLP WITH 1 LAYER AND AVERAGE VOTING -- OUR LOSS
    # hdlr = logging.FileHandler(os.path.join(odir_checkpoint, 'densenet_mlp_averagevoting.log'))

    # THIS IS FOR DENSENET PRETRAINED WITH MLP WITH 1 LAYER AND AVERAGE VOTING WITH RELU -- OUR LOSS
    # hdlr = logging.FileHandler(os.path.join(odir_checkpoint, 'densenet_mlp_averagevoting_relu.log'))

    # THIS IS FOR DENSENET PRETRAINED WITH MLP WITH 1 LAYER AND MAX POOLING OVER THE VIEWS -- OUR LOSS
    # hdlr = logging.FileHandler(os.path.join(odir_checkpoint, 'densenet_mlp_maxpooling.log'))

    # THIS IS FOR DENSENET PRETRAINED WITH MLP WITH 1 LAYER AND MAX POOLING OVER THE VIEWS WITH RELU -- OUR LOSS
    # hdlr = logging.FileHandler(os.path.join(odir_checkpoint, 'densenet_mlp_maxpooling_relu.log'))

    # THIS IS FOR FROZEN DENSENET PRETRAINED WITH MLP WITH 1 LAYER AND AVERAGE VOTING -- OUR LOSS
    # hdlr = logging.FileHandler(os.path.join(odir_checkpoint, 'frozen_densenet_mlp_averagevoting.log'))

    # THIS IS FOR FROZEN DENSENET PRETRAINED WITH MLP WITH 1 LAYER AND AVERAGE VOTING WITH RELU -- OUR LOSS
    # hdlr = logging.FileHandler(os.path.join(odir_checkpoint, 'frozen_densenet_mlp_averagevoting_relu.log'))

    # THIS IS FOR FROZEN DENSENET PRETRAINED WITH MLP WITH 1 LAYER AND MAX POOLING OVER THE VIEWS -- OUR LOSS
    # hdlr = logging.FileHandler(os.path.join(odir_checkpoint, 'frozen_densenet_mlp_maxpooling.log'))

    # THIS IS FOR FROZEN DENSENET PRETRAINED WITH MLP WITH 1 LAYER AND MAX POOLING OVER THE VIEWS WITH RELU -- OUR LOSS
    # hdlr = logging.FileHandler(os.path.join(odir_checkpoint, 'frozen_densenet_mlp_maxpooling_relu.log'))

    # THIS IS FOR RESNET PRETRAINED WITH MLP WITH 1 LAYER AND MAX POOLING OVER THE VIEWS -- OUR LOSS
    # hdlr = logging.FileHandler(os.path.join(odir_checkpoint, 'resnet_mlp_maxpooling.log'))

    # THIS IS FOR CNN 2 LAYERS + AVERAGE VOTING -- OUR LOSS
    # hdlr = logging.FileHandler(os.path.join(odir_checkpoint, 'cnn_2layers_averagevoting.log'))

    # THIS IS FOR CNN 2 LAYERS + MAX POOLING OVER VIEWS -- OUR LOSS
    # hdlr = logging.FileHandler(os.path.join(odir_checkpoint, 'cnn_2layers_maxpooling.log'))

    # THIS IS FOR CNN 2 LAYERS + MLP + AVERAGE VOTING -- OUR LOSS
    # hdlr = logging.FileHandler(os.path.join(odir_checkpoint, 'cnn_2layers_mlp_averagevoting.log'))

    # THIS IS FOR CNN 2 LAYERS + MLP + MAX POOLING OVER VIEWS -- OUR LOSS
    # hdlr = logging.FileHandler(os.path.join(odir_checkpoint, 'cnn_2layers_mpl_maxpooling.log'))

    # THIS IS FOR CNN 2 LAYERS + AVERAGE VOTING WITH RELU -- OUR LOSS
    # hdlr = logging.FileHandler(os.path.join(odir_checkpoint, 'cnn_2layers_averagevoting_relu.log'))

    # THIS IS FOR CNN 2 LAYERS + MAX POOLING OVER VIEWS WITH RELU -- OUR LOSS
    # hdlr = logging.FileHandler(os.path.join(odir_checkpoint, 'cnn_2layers_maxpooling_relu.log'))

    # THIS IS FOR CNN 2 LAYERS + MLP + AVERAGE VOTING WITH RELU -- OUR LOSS
    # hdlr = logging.FileHandler(os.path.join(odir_checkpoint, 'cnn_2layers_mlp_averagevoting_relu.log'))

    # THIS IS FOR CNN 2 LAYERS + MLP + MAX POOLING OVER VIEWS WITH RELU -- OUR LOSS
    # hdlr = logging.FileHandler(os.path.join(odir_checkpoint, 'cnn_2layers_mpl_maxpooling_relu.log'))

    # THIS IS FOR MLP + AVERAGE VOTING -- OUR LOSS
    # hdlr = logging.FileHandler(os.path.join(odir_checkpoint, 'mlp_averagevoting.log'))
    # hdlr = logging.FileHandler(os.path.join(odir_checkpoint, 'mlp_averagevoting_nodropout.log'))

    # THIS IS FOR MLP + MAX POOLING -- OUR LOSS
    # hdlr = logging.FileHandler(os.path.join(odir_checkpoint, 'mlp_maxpooling.log'))
    # hdlr = logging.FileHandler(os.path.join(odir_checkpoint, 'mlp_maxpooling_nodropout.log'))

    # FOR TESTING
    hdlr = logging.FileHandler(os.path.join(odir_checkpoint, 'testing.log'))

    # Set the format for the logger
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)

    return logger, hdlr


# Initialize the logger
logger, hdlr = init_the_logger(hdlr)


def back_prop(batch_costs):
    """
    Perform back propagation for a batch
    :param batch_costs: The costs for the batch
    :return: The average cost of the batch
    """
    batch_cost = sum(batch_costs) / float(len(batch_costs))
    batch_cost.backward()
    optimizer.step()
    optimizer.zero_grad()
    batch_aver_cost = batch_cost.cpu().item()
    return batch_aver_cost


# HERE YOU PASS POSITIVE AND NEGATIVE WEIGHTS
# IT IS THE LOSS FROM THE PAPER
# def weighted_binary_cross_entropy(output, target, weights=None):
#     if weights is not None:
#         assert len(weights) == 2
#         loss = weights[1] * (target * torch.log(output)) + weights[0] * ((1 - target) * torch.log(1 - output))
#     else:
#         loss = target * torch.log(output) + (1 - target) * torch.log(1 - output)
#     return torch.neg(torch.mean(loss))

print()
print('Loading Data...')
print()

print('Loading ELBOW')
study_data_elbow = get_study_level_data(study_elbow)
print('Loading FINGER')
study_data_finger = get_study_level_data(study_finger)
print('Loading FOREARM')
study_data_forearm = get_study_level_data(study_forearm)
print('Loading HAND')
study_data_hand = get_study_level_data(study_hand)
print('Loading WRIST')
study_data_wrist = get_study_level_data(study_wrist)
print('Loading SHOULDER')
study_data_shoulder = get_study_level_data(study_shoulder)
print('Loading HUMERUS')
study_data_humerus = get_study_level_data(study_humerus)
print()
print('Data Loaded!')
print()

frames_train = [study_data_elbow['train'],
                study_data_finger['train'],
                study_data_forearm['train'],
                study_data_hand['train'],
                study_data_wrist['train'],
                study_data_shoulder['train'],
                study_data_humerus['train']]

frames_dev = [study_data_elbow['valid'],
              study_data_finger['valid'],
              study_data_forearm['valid'],
              study_data_hand['valid'],
              study_data_wrist['valid'],
              study_data_shoulder['valid'],
              study_data_humerus['valid']]

for_test_dev = pd.concat(frames_dev)

# Shuffle it first and then split it
# Set random state so the shuffling will always have the same result
for_test_dev = for_test_dev.sample(frac=1, random_state=seed)

study_data = {'train': pd.concat(frames_train), 'valid': for_test_dev.iloc[700:], 'test': for_test_dev.iloc[:700]}

# FOR TESTING PURPOSES -- PER STUDY

# study_data_elbow = get_study_level_data(study_elbow)
# frames_train = [study_data_elbow['train']]
# frames_dev = [study_data_elbow['valid']]

# study_data_finger = get_study_level_data(study_finger)
# frames_train = [study_data_finger['train']]
# frames_dev = [study_data_finger['valid']]

# study_data_forearm = get_study_level_data(study_forearm)
# frames_train = [study_data_forearm['train']]
# frames_dev = [study_data_forearm['valid']]

# study_data_hand = get_study_level_data(study_hand)
# frames_train = [study_data_hand['train']]
# frames_dev = [study_data_hand['valid']]

# study_data_wrist = get_study_level_data(study_wrist)
# frames_train = [study_data_wrist['train']]
# frames_dev = [study_data_wrist['valid']]

# study_data_shoulder = get_study_level_data(study_shoulder)
# frames_train = [study_data_shoulder['train']]
# frames_dev = [study_data_shoulder['valid']]

# study_data_humerus = get_study_level_data(study_humerus)
# frames_train = [study_data_humerus['train']]
# frames_dev = [study_data_humerus['valid']]

# for_test_dev = pd.concat(frames_dev)
# for_test_dev = for_test_dev.sample(frac=1, random_state=seed)
# study_data = {'train': pd.concat(frames_train), 'valid': for_test_dev.iloc[70:], 'test': for_test_dev.iloc[:70]}

# END FOR TESTING PURPOSES

# Create the dataloaders for the data
data_cat = ['train', 'valid', 'test']
dataloaders, image_shape = get_dataloaders(study_data, batch_size=1)
dataset_sizes = {x: len(study_data[x]) for x in data_cat}

# find weights for the positive class (as pos_weight)
# this loss will be different from the paper
# i think it makes sense to only do it in the training phase

# Abnormal is our positive / we find how many views are abnormal and normal
train_dataframe = study_data['train']
num_abnormal_images = train_dataframe[train_dataframe['Path'].str.contains('positive')]['Count'].sum()
num_normal_images = train_dataframe[train_dataframe['Path'].str.contains('negative')]['Count'].sum()

# Abnormal weight
pos_weight = torch.FloatTensor(np.array(num_abnormal_images / (num_abnormal_images + num_normal_images)))

# normal weight
# neg_weight = torch.FloatTensor(np.array(num_normal_images / (num_abnormal_images + num_normal_images)))

# weights for weighted binary cross entropy
# weights = [neg_weight, pos_weight]

# Set the learning rate, batch size, epochs and patience
lr = 0.001
batch_size = 64
epochs = 20
max_patience = 5

# Set if you want to resume the training
resume = False
# Set if you want to just evaluate the test dataset
eval_test = True

# ================================== DEFINE MODEL ================================== #

# model = PretrainedDensenetAverageVoting(hidden_size=500, num_class=1)

# model = PretrainedDensenetAverageVotingRELU(hidden_size=500, num_class=1)

# model = PretrainedDensenet(hidden_size=500, num_class=1)

# model = PretrainedDensenetRELU(hidden_size=500, num_class=1)

# model = PretrainedDensenetAverageVoting(hidden_size=500, num_class=1, frozen=False)

# model = PretrainedDensenetAverageVotingRELU(hidden_size=500, num_class=1, frozen=False)

# model = PretrainedDensenet(hidden_size=500, num_class=1, frozen=False)

model = PretrainedDensenetRELU(hidden_size=500, num_class=1, frozen=False)

# model = PretrainedResnet(hidden_size=500, num_class=1)

# model = MLP_With_Average_Voting(input_dim=3 * image_shape[0] * image_shape[1],
#                                 n_classes=1,
#                                 hidden_1=500,
#                                 hidden_2=200,
#                                 hidden_3=100,
#                                 dropout=0.3)

# model = MLP_With_Max_Pooling(input_dim=3 * image_shape[0] * image_shape[1],
#                              n_classes=1,
#                              hidden_1=500,
#                              hidden_2=200,
#                              hidden_3=100,
#                              dropout=0.3)

# model = CNN_With_Average_Voting(input_channels=3, input_shape=image_shape,
#                                  n_classes=1,
#                                  n_filters_1=10,
#                                  n_filters_2=20,
#                                  dropout=0.3)

# model = CNN_With_Max_Pooling(input_channels=3, input_shape=image_shape,
#                                  n_classes=1,
#                                  n_filters_1=10,
#                                  n_filters_2=20,
#                                  dropout=0.3)

# model = CNN_MLP_Average_Voting(input_channels=3, input_shape=image_shape,
#                                  n_classes=1,
#                                  n_filters_1=10,
#                                  n_filters_2=20,
#                                  hidden_size=500,
#                                  dropout=0.3)

# model = CNN_MLP_Max_Pooling(input_channels=3,
#                                  input_shape=image_shape,
#                                  n_classes=1,
#                                  n_filters_1=10,
#                                  n_filters_2=20,
#                                  hidden_size=500,
#                                  dropout=0.3)

# model = CNN_With_Average_VotingRELU(input_channels=3, input_shape=image_shape,
#                                  n_classes=1,
#                                  n_filters_1=10,
#                                  n_filters_2=20,
#                                  dropout=0.3)

# model = CNN_With_Max_PoolingRELU(input_channels=3, input_shape=image_shape,
#                                  n_classes=1,
#                                  n_filters_1=10,
#                                  n_filters_2=20,
#                                  dropout=0.3)

# model = CNN_MLP_Average_VotingRELU(input_channels=3, input_shape=image_shape,
#                                  n_classes=1,
#                                  n_filters_1=10,
#                                  n_filters_2=20,
#                                  hidden_size=500,
#                                  dropout=0.3)

# model = CNN_MLP_Max_PoolingRELU(input_channels=3,
#                                  input_shape=image_shape,
#                                  n_classes=1,
#                                  n_filters_1=10,
#                                  n_filters_2=20,
#                                  hidden_size=500,
#                                  dropout=0.3)

# ==================================              ================================== #

# Print the parameters of the model
print_params(model)

# Get the model parameters
paramaters = model.parameters()

# Set the loss function
loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# Set the optimizer
optimizer = torch.optim.Adam(params=paramaters, lr=lr)

# Get the dataset iterators
train_iterator = dataloaders['train']
dev_iterator = dataloaders['valid']
test_iterator = dataloaders['test']

# Initialize values for best auc and best epoch
best_auc = -1000.0
best_epoch = 0

# Load json file to store model results or create an empty dictionary
results_mura = None
if os.path.exists('/mnt/data/sotiris/results_mura.json'):
    with open('/mnt/data/sotiris/results_mura.json') as fi:
        results_mura = json.load(fi)
else:
    results_mura = dict()

if use_cuda:
    print()
    print('GPU available!!')
    print()
    model = model.cuda()


def evaluate(iterator, model):
    """
    Method that evaluates the dev/test sets
    :param iterator: The dataset iterator
    :param model: The model
    :return: Metrics for the set
    """
    # Perform all actions without keeping gradients
    with torch.no_grad():
        # Set the model to evaluation mode
        model.eval()
        # Initialize values and lists
        batch_preds = []
        eval_costs = []
        eval_cost = 0.0
        batch_labels = []
        aucs = []
        aps = []
        # Iterate the set
        for ev_batch in iterator:
            # Get the images and the labels
            dev_images = ev_batch['images']
            dev_labels = ev_batch['labels'].float()

            # Cast them to cuda if necessary
            if use_cuda:
                dev_images = dev_images.cuda()
                dev_labels = dev_labels.cuda()

            # Reset the gradients in the optimizer
            optimizer.zero_grad()
            # Pass the images through the model to get the predictions
            dev_preds = model(dev_images)
            # Calculate the accumulated loss
            eval_cost += float(loss(dev_preds, dev_labels).cpu().item())

            # Append the labels and preds to the batch lists
            batch_labels.append(dev_labels)
            batch_preds.append(dev_preds)

            # If we have reached the batch size
            if len(batch_preds) == batch_size:
                # Get the average of the losses and append it to the list
                eval_costs.append(eval_cost / batch_size)
                # Set the accumulated loss to 0
                eval_cost = 0
                # Pass the batch predictions through a sigmoid
                sigmoid_dev_preds = torch.sigmoid(torch.stack(batch_preds))
                # Calculate auc score
                dev_auc_easy = roc_auc_score(torch.stack(batch_labels).cpu().numpy(),
                                             sigmoid_dev_preds.cpu().numpy())
                # Calculate average precision
                average_precision = average_precision_score(torch.stack(batch_labels).cpu().numpy(),
                                                            sigmoid_dev_preds.cpu().numpy())
                # Append scores to the lists
                aucs.append(dev_auc_easy)
                aps.append(average_precision)
                # Reset the lists
                batch_labels = []
                batch_preds = []
    # Return metrics
    return dev_auc_easy, aucs, aps, eval_costs


def evaluate_cam(iterator, model, num_of_images):
    """
    Method that evaluates the dev/test set and also creates the gradCAM images
    :param iterator: The dataset iterator
    :param model: The model
    :param num_of_images: The number of images to get for CAM
    :return: Metrics for the set
    """
    # Set the model to evaluation mode
    model.eval()
    # Initialize values and lists
    batch_preds = []
    eval_costs = []
    eval_cost = 0.0
    batch_labels = []
    aucs = []
    aps = []
    img_i = 0
    dev_auc_easy = 0
    # Iterate the set
    for ev_batch in iterator:
        # Get the images and the labels
        dev_images = ev_batch['images']
        dev_labels = ev_batch['labels'].float()

        # Cast them to cuda if necessary
        if use_cuda:
            dev_images = dev_images.cuda()
            dev_labels = dev_labels.cuda()

        # Reset the gradients in the optimizer
        optimizer.zero_grad()

        # Create gradCAM images only for the first n instances
        if img_i <= num_of_images:
            # Generate heatmap
            # as in: https://medium.com/@stepanulyanin/implementing-grad-cam-in-pytorch-ea0937c31e82

            import cv2

            # Get the instance's path to the image file
            pathImageFiles = ev_batch['paths']
            # Set the output image's file
            pathOutputFile = 'cam_images/test{}.jpg'.format(img_i)
            # Increment the output image id
            img_i += 1

            # Get predictions with hook on the gradients
            cam_output = model.forward_cam(dev_images)

            # Legacy for dev -- so that we don't pass it 2 times
            dev_preds = cam_output
            eval_cost += float(loss(dev_preds, dev_labels).cpu().item())

            # Get the gradient of the output with respect to the parameters of the model
            cam_output.backward()

            # Pull the gradients out of the model
            gradients = model.get_activations_gradient()

            # Pool the gradients across the channels
            pooled_gradients = torch.mean(gradients, dim=[2, 3])

            # Get the activations of the last convolutional layer
            activations = model.get_activations(dev_images).detach()

            # Weight the channels by corresponding gradients
            for v in range(len(ev_batch['paths'][0])):
                for i in range(activations.shape[1]):
                    activations[v, i, :, :] *= pooled_gradients[v, i]

            # Average the channels of the activations
            heatmaps = torch.mean(activations, dim=1)

            # Create plot for the heatmaps and the superposed image
            import matplotlib.pyplot as plt
            fig, axis = plt.subplots(len(ev_batch['paths']), 2)
            if len(ev_batch['paths']) == 1:
                axis = axis.reshape(1, 2)
            fig.suptitle('/'.join(ev_batch['paths'][0][0].split('/')[5:-1]) +
                         '\nTrue: {} -- Predicted: {:.3f}'.format(dev_labels.cpu().item(),
                                                                          F.sigmoid(cam_output).cpu().item()))

            # For every view in the instance
            for v in range(len(ev_batch['paths'])):
                # leaky relu on top of the heatmap
                # or maybe better use relu
                # heatmap = F.leaky_relu(heatmaps[v])

                # Pass the heatmaps from a relu to throw negative scores
                heatmap = F.relu(heatmaps[v])

                # Normalize the heatmap
                h_max = torch.max(heatmap)
                if h_max != 0.0:
                    heatmap /= h_max

                # Save the heatmaps -- for debugging
                # plt.matshow(heatmap.cpu().numpy())
                # plt.savefig('{}_matrix.png'.format(v))
                # plt.clf()

                # Add the heatmap for hte view in the plot
                axis[v, 0].matshow(heatmap.cpu().numpy())
                axis[v, 0].axis('off')

                # Read the image from the path
                imgOriginal = cv2.imread(pathImageFiles[v][0])

                # Resize the heatmap to the image's dimensions
                heatmap = cv2.resize(heatmap.cpu().numpy(), (imgOriginal.shape[1], imgOriginal.shape[0]))

                # Cast heatmap values to [0,255] ints
                heatmap = np.uint8(255 * heatmap)

                # # Use opencv to superimpose image
                # heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                # img = heatmap * 0.4 + imgOriginal
                # cv2.imwrite('{}_'.format(v) + pathOutputFile, img)

                # Use matplotlib instead of opencv
                # plt.title('View: {}\nTrue: {} -- Predicted: {:.3f}'.format(v, dev_labels.cpu().item(), F.sigmoid(cam_output).cpu().item()))
                # plt.imshow(imgOriginal)
                # plt.imshow(heatmap, cmap='jet', alpha=0.4)
                # plt.savefig('{}_'.format(v) + pathOutputFile)
                # plt.clf()

                # Add superposed image to the plot
                axis[v, 1].imshow(imgOriginal)
                axis[v, 1].imshow(heatmap, cmap='jet', alpha=0.4)
                axis[v, 1].axis('off')

            # Save the instance plot
            fig.savefig(pathOutputFile, dpi=600)

            # END
        else:
            # Get the predictions from the model
            dev_preds = model(dev_images)
            # Calcualte the accumulated loss
            eval_cost += float(loss(dev_preds, dev_labels).cpu().item())

        # Append the labels and preds to the batch lists
        batch_labels.append(dev_labels)
        batch_preds.append(dev_preds)

        # If we have reached the batch size
        if len(batch_preds) == batch_size:
            # Get the average of the losses and append it to the list
            eval_costs.append(eval_cost / batch_size)
            # Set the accumulated loss to 0
            eval_cost = 0
            # Pass the batch predictions through a sigmoid
            sigmoid_dev_preds = torch.sigmoid(torch.stack(batch_preds))
            # Calculate auc score
            dev_auc_easy = roc_auc_score(torch.stack(batch_labels).cpu().detach().numpy(),
                                         sigmoid_dev_preds.cpu().detach().numpy())
            # Calculate average precision
            average_precision = average_precision_score(torch.stack(batch_labels).cpu().detach().numpy(),
                                                        sigmoid_dev_preds.cpu().detach().numpy())
            # Append scores to the lists
            aucs.append(dev_auc_easy)
            aps.append(average_precision)
            # Reset the lists
            batch_labels = []
            batch_preds = []
    # Return metrics
    return dev_auc_easy, aucs, aps, eval_costs

# Initialize values
epoch = -1
patience = max_patience

# If resume is set
if resume:
    # Use cuda if possible
    if torch.cuda.is_available():
        print()
        print('GPU available..will resume training!!')
        print()
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Load the checkpoint
    modelcheckpoint = torch.load(os.path.join(odir_checkpoint, progress_checkpoint), map_location=device)
    model.load_state_dict(modelcheckpoint['state_dict'])
    epoch = modelcheckpoint['epoch']
    optimizer.load_state_dict(modelcheckpoint['optimizer'])
    best_auc = modelcheckpoint['auc']
    patience = modelcheckpoint['patience']
    print()
    print('Resuming from file: {}'.format(progress_checkpoint))
    print('Best auc was: {}'.format(best_auc))
    print('The epoch was: {}'.format(epoch))
    print()
    logger.info('')
    logger.info('Resuming from file: {}'.format(progress_checkpoint))
    logger.info('Best auc was: {}'.format(best_auc))
    logger.info('The epoch was: {}'.format(epoch))
    logger.info('')
    # If resumed model has already early stopped
    if patience == 0:
        print()
        print('Resumed model has patience 0, which means it had early stopped.')
        print('Quitting training...')
        print()
        logger.info('')
        logger.info('Resumed model has patience 0, which means it had early stopped.')
        logger.info('Quitting training...')
        logger.info('')
    # If resumed model has already finished training (reached max epochs)
    if epoch == epochs - 1:
        print()
        print('Resumed model has already been trained for the max epochs.')
        print('Quitting training...')
        print()
        logger.info('')
        logger.info('Resumed model has already been trained for the max epochs.')
        logger.info('Quitting training...')
        logger.info('')

# If patience isn't 0 and the model hasn't reached the max epochs
if patience != 0 and epoch != epochs - 1 and not eval_test:
    print()
    print('Training model...')
    logger.info('')
    logger.info('Training model...')
    # For each epoch
    for epoch in tqdm(range(epoch + 1, epochs)):

        print()
        print('Epoch: {}'.format(epoch))
        print()
        logger.info('')
        logger.info('Epoch: {}'.format(epoch))
        logger.info('')

        # Set the model to train mode
        model.train()
        # Initialize lists
        batch_costs = []
        batch_logits = []
        batch_labels = []
        epoch_costs = []
        train_aucs = []
        dev_aucs = []
        dev_aps = []
        # Reset optimizer gradients to zero
        optimizer.zero_grad()

        # Iterate the set
        for batch in tqdm(train_iterator):
            # Get the images and the labels
            images = batch['images']
            labels = batch['labels'].float()
            # Cast them to cuda if necessary
            if use_cuda:
                images = images.cuda()
                labels = labels.cuda()
            # Get the logits from the model
            logits = model(images)
            # Append to list
            batch_logits.append(logits)
            batch_labels.append(labels)
            # Calculate the loss
            cost = loss(logits, labels)

            # cost = weighted_binary_cross_entropy(logits, labels, weights)

            # Perform clipping
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)

            # Append the loss to the list
            batch_costs.append(cost)

            # If we reached batch size
            if len(batch_costs) == batch_size:
                # Perform back propagation for the batch
                batch_aver_cost = back_prop(batch_costs)
                epoch_costs.append(batch_aver_cost)
                # Calculate the auc score
                train_auc = roc_auc_score(torch.stack(batch_labels).cpu().detach().numpy(),
                                          torch.sigmoid(torch.stack(batch_logits)).cpu().detach().numpy())
                # Append to list
                train_aucs.append(train_auc)
                batch_costs = []
                batch_logits = []
                batch_labels = []

        print('Epoch Average Loss: {}, Epoch Average AUC: {}, Epoch: {} '.format(
            sum(epoch_costs) / float(len(epoch_costs)), np.mean(train_aucs), epoch))

        logger.info('Epoch Average Loss: {}, Epoch Average AUC: {}, Epoch: {} '.format(
            sum(epoch_costs) / float(len(epoch_costs)), np.mean(train_aucs), epoch))

        print()
        print(40 * '*')
        print('Evaluating on the dev set...')
        print()

        logger.info('')
        logger.info(40 * '*')
        logger.info('Evaluating on the dev set...')
        logger.info('')
        # Evaluate on the dev set
        dev_auc, dev_aucs, dev_aps, dev_costs = evaluate(dev_iterator, model)

        print('Average Loss on Dev Set: {}, Epoch: {}'.format(np.mean(dev_costs), epoch))
        print('AUC on Dev Set: {}, Epoch: {}'.format(np.mean(dev_aucs), epoch))
        print('Average Precision on dev set: {}, Epoch: {}'.format(np.mean(dev_aps), epoch))
        print(40 * '*')

        logger.info('Average Loss on Dev Set: {}, Epoch: {}'.format(np.mean(dev_costs), epoch))
        logger.info('AUC on Dev Set: {}, Epoch: {}'.format(np.mean(dev_aucs), epoch))
        logger.info('Average Precision on dev set: {}, Epoch: {}'.format(np.mean(dev_aps), epoch))
        logger.info(40 * '*')
        logger.info('')

        # If we found new best dev auc score
        if np.mean(dev_aucs) > best_auc:
            best_auc = np.mean(dev_aucs)
            best_epoch = epoch

            state = {'epoch': epoch,
                     'state_dict': model.state_dict(),
                     'optimizer': optimizer.state_dict(),
                     'auc': best_auc,
                     'patience': max_patience,
                     'best_epoch': best_epoch}

            # Reset patience
            patience = max_patience

            # Save the best checkpoint
            save_checkpoint(state, filename=os.path.join(odir_checkpoint, best_checkpoint_name))
        else:
            # Reduce patience by 1
            patience -= 1

        # Save progress checkpoint
        state = {'epoch': epoch,
                 'state_dict': model.state_dict(),
                 'optimizer': optimizer.state_dict(),
                 'auc': best_auc,
                 'patience': patience,
                 'best_epoch': best_epoch}

        save_checkpoint(state, filename=os.path.join(odir_checkpoint, progress_checkpoint))

        # Save metrics and info to the json
        model_name = best_checkpoint_name.split('.')[0]
        if model_name not in results_mura.keys():
            results_mura[model_name] = list()
        results_mura[model_name].append({
            'epoch': epoch,
            'patience': patience,
            'train_loss': sum(epoch_costs) / float(len(epoch_costs)),
            'train_auc': np.mean(train_aucs),
            'dev_loss': np.mean(dev_costs),
            'dev_auc': np.mean(dev_aucs),
            'dev_ap': np.mean(dev_aps),
            'best_epoch': best_epoch,
            'best_auc': best_auc
        })
        with open('/mnt/data/sotiris/results_mura.json', 'w') as out:
            json.dump(results_mura, out)

        # If the max_patience_th time it still hasn't improved then stop the training
        if patience == 0:
            print()
            print('Early stopping at epoch: {}'.format(epoch))
            print()
            logger.info('')
            logger.info('Early stopping at epoch: {}'.format(epoch))
            logger.info('')
            break

    print()
    print(40 * '-')
    print("Best AUC {} at epoch: {}".format(best_auc, best_epoch))
    print(40 * '-')

    print()
    print('=' * 90)
    print()

    logger.info('')
    logger.info(40 * '-')
    logger.info("Best AUC {} at epoch: {}".format(best_auc, best_epoch))
    logger.info(40 * '-')
    logger.info('')
    logger.info('')

    logger.info('=' * 90)

print()
print('Evaluating on the test set...')
print()

if use_cuda:
    print()
    print('GPU available...')
    print()
    device = torch.device('cuda')

else:
    device = torch.device('cpu')

# Load best checkpoint and eval on the test set
best_check = torch.load(os.path.join(odir_checkpoint, best_checkpoint_name), map_location=device)

model.load_state_dict(best_check['state_dict'])

if use_cuda:
    model = model.cuda()

# Evaluate the test set
# _, test_aucs, test_aps, _ = evaluate(test_iterator, model)

# Evaluate the test set and create gradcam images
_, test_aucs, test_aps, _ = evaluate_cam(test_iterator, model, 100)

print()
print('Best Epoch:', best_check['epoch'])
print('Best Auc on Dev Set:', best_check['auc'])
print('Auc on Test set:', np.mean(test_aucs))
print()
print('=' * 90)
print()

logger.info('Best Epoch: {}'.format((best_check['epoch'])))
logger.info('Best Auc on Dev Set: {}'.format(str(best_check['auc'])))
logger.info('Auc on Test set: {}'.format(str(np.mean(test_aucs))))
logger.info('=' * 90)
