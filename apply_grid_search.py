"""

In this file we run grid search to finetune hyperparameters

"""

# Imports

import pickle
import tempfile
import time
import logging
import click
import random
import numpy as np
import os
import torch
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, pyll
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import pickle
from torch.utils.data import DataLoader
from models import MLP_With_Average_Voting, PretrainedDensenet, PretrainedResnet, CNN_With_Average_Voting, \
    MLP_With_Max_Pooling, CNN_MLP_Average_Voting, CNN_MLP_Max_Pooling, PretrainedDensenetAverageVoting, \
    PretrainedDensenetRELU
from sklearn.metrics import roc_curve, auc, roc_auc_score, average_precision_score
import re
import argparse
import logging
import pandas as pd
import json
from dataloader import get_study_level_data, get_dataloaders

# Seed for our experiments
my_seed = 1997
random.seed(my_seed)
np.random.seed(my_seed)
torch.manual_seed(my_seed)

# Setting cuda for GPU if it is available
use_cuda = torch.cuda.is_available()
if use_cuda:
    torch.cuda.manual_seed(my_seed)

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

    if (hdlr is not None):
        logger.removeHandler(hdlr)

    # FOR GRID SEARCH
    hdlr = logging.FileHandler(os.path.join(odir_checkpoint, 'grid_search_tests.log'))

    # Set the format for the logger
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)

    return logger, hdlr

# Initialize the logger
logger, hdlr = init_the_logger(hdlr)


def back_prop(batch_costs, optimizer):
    """
    Perform back propagation for a batch
    :param batch_costs: The costs for the batch
    :param optimizer: The optimizer
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


# load data for one study
# study_data = get_study_level_data(study_type=study_humerus)

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

# print()
for_test_dev = pd.concat(frames_dev)

# shuffle it first and then split it
# set random state so the shuffling will always have the same result
for_test_dev = for_test_dev.sample(frac=1, random_state=my_seed)

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

# Create the dataloaders
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

def evaluate(iterator, model, optimizer, loss, batch_size):
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


def run_model(search_space, run_id):
    """
    This method runs a model with specific hyperparameters
    :param search_space: The search space for the model and hyperparameters
    :param run_id: The run id of the grid search run
    :return: Loss and run_id
    """
    # Unpack search space
    input_dim = search_space['input_dim']
    num_classes = search_space['num_classes']
    hidden1 = search_space['hidden1']
    hidden2 = search_space['hidden2']
    hidden3 = search_space['hidden3']
    dropout = search_space['dropout']
    num_filters1 = search_space['num_filters1']
    num_filters2 = search_space['num_filters2']
    input_channels = search_space['input_channels']
    learning_rate = search_space['learning_rate']
    epochs = search_space['epochs']
    max_patience = search_space['max_patience']
    resume = search_space['resume']
    batch_size = search_space['batch_size']
    what_do_you_want = search_space['what_do_you_want']

    # Save the trials object before starting each new experiment
    if not os.path.exists('/mnt/data/sotiris/checkpoints/{}/'.format(what_do_you_want)):
        os.makedirs('/mnt/data/sotiris/checkpoints/{}/'.format(what_do_you_want))

    # with open('/mnt/data/sotiris/checkpoints/{}/{}_grid_search_trials.p'.format(what_do_you_want, what_do_you_want),
    #           'wb') as out:
    #     pickle.dump(trials, out)

    # Initialize the correct model according to the variable
    if what_do_you_want == 'densenet_averagevoting':
        model = PretrainedDensenetAverageVoting(hidden_size=hidden1, num_class=num_classes)

    elif what_do_you_want == 'densenet_maxpooling':
        model = PretrainedDensenet(hidden_size=hidden1, num_class=num_classes)

    elif what_do_you_want == 'densenet_maxpooling_relu':
        model = PretrainedDensenetRELU(hidden_size=hidden1, num_class=num_classes)

    elif what_do_you_want == 'resnet':
        model = PretrainedResnet(hidden_size=hidden1, num_class=num_classes)

    elif what_do_you_want == 'mlp_averagevoting':
        model = MLP_With_Average_Voting(input_dim=input_dim,
                                        n_classes=num_classes,
                                        hidden_1=hidden1,
                                        hidden_2=hidden2,
                                        hidden_3=hidden3,
                                        dropout=dropout)

    elif what_do_you_want == 'mlp_maxpooling':
        model = MLP_With_Max_Pooling(input_dim=input_dim,
                                     n_classes=num_classes,
                                     hidden_1=hidden1,
                                     hidden_2=hidden2,
                                     hidden_3=hidden3,
                                     dropout=dropout)

    elif what_do_you_want == 'cnn_averagevoting':
        model = CNN_With_Average_Voting(input_channels=input_channels,
                                        input_shape=image_shape,
                                        n_classes=num_classes,
                                        n_filters_1=num_filters1,
                                        n_filters_2=num_filters2,
                                        dropout=dropout)

    elif what_do_you_want == 'cnn_mlp_averagevoting':
        model = CNN_MLP_Average_Voting(input_channels=input_channels,
                                       input_shape=image_shape,
                                       n_classes=num_classes,
                                       n_filters_1=num_filters1,
                                       n_filters_2=num_filters2,
                                       hidden_size=hidden1,
                                       dropout=dropout)
    else:
        model = CNN_MLP_Max_Pooling(input_channels=input_channels,
                                    input_shape=image_shape,
                                    n_classes=num_classes,
                                    n_filters_1=num_filters1,
                                    n_filters_2=num_filters2,
                                    hidden_size=hidden1,
                                    dropout=dropout)

    # Print the parameters of the model
    print_params(model)

    # Get the model parameters
    paramaters = model.parameters()

    # Set the loss function
    loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Set the optimizer
    optimizer = torch.optim.Adam(params=paramaters, lr=learning_rate)

    # Get the dataset iterators
    train_iterator = dataloaders['train']
    dev_iterator = dataloaders['valid']
    test_iterator = dataloaders['test']

    # Initialize values for best auc and best epoch
    best_auc = -1000.0
    best_epoch = 0

    # Load json file to store model results or create an empty dictionary
    results_grid_search = None
    if os.path.exists('/mnt/data/sotiris/results_grid_search.json'):
        with open('/mnt/data/sotiris/results_grid_search.json') as fi:
            results_grid_search = json.load(fi)
    else:
        results_grid_search = dict()

    if use_cuda:
        model = model.cuda()

    # Initialize values
    epoch = -1
    patience = max_patience

    # Run a trial
    if patience != 0 and epoch != epochs - 1:
        logger.info('')
        logger.info('Training model...')
        # For each epoch
        for epoch in tqdm(range(epoch + 1, epochs)):

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

            logger.info('Epoch Average Loss: {}, Epoch Average AUC: {}, Epoch: {} '.format(
                sum(epoch_costs) / float(len(epoch_costs)), np.mean(train_aucs), epoch))

            logger.info('')
            logger.info(40 * '*')
            logger.info('Evaluating on the dev set...')
            logger.info('')
            # Evaluate on the dev set
            dev_auc, dev_aucs, dev_aps, dev_costs = evaluate(iterator=dev_iterator,
                                                             model=model,
                                                             batch_size=batch_size,
                                                             loss=loss,
                                                             optimizer=optimizer)

            logger.info('Average Loss on Dev Set: {}, Epoch: {}'.format(np.mean(dev_costs), epoch))
            logger.info('AUC on Dev Set: {}, Epoch: {}'.format(np.mean(dev_aucs), epoch))
            logger.info('Average Precision on dev set: {}, Epoch: {}'.format(np.mean(dev_aps), epoch))
            logger.info(40 * '*')
            logger.info('')

            # Find checkpoint from variable what_do_you_want and trial
            best_checkpoint_name = 'grid_search_trial_{}.pth.tar'.format(run_id)

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
                if not os.path.exists(os.path.join(odir_checkpoint, '{}/'.format(what_do_you_want))):
                    os.makedirs(os.path.join(odir_checkpoint, '{}/'.format(what_do_you_want)))

                save_checkpoint(state, filename=os.path.join(odir_checkpoint, what_do_you_want, best_checkpoint_name))
            else:
                # Reduce patience by 1
                patience -= 1

            # save progress -- not needed here
            # state = {'epoch': epoch,
            #          'state_dict': model.state_dict(),
            #          'optimizer': optimizer.state_dict(),
            #          'auc': best_auc,
            #          'patience': patience,
            #          'best_epoch': best_epoch}
            #
            # save_checkpoint(state, filename=os.path.join(odir_checkpoint, progress_checkpoint))

            # Save metrics and info to the json
            model_name = what_do_you_want
            if model_name not in results_grid_search.keys():
                results_grid_search[model_name] = dict()
            if run_id not in results_grid_search[model_name].keys():
                results_grid_search[model_name][run_id] = dict()
                results_grid_search[model_name][run_id]['hyperparameters'] = search_space
                results_grid_search[model_name][run_id]['results'] = list()
            results_grid_search[model_name][run_id]['results'].append({
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
            with open('/mnt/data/sotiris/results_grid_search.json', 'w') as out:
                json.dump(results_grid_search, out)

            # If the max_patience_th time it still hasn't improved then stop the training
            if patience == 0:
                logger.info('')
                logger.info('Early stopping at epoch: {}'.format(epoch))
                logger.info('')
                break

    return {'best_auc': best_auc,
            'run_id': run_id}


# Initialize constant hyperparameters for all search spaces
input_dim = 3 * image_shape[0] * image_shape[1]
num_classes = 1
hidden2 = 200
hidden3 = 100
dropout = 0.3
num_filters1 = None
num_filters2 = None
input_channels = 3
epochs = 20
max_patience = 5
batch_size = 64

# Set to resume training
resume = False
# Set to choose model to fine tune
what_do_you_want = 'densenet_maxpooling_relu'

# Define search spaces for grid search

_search_space_1 = {'input_dim': input_dim,
                   'num_classes': num_classes,
                   'hidden1': 250,
                   'hidden2': hidden2,
                   'hidden3': hidden3,
                   'dropout': dropout,
                   'num_filters1': None,
                   'num_filters2': None,
                   'input_channels': input_channels,
                   'learning_rate': 0.0001,
                   'epochs': epochs,
                   'max_patience': max_patience,
                   'resume': resume,
                   'batch_size': batch_size,
                   'what_do_you_want': what_do_you_want}

_search_space_2 = {'input_dim': input_dim,
                   'num_classes': num_classes,
                   'hidden1': 250,
                   'hidden2': hidden2,
                   'hidden3': hidden3,
                   'dropout': dropout,
                   'num_filters1': None,
                   'num_filters2': None,
                   'input_channels': input_channels,
                   'learning_rate': 0.0005,
                   'epochs': epochs,
                   'max_patience': max_patience,
                   'resume': resume,
                   'batch_size': batch_size,
                   'what_do_you_want': what_do_you_want}

_search_space_3 = {'input_dim': input_dim,
                   'num_classes': num_classes,
                   'hidden1': 250,
                   'hidden2': hidden2,
                   'hidden3': hidden3,
                   'dropout': dropout,
                   'num_filters1': None,
                   'num_filters2': None,
                   'input_channels': input_channels,
                   'learning_rate': 0.001,
                   'epochs': epochs,
                   'max_patience': max_patience,
                   'resume': resume,
                   'batch_size': batch_size,
                   'what_do_you_want': what_do_you_want}

_search_space_4 = {'input_dim': input_dim,
                   'num_classes': num_classes,
                   'hidden1': 500,
                   'hidden2': hidden2,
                   'hidden3': hidden3,
                   'dropout': dropout,
                   'num_filters1': None,
                   'num_filters2': None,
                   'input_channels': input_channels,
                   'learning_rate': 0.0001,
                   'epochs': epochs,
                   'max_patience': max_patience,
                   'resume': resume,
                   'batch_size': batch_size,
                   'what_do_you_want': what_do_you_want}

_search_space_5 = {'input_dim': input_dim,
                   'num_classes': num_classes,
                   'hidden1': 500,
                   'hidden2': hidden2,
                   'hidden3': hidden3,
                   'dropout': dropout,
                   'num_filters1': None,
                   'num_filters2': None,
                   'input_channels': input_channels,
                   'learning_rate': 0.0005,
                   'epochs': epochs,
                   'max_patience': max_patience,
                   'resume': resume,
                   'batch_size': batch_size,
                   'what_do_you_want': what_do_you_want}

_search_space_6 = {'input_dim': input_dim,
                   'num_classes': num_classes,
                   'hidden1': 500,
                   'hidden2': hidden2,
                   'hidden3': hidden3,
                   'dropout': dropout,
                   'num_filters1': None,
                   'num_filters2': None,
                   'input_channels': input_channels,
                   'learning_rate': 0.001,
                   'epochs': epochs,
                   'max_patience': max_patience,
                   'resume': resume,
                   'batch_size': batch_size,
                   'what_do_you_want': what_do_you_want}

_search_space_7 = {'input_dim': input_dim,
                   'num_classes': num_classes,
                   'hidden1': 750,
                   'hidden2': hidden2,
                   'hidden3': hidden3,
                   'dropout': dropout,
                   'num_filters1': None,
                   'num_filters2': None,
                   'input_channels': input_channels,
                   'learning_rate': 0.0001,
                   'epochs': epochs,
                   'max_patience': max_patience,
                   'resume': resume,
                   'batch_size': batch_size,
                   'what_do_you_want': what_do_you_want}

_search_space_8 = {'input_dim': input_dim,
                   'num_classes': num_classes,
                   'hidden1': 750,
                   'hidden2': hidden2,
                   'hidden3': hidden3,
                   'dropout': dropout,
                   'num_filters1': None,
                   'num_filters2': None,
                   'input_channels': input_channels,
                   'learning_rate': 0.0005,
                   'epochs': epochs,
                   'max_patience': max_patience,
                   'resume': resume,
                   'batch_size': batch_size,
                   'what_do_you_want': what_do_you_want}

_search_space_9 = {'input_dim': input_dim,
                   'num_classes': num_classes,
                   'hidden1': 750,
                   'hidden2': hidden2,
                   'hidden3': hidden3,
                   'dropout': dropout,
                   'num_filters1': None,
                   'num_filters2': None,
                   'input_channels': input_channels,
                   'learning_rate': 0.001,
                   'epochs': epochs,
                   'max_patience': max_patience,
                   'resume': resume,
                   'batch_size': batch_size,
                   'what_do_you_want': what_do_you_want}

search_space_for_grid = [_search_space_1,
                         _search_space_2,
                         _search_space_3,
                         _search_space_4,
                         _search_space_5,
                         _search_space_6,
                         _search_space_7,
                         _search_space_8,
                         _search_space_9]

# Run the trials
runs = {}
for idx, s_space in enumerate(search_space_for_grid):
    runs[idx] = run_model(s_space, idx)

print(runs)

# with open('/mnt/data/sotiris/checkpoints/{}/{}_hyperopt_trials.p'.format(what_do_you_want, what_do_you_want),
#           'wb') as out:
#     pickle.dump(trials, out)
