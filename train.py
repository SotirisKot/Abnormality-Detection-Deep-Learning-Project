seed = 1997

import random
from random import shuffle
random.seed(seed)

import numpy as np
np.random.seed(seed)

import os
import scipy.sparse as sp

import torch
use_cuda = torch.cuda.is_available()
torch.manual_seed(seed)
if use_cuda:
    torch.cuda.manual_seed(seed)

from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import pickle
from torch.utils.data import DataLoader
from models import MLP_With_Average_Pooling, PretrainedDensenet, CNN_With_Average_Pooling
from sklearn.metrics import roc_curve, auc, roc_auc_score, average_precision_score
import re
import argparse
import logging


from dataloader import get_study_level_data, get_dataloaders

odir_checkpoint = './checkpoints/'
# odir_checkpoint = './logs/'


hdlr  = None
study = 'XR_FINGER'

best_chekpoint_name = 'simple_cnn_no_dropout_{}.pth.tar'.format(study)

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


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


def init_the_logger(hdlr):
    if not os.path.exists(odir_checkpoint):
        os.makedirs(odir_checkpoint)
    od = odir_checkpoint.split('/')[-1]  # 'gcn_lr_skip_connections'
    logger = logging.getLogger(od)

    if (hdlr is not None):
        logger.removeHandler(hdlr)

    hdlr = logging.FileHandler(os.path.join(odir_checkpoint, 'simple_cnn_dropout_{}.log'.format(study)))
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)
    return logger, hdlr


logger, hdlr = init_the_logger(hdlr)


def back_prop(batch_costs):
    batch_cost = sum(batch_costs) / float(len(batch_costs))
    batch_cost.backward()
    optimizer.step()
    optimizer.zero_grad()
    batch_aver_cost = batch_cost.cpu().item()
    return batch_aver_cost


# HERE YOU PASS POSITIVE AND NEGATIVE WEIGHTS
# IT IS THE LOSS FROM THE PAPER
def weighted_binary_cross_entropy(output, target, weights=None):
    if weights is not None:
        assert len(weights) == 2
        loss = weights[1] * (target * torch.log(output)) + weights[0] * ((1 - target) * torch.log(1 - output))
    else:
        loss = target * torch.log(output) + (1 - target) * torch.log(1 - output)
    return torch.neg(torch.mean(loss))


# load data for one study
study_data = get_study_level_data(study_type=study)

# dataloaders for a study
data_cat = ['train', 'valid']
dataloaders, image_shape = get_dataloaders(study_data, batch_size=1)
dataset_sizes = {x: len(study_data[x]) for x in data_cat}


# find weights for the positive class (as pos_weight)
# this loss will be different from the paper
# i think it makes sense to only do it in the training phase

# abnormal is our positive / we find how many views are abnormal and normal
# not the studies
train_dataframe     = study_data['train']
num_abnormal_images = train_dataframe[train_dataframe['Path'].str.contains('positive')]['Count'].sum()
num_normal_images   = train_dataframe[train_dataframe['Path'].str.contains('negative')]['Count'].sum()

# abnormal weight
pos_weight          = torch.FloatTensor(np.array(num_abnormal_images / (num_abnormal_images + num_normal_images)))


lr                  = 0.001
batch_size          = 64
epochs              = 10


# some pretrained models that we can use
pretrained          = False
pretrained_model    = 'densenet169'

# pretrained_model    = 'densenet121'
# pretrained_model    = 'densenet201'
# pretrained_model    = 'densenet161'

# ================================== DEFINE MODEL ================================== #


if pretrained:
    model               = PretrainedDensenet(pretrained_model, num_class=1)
else:

    # model               = MLP_With_Average_Pooling(input_dim=3*image_shape,
    #                                                n_classes=1,
    #                                                hidden_1=5000,
    #                                                hidden_2=1000,
    #                                                hidden_3=100,
    #                                                dropout=0.3)

    model = CNN_With_Average_Pooling(input_channels=3,
                                     n_classes=1,
                                     n_filters_1=10,
                                     n_filters_2=20,
                                     dropout=0.3)


# ==================================              ================================== #

print_params(model)

paramaters          = model.parameters()

loss                = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer           = torch.optim.Adam(params=paramaters, lr=lr)

train_iterator      = dataloaders['train']
dev_iterator        = dataloaders['valid']

best_auc            = -1000.0
best_epoch          = 0


if use_cuda:
    print('GPU available!!')
    model = model.cuda()


def evaluate(iterator, model):
    with torch.no_grad():
        model.eval()
        batch_preds  = []
        batch_labels = []
        aucs         = []
        aps          = []
        for ev_batch in iterator:

            dev_images = ev_batch['images']
            dev_labels = ev_batch['labels']

            if use_cuda:
                dev_images = dev_images.cuda()
                dev_labels = dev_labels.cuda()

            optimizer.zero_grad()
            dev_preds = model(dev_images)
            batch_labels.append(dev_labels)
            batch_preds.append(dev_preds)

            if len(batch_preds) == batch_size:

                sigmoid_dev_preds = torch.sigmoid(torch.stack(batch_preds))

                dev_auc_easy = roc_auc_score(torch.stack(batch_labels).cpu().numpy(),
                                             sigmoid_dev_preds.cpu().numpy())
                average_precision = average_precision_score(torch.stack(batch_labels).cpu().numpy(),
                                                            sigmoid_dev_preds.cpu().numpy())

                aucs.append(dev_auc_easy)
                aps.append(average_precision)

                batch_labels = []
                batch_preds  = []

    return dev_auc_easy, aucs, aps


for epoch in range(epochs):

    model.train()
    batch_costs   = []
    batch_logits  = []
    batch_labels  = []
    epoch_costs   = []
    train_aucs    = []
    dev_aucs      = []
    dev_aps       = []
    optimizer.zero_grad()

    for batch in tqdm(train_iterator):

        images = batch['images']
        labels = batch['labels'].float()

        if use_cuda:
            images = images.cuda()
            labels = labels.cuda()

        logits = model(images)
        batch_logits.append(logits)
        batch_labels.append(labels)

        cost   = loss(logits, labels)

        # CLIPPING IF IT IS NEEDED
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
        batch_costs.append(cost)

        if len(batch_costs) == batch_size:

            batch_aver_cost = back_prop(batch_costs)
            epoch_costs.append(batch_aver_cost)

            train_auc_easy = roc_auc_score(torch.stack(batch_labels).cpu().detach().numpy(),
                                           torch.sigmoid(torch.stack(batch_logits)).cpu().detach().numpy())
            train_aucs.append(train_auc_easy)

            batch_costs  = []
            batch_logits = []
            batch_labels = []

    print('Epoch Average Loss: {}, Epoch Average AUC: {}, Epoch: {} '.format(
        sum(epoch_costs) / float(len(epoch_costs)), np.mean(train_aucs), epoch))

    logger.info('Epoch Average Loss: {}, Epoch Average AUC: {}, Epoch: {} '.format(
        sum(epoch_costs) / float(len(epoch_costs)), np.mean(train_aucs), epoch))

    print()
    print('Evaluating on the dev set...')
    print()

    logger.info('Evaluating on the dev set...')
    # evaluate on the dev set
    dev_auc, dev_aucs, dev_aps = evaluate(dev_iterator, model)

    print(40 * '*')
    print('AUC on Dev Set: {}, Epoch: {}'.format(np.mean(dev_aucs), epoch))
    print('Average Precision on dev set: {}, Epoch: {}'.format(np.mean(dev_aps), epoch))
    print(40 * '*')

    logger.info(40 * '*')
    logger.info('AUC on Dev Set: {}, Epoch: {}'.format(np.mean(dev_aucs), epoch))
    logger.info('Average Precision on dev set: {}, Epoch: {}'.format(np.mean(dev_aps), epoch))
    logger.info(40 * '*')

    if np.mean(dev_aucs) > best_auc:
        best_auc = np.mean(dev_aucs)
        best_epoch = epoch

        state = {'epoch': epoch + 1,
                 'state_dict': model.state_dict(),
                 'optimizer': optimizer.state_dict(),
                 'auc': best_auc,
                 'best_epoch': best_epoch}

        save_checkpoint(state, filename=os.path.join(odir_checkpoint, best_chekpoint_name))


print(40*'-')
print("Best AUC {} at epoch: {}".format(best_auc, best_epoch))
print(40*'-')

print('=' * 90)
print()

logger.info(40*'-')
logger.info("Best AUC {} at epoch: {}".format(best_auc, best_epoch))
logger.info(40*'-')

logger.info('=' * 90)