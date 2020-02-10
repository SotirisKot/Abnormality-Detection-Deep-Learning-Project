"""

This file will parse the results_mura.json file in order to create plots for the models

"""

import matplotlib.pyplot as plt
import json
import os

plots_folder = '/mnt/data/sotiris/mura_plots/'

# Create plots folder if it does not exist
if not os.path.exists(plots_folder):
    os.makedirs(plots_folder)

# Open and load the json file
with open('/mnt/data/sotiris/results_mura.json') as fi:
    results_mura = json.load(fi)  # type: dict

# Iterate all the model keys in the dictionary
for k in results_mura.keys():
    # Get the model results
    results = results_mura[k]
    # Get the best epoch and its auc score
    epochs = results[-1]['epoch']
    best_epoch = results[-1]['best_epoch']
    best_auc = results[-1]['best_auc']
    # Create lists from the results
    train_loss = [results[i]['train_loss'] for i in range(len(results))]
    train_auc = [results[i]['train_auc'] for i in range(len(results))]
    dev_loss = [results[i]['dev_loss'] for i in range(len(results))]
    dev_auc = [results[i]['dev_auc'] for i in range(len(results))]
    dev_ap = [results[i]['dev_ap'] for i in range(len(results))]
    # Plot the results
    x = [i for i in range(0, epochs+1)]
    # Train and Dev Losses
    plt.title('Best Epoch: {} -- Best AUC: {}'.format(best_epoch, best_auc))
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.ylim(bottom=0)
    plt.plot(x, train_loss, 'b-o', label='Train Loss')
    plt.plot(x, dev_loss, 'r-o', label='Dev Loss')
    plt.xticks(x)
    plt.legend()
    plt.savefig(os.path.join(plots_folder, '{}_losses.png'.format(k)))
    plt.clf()
    # Train and Dev AUC Scores
    plt.title('Best Epoch: {} -- Best AUC: {}'.format(best_epoch, best_auc))
    plt.xlabel('Epochs')
    plt.ylabel('AUC Score')
    plt.ylim(bottom=0)
    plt.plot(x, train_auc, 'b-o', label='Train AUC')
    plt.plot(x, dev_auc, 'r-o', label='Dev AUC')
    plt.xticks(x)
    plt.legend()
    plt.savefig(os.path.join(plots_folder, '{}_aucs.png'.format(k)))
    plt.clf()
    # Dev AP Score
    plt.title('Best Epoch: {} -- Best AUC: {}'.format(best_epoch, best_auc))
    plt.xlabel('Epochs')
    plt.ylabel('AP Score')
    plt.ylim(bottom=0)
    plt.plot(x, dev_auc, 'r-o', label='Dev AP')
    plt.xticks(x)
    plt.legend()
    plt.savefig(os.path.join(plots_folder, '{}_aps.png'.format(k)))
    plt.clf()

# Train Loss of all models
for k in results_mura.keys():
    # Get the model results
    results = results_mura[k]
    # Create lists from the results
    train_loss = [results[i]['train_loss'] for i in range(len(results))]
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.ylim(bottom=0)
    plt.plot(train_loss, '-o', label='Train Loss {}'.format(k))

# Plot the results
epochs = 20
x = [i for i in range(0, epochs + 1)]
plt.xticks(x)
plt.legend()
plt.savefig(os.path.join(plots_folder, 'all_models_train_losses.png'))
plt.clf()

# Dev Loss of all models
for k in results_mura.keys():
    # Get the model results
    results = results_mura[k]
    # Create lists from the results
    dev_loss = [results[i]['dev_loss'] for i in range(len(results))]
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.ylim(bottom=0)
    plt.plot(dev_loss, '-o', label='Dev Loss {}'.format(k))

# Plot the results
epochs = 20
x = [i for i in range(0, epochs + 1)]
plt.xticks(x)
plt.legend()
plt.savefig(os.path.join(plots_folder, 'all_models_dev_losses.png'))
plt.clf()

# Train AUC of all models
for k in results_mura.keys():
    # Get the model results
    results = results_mura[k]
    # Create lists from the results
    train_auc = [results[i]['train_auc'] for i in range(len(results))]
    plt.xlabel('Epochs')
    plt.ylabel('Auc')
    plt.ylim(bottom=0)
    plt.plot(train_auc, '-o', label='Train Auc {}'.format(k))

# Plot the results
epochs = 20
x = [i for i in range(0, epochs + 1)]
plt.xticks(x)
plt.legend()
plt.savefig(os.path.join(plots_folder, 'all_models_train_aucs.png'))
plt.clf()

# Dev AUC of all models
for k in results_mura.keys():
    # Get the model results
    results = results_mura[k]
    # Create lists from the results
    dev_auc = [results[i]['dev_auc'] for i in range(len(results))]
    plt.xlabel('Epochs')
    plt.ylabel('Auc')
    plt.ylim(bottom=0)
    plt.plot(dev_auc, '-o', label='Dev Auc {}'.format(k))

# Plot the results
epochs = 20
x = [i for i in range(0, epochs + 1)]
plt.xticks(x)
plt.legend()
plt.savefig(os.path.join(plots_folder, 'all_models_dev_aucs.png'))
plt.clf()
