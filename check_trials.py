"""

This file is used in order to check the Trials object that hyperopt has saved

"""

import pickle, os

base_dir = '/mnt/data/sotiris/checkpoints/'

experiment = 'densenet_maxpooling_relu'

with open(os.path.join(base_dir, experiment, experiment+'_hyperopt_trials.p'), 'rb') as fi:
    trials = pickle.load(fi)

print()

