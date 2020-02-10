"""

In this file we create the MURA Dataset object

"""

import os
import pandas as pd
from tqdm import tqdm
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets.folder import pil_loader


# CODE FROM https://github.com/pyaf/DenseNet-MURA-PyTorch/blob/master/pipeline.py
# WE CAN CHANGE IT/MAKE IT BETTER

# Data categories
data_cat = ['train', 'valid', 'test']


def get_study_level_data(study_type):
    study_data = {}
    study_label = {'positive': 1, 'negative': 0}
    for phase in data_cat:

        if phase == 'test':
            continue

        BASE_DIR = '/mnt/data/sotiris/MURA-v1.1/%s/%s/' % (phase, study_type)
        # BASE_DIR = 'drive/My Drive/MURA Project/MURA-v1.1/%s/%s/' % (phase, study_type)

        patients = os.listdir(BASE_DIR)  # list of patient folder names
        study_data[phase] = pd.DataFrame(columns=['Path', 'Count', 'Label'])
        i = 0
        for patient in tqdm(patients):  # for each patient folder
            for study in os.listdir(BASE_DIR + patient):  # for each study in that patient folder
                label = study_label[study.split('_')[1]]  # get label 0 or 1
                path = BASE_DIR + patient + '/' + study + '/'  # path to this study
                study_data[phase].loc[i] = [path, len(os.listdir(path)), label]  # add new row
                i += 1
    return study_data


"""
Dataset object wrapper for MURA Dataset
"""
class MURA_dataset(Dataset):

    def __init__(self, df, transform=None):

        # The dataframe
        self.df = df

        # Transform the image --> rescale, rotate, flip etc
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, image_idx):
        # Get the study path
        study_path = self.df.iloc[image_idx, 0]
        count = self.df.iloc[image_idx, 1]
        images = []
        paths = []
        for i in range(count):
            # We added the if os exists because of the corrupted MURA dataset that has corrupted images starting with ._
            # Load the image using the path
            if os.path.exists(study_path + 'image%s.png' % (i + 1)):
                image = pil_loader(study_path + 'image%s.png' % (i + 1))
                paths.append(study_path + 'image%s.png' % (i + 1))
                images.append(self.transform(image))
            else:
                break
        # Stack the views of the study and create the sample
        images = torch.stack(images)
        label = self.df.iloc[image_idx, 2]
        sample = {'images': images, 'paths': paths, 'labels': label}
        return sample


def get_dataloaders(data, batch_size=8, study_level=False):
    """
    Returns the dataloaders for train, dev and test set
    """
    # In the paper they rescale the images to 320x320
    # They augment the data with inversion and rotations
    # We use 224x224 because the pretrained densenet is trained on that dimensions

    image_shape = (224, 224)
    # image_shape = (100, 100)

    # Set the data augmentation transforms for each set
    data_transforms = {
        'train': transforms.Compose([
                transforms.Resize((image_shape[0], image_shape[1])),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                # transforms.Grayscale(),
                transforms.ToTensor(),
                # these normalization are from the densenet
                # mean = [0.485, 0.456, 0.406]
                # std  = [0.229, 0.224, 0.225]
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize((image_shape[0], image_shape[1])),
            # transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize((image_shape[0], image_shape[1])),
            # transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
    # Apply the transformations and create the datasets
    image_datasets = {x: MURA_dataset(data[x], transform=data_transforms[x]) for x in data_cat}
    # Create the dataloaders from the datasets
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True) for x in data_cat}
    # Return the dataloaders and the image shape
    return dataloaders, image_shape


# ================================ TEST THE DATALOADERS ================================ #

# I THINK WE CAN ONLY TRAIN FOR ONE STUDY_TYPE AT A TIME
# IN THE PAPER THEY ALSO HAVE DIFFERENT RESULTS FOR EACH STUDY_TYPE
# THE BEST RESULTS ARE ON FINGER AND WRIST STUDIES

# study_data = get_study_level_data(study_type='XR_ELBOW')
#
#
# # MAYBE IT IS WRONG TO SET BATCH_SIZE > 1. BECAUSE THEN YOU FEED FOR EXAMPLE STUDIES WHERE EACH STUDY HAS ONE OR MORE
# # VIEWS. SOME HAVE 3 VIEWS, OTHER HAS 7 VIEWS. WHEN YOU BATCH THE STUDIES YOU STILL NEED TO MASK THE VIEWS THAT BELONG
# # TO EACH STUDY.
# # IF WE SET BATCH_SIZE = 1, THEN WE JUST NEED TO CLASSIFY ONE STUDY WITH ONE OR MORE VIEWS.
# # WE CAN FORWARD MANY STUDIES AND THEN BACKPROPAGATE THROUGH ALL OF THEM.
#
# dataloaders = get_dataloaders(study_data, batch_size=1)
# dataset_sizes = {x: len(study_data[x]) for x in data_cat}
# #
# #
# # # test the dataloaders
# #
# for batch in tqdm(dataloaders[0]['train']):
#     # print(batch)
#     pass
