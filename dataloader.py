import os
import pandas as pd
from tqdm import tqdm
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets.folder import pil_loader


# CODE FROM https://github.com/pyaf/DenseNet-MURA-PyTorch/blob/master/pipeline.py
# WE CAN CHANGE IT/MAKE IT BETTER

data_cat = ['train', 'valid']  # data categories


def get_study_level_data(study_type):
    study_data = {}
    study_label = {'positive': 1, 'negative': 0}
    for phase in data_cat:

        # drive/My Drive/DeepLearningProject/
        BASE_DIR = 'MURA-v1.1/%s/%s/' % (phase, study_type)

        # BASE_DIR = 'drive/My Drive/DeepLearningProject/MURA-v1.1/%s/%s/' % (phase, study_type)
        patients = list(os.walk(BASE_DIR))[0][1]  # list of patient folder names
        study_data[phase] = pd.DataFrame(columns=['Path', 'Count', 'Label'])
        i = 0
        for patient in tqdm(patients):  # for each patient folder
            for study in os.listdir(BASE_DIR + patient):  # for each study in that patient folder
                label = study_label[study.split('_')[1]]  # get label 0 or 1
                path = BASE_DIR + patient + '/' + study + '/'  # path to this study
                study_data[phase].loc[i] = [path, len(os.listdir(path)), label]  # add new row
                i += 1
    return study_data


class MURA_dataset(Dataset):

    def __init__(self, df, transform=None):

        # the dataframe
        self.df = df

        # transform the image --> rescale, rotate, flip etc
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, image_idx):
        study_path = self.df.iloc[image_idx, 0]
        count = self.df.iloc[image_idx, 1]
        images = []
        for i in range(count):
            image = pil_loader(study_path + 'image%s.png' % (i + 1))
            images.append(self.transform(image))
        images = torch.stack(images)
        label = self.df.iloc[image_idx, 2]
        sample = {'images': images, 'labels': label}
        return sample


def get_dataloaders(data, batch_size=8, study_level=False):

    # IN THE PAPER THEY RESCALE THE IMAGES TO 320 x 320
    # THEY AUGMENT THE DATA WITH INVERSIONS AND ROTATIONS.
    image_shape = (100, 100)

    data_transforms = {
        'train': transforms.Compose([
                transforms.Resize((image_shape[0], image_shape[1])),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                # transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize((image_shape[0], image_shape[1])),
            # transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    image_datasets = {x: MURA_dataset(data[x], transform=data_transforms[x]) for x in data_cat}
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True) for x in data_cat}
    return dataloaders, image_shape[0] * image_shape[1]


# ================================ TEST THE DATALOADERS ================================ #

# I THINK WE CAN ONLY TRAIN FOR ONE STUDY_TYPE AT A TIME
# IN THE PAPER THEY ALSO HAVE DIFFERENT RESULTS FOR EACH STUDY_TYPE
# THE BEST RESULTS ARE ON FINGER AND WRIST STUDIES

# study_data = get_study_level_data(study_type='XR_WRIST')
#
#
# # MAYBE IT IS WRONG TO SET BATCH_SIZE > 1. BECAUSE THEN YOU FEED FOR EXAMPLE STUDIES WHERE EACH STUDY HAS ONE OR MORE
# # VIEWS. SOME HAVE 3 VIEWS, OTHER HAS 7 VIEWS. WHEN YOU BATCH THE STUDIES YOU STILL NEED TO MASK THE VIEWS THAT BELONG
# # TO EACH STUDY.
# # IF WE SET BATCH_SIZE = 1, THEN WE JUST NEED TO CLASSIFY ONE STUDY WITH ONE OR MORE VIEWS.
# # WE CAN FORWARD MANY STUDIES AND THEN BACKPROPAGATE THROUGH ALL OF THEM LIKE DIMITRIS DOES.
#
# dataloaders = get_dataloaders(study_data, batch_size=1)
# dataset_sizes = {x: len(study_data[x]) for x in data_cat}
#
#
# # test the dataloaders
#
# for batch in dataloaders[0]['train']:
#     print(batch)

