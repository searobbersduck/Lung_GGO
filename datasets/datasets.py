import os
import torch

from torch.utils.data import Dataset, DataLoader

import sys
root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
print(root)
sys.path.append(root)
from external_lib.MedCommon.utils.data_aug_utils import DATA_AUGMENTATION_UTILS
sys.path.append(os.path.join(root, 'external_lib/torchio'))

import torchio as tio
import numpy as np

class GGO_DS(Dataset):
    def __init__(self, data_root, config_file, image_shape, phase):
        self.phase = phase
        self.image_files = []
        self.labels = []
        self.pos_image_files = []
        self.neg_image_files = []
        with open(config_file) as f:
            for line in f.readlines():
                line = line.strip()
                if line is None or len(line) == 0:
                    continue
                ss = line.split('\t')
                image_file = os.path.join(data_root, ss[0])
                if not os.path.isfile(image_file):
                    continue
                self.image_files.append(image_file)
                if ss[1] == 'pos':
                    label = 1
                    self.pos_image_files.append(image_file)
                else:
                    label = 0
                    self.neg_image_files.append(image_file)
                self.labels.append(label)
        
        subjects = []
        for image_file in self.image_files:
            subject = tio.Subject(src=tio.ScalarImage(image_file))
            subjects.append(subject)

        pos_subjects = []
        for image_file in self.pos_image_files:
            subject = tio.Subject(src=tio.ScalarImage(image_file))
            pos_subjects.append(subject)
        
        neg_subjects = []
        for image_file in self.pos_image_files:
            subject = tio.Subject(src=tio.ScalarImage(image_file))
            neg_subjects.append(subject)

        if phase == 'train':
            self.transforms = DATA_AUGMENTATION_UTILS.get_common_transform(image_shape, 'GAN')
        else:
            self.transforms = DATA_AUGMENTATION_UTILS.get_common_transform(image_shape, 'GAN_INFERENCE')
        
        self.subjects_dataset = tio.SubjectsDataset(subjects, transform=self.transforms)
        if phase == 'train':
            self.pos_subjects_dataset = tio.SubjectsDataset(pos_subjects, transform=self.transforms)
            self.neg_subjects_dataset = tio.SubjectsDataset(neg_subjects, transform=self.transforms)
            self.n_pos = self.pos_subjects_dataset.__len__()
            self.n_neg = self.neg_subjects_dataset.__len__()

    def __len__(self):
        return self.subjects_dataset.__len__()

    def __getitem__(self, item):
        if self.phase == 'train':
            # if np.random.rand() < 0.5:
            #     image = self.pos_subjects_dataset.__getitem__(item%self.n_pos)
            #     label = 1
            # else:
            #     image = self.pos_subjects_dataset.__getitem__(item%self.n_neg)
            #     label = 0
            image = self.subjects_dataset.__getitem__(item)
            label = self.labels[item]
        else:
            image = self.subjects_dataset.__getitem__(item)
            label = self.labels[item]
        image = image['src']['data'].float()
        return image, label


def test_GGO_DS():
    data_root = '/data/medical/hospital/cz/ggo/cz/block_66'
    config_file = '/data/medical/hospital/cz/ggo/cz/config/val.txt'
    image_shape = [64, 64, 64]
    phase = 'train'

    ds = GGO_DS(data_root, config_file, image_shape, phase='val')
    dataloader = DataLoader(ds, batch_size=2, shuffle=True, drop_last=True, num_workers=1, pin_memory=False)

    for index, (images, labels) in enumerate(dataloader):
        print(images.shape)
        print('hello world!')



if __name__ == '__main__':
    test_GGO_DS()