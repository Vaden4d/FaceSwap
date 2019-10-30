import numpy as np
import os

from utils import download_data

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt

class Data(Dataset):

    def __init__(self, data_path=None):
        '''
        Arguments:
            Path to train data folder (string): Folder with faces
        '''
        if data_path:
            #self.files = next(os.walk(data_path))[2]
            #self._data = download_data(data_path, self.files)
            self.files = np.load('../files.npy')
            self._data = torch.load('../images.pt')

            self.n = len(self.data)
        else:
            self.files = None
            self._data = None
            self.n = 0

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        self._data = value
        self.n = len(value)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return self.data[idx]

class Splitter():

    def __init__(self, obj):

        self.obj = obj
        self.train_size = 1
        self.test_size = 0
        self.valid_size = 0

    def train_test_split(self, test_size=0.1, random_state=42):
        np.random.seed(random_state)

        self.test_size = test_size
        self.train_size = 1 - test_size

        self.train_max_index = int(self.train_size * self.obj.n)

        files = self.obj.files.copy()
        np.random.shuffle(files)

        train_files = files[:self.train_max_index]
        test_files = files[self.train_max_index:]

        train_obj = Data()
        test_obj = Data()

        train_obj.files = train_files
        test_obj.files = test_files

        train_obj.data = self.obj.data[:self.train_max_index]
        test_obj.data = self.obj.data[self.train_max_index:]

        return train_obj, test_obj


    def train_validation_test_split(self, valid_size=0.2, test_size=0.2, random_state=42):
        np.random.seed(random_state)

        self.test_size = test_size
        self.valid_size = valid_size
        self.train_size = 1 - test_size - valid_size

        self.train_max_index = int(self.train_size * self.obj.n)
        self.test_max_index = int((self.train_size + self.test_size) * self.obj.n)

        files = self.obj.files.copy()
        np.random.shuffle(files)

        train_files = files[:self.train_max_index]
        test_files = files[self.train_max_index:self.test_max_index]
        valid_files = files[self.test_max_index:]

        train_obj = Data()
        test_obj = Data()
        valid_obj = Data()

        train_obj.files = train_files
        test_obj.files = test_files
        valid_obj.files = valid_files

        train_obj.data = self.obj.data[:self.train_max_index]
        test_obj.data = self.obj.data[self.train_max_index:self.test_max_index]
        valid_obj.data = self.obj.data[self.test_max_index:]

        return train_obj, test_obj, valid_obj


if __name__ == '__main__':
    obj = Data('../only_faces_one')
    splitter = Splitter(obj)
    train, test, valid = splitter.train_validation_test_split()
    #print(len(train))
    #print(len(test))
    #print(len(valid))
    #print(train[1:3])
    print(train.files)
    print(test.files)
    print(valid.files)
