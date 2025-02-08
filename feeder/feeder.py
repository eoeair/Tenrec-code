# torch
import torch

# duckdb
import duckdb

# np
import numpy as np

class Feeder(torch.utils.data.Dataset):
    """ Feeder for RS
    Arguments:
        data_path: the path to data, the shape of data should be (N, C, T, V, M)
        mode: must be train or test
    """

    def __init__(self,
                 data_path,
                 mode,):
        self.data_path = data_path
        self.mode = mode

        self.load_data()

    def load_data(self):
        # data: N C V T M

        # load label
        self.label = np.load(self.label_path)

        # load data
        self.data = np.load(self.data_path)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        # get data
        data_numpy = self.data[index]
        label = self.label[index]



        return data_numpy, label
