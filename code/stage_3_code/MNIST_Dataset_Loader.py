# Load the MNIST dataset
from code.base_class.dataset import dataset
import pickle
import numpy as np
import pandas as pd

class MNIST_Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_file_name = None
    
    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def load(self):
        print('loading training data...')
        with open(self.dataset_source_folder_path + self.dataset_file_name, 'rb') as f:
            self.data = pickle.load(f)

        train_X = [instance["image"] for instance in self.data["train"]]
        train_y = [instance["label"] for instance in self.data["train"]]

        test_X = [instance["image"] for instance in self.data["test"]]
        test_y = [instance["label"] for instance in self.data["test"]]
        
        return { 
            'train': {
                "y": train_y,
                "X": train_X
            },
            'test': {
                "y": test_y,
                "X": test_X
            }
        }