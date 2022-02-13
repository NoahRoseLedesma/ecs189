# Load the MNIST dataset
from ast import arguments
from code.base_class.dataset import dataset
import pickle
import numpy as np
import pandas as pd
from scipy.ndimage.interpolation import rotate
from scipy.ndimage import shift
from tqdm import tqdm
import os

class ORL_Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_file_name = None
    
    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    # Create augmentations of training images to increase the training dataset size as suggested by Lu et. al.
    # https://doi.org/10.1080/21642583.2020.1836526
    def create_augmentations(self, image):
        # Create a list of augmented images
        augments = []

        # Rotations
        for i in range(-20, 20, 5):
            augments.append(rotate(image, angle=i, reshape=False))

        # Mirrored
        augments.append(np.fliplr(image))

        # Shifting
        for x in range(-5, 5, 5):
            for y in range(-5, 5, 5):
                augments.append(shift(image, (x, y)))

        return augments

    def load(self):
        print('loading training data...')
        # Check if the augmented data pickle file exists
        if not os.path.isfile(self.dataset_source_folder_path + 'ORL_augmented.pkl'):
            print('creating augmented data...')
            # Create the augmented data
            with open(self.dataset_source_folder_path + self.dataset_file_name, 'rb') as f:
                self.data = pickle.load(f)

            # This dataset has RGB images with repeated values for each pixel.
            # We need to convert it to grayscale.
            train_X = [np.dot(instance["image"], [1, 0, 0]) for instance in self.data["train"]]
            train_y = [instance["label"] for instance in self.data["train"]]

            test_X = [np.dot(instance["image"], [1, 0, 0]) for instance in self.data["test"]]
            test_y = [instance["label"] for instance in self.data["test"]]

            # Generate augmentations
            augmentations_X = []
            augmentations_y = []
            for image, label in tqdm(zip(train_X, train_y)):
                new_augments = self.create_augmentations(image)
                augmentations_X += new_augments
                augmentations_y += [label] * len(new_augments)

            # Shuffle the data
            indices = np.arange(len(augmentations_X))
            np.random.shuffle(indices)
            augmentations_X = np.array(augmentations_X)[indices]
            augmentations_y = np.array(augmentations_y)[indices]
            
            data_obj = {
                'train': {
                    "y": augmentations_y,
                    "X": augmentations_X
                },
                'test': {
                    "y": test_y,
                    "X": test_X
                }
            }

            # Save the augmented data
            with open(self.dataset_source_folder_path + 'ORL_augmented.pkl', 'wb') as f:
                pickle.dump(data_obj, f)
        else:
            # Load the augmented data
            print('loading augmented data...')
            with open(self.dataset_source_folder_path + 'ORL_augmented.pkl', 'rb') as f:
                data_obj = pickle.load(f)
        
        print("Augmented training data size: ", len(data_obj["train"]["X"]))

        return data_obj