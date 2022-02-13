# Load the MNIST dataset
from code.base_class.dataset import dataset
import pickle
import numpy as np
from scipy.ndimage.interpolation import rotate
from scipy.ndimage import shift
from tqdm import tqdm
import os
import torchvision
import torch

class CIFAR_Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_file_name = None
    
    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    # Create augmentations of training images to increase the training dataset size.
    # This method was used for the ORL dataset.
    def create_augmentations(self, image):
        # Create a list of augmented images
        augments = [image]

        # Rotations
        # Randomly decide to rotate the image with probability 0.3
        if np.random.rand() < 0.3:
            # Rotate the image by a random angle between -20 and 20 degrees
            angle = np.random.randint(-20, 20)
            augments.append(rotate(image, angle, reshape=False))

        # Randomly decide to flip the image horizontally with probability 0.3
        if np.random.rand() < 0.3:
            augments.append(np.fliplr(image).copy())

        # Randomly decide to shift the image with probability 0.3
        if np.random.rand() < 0.3:
            # Randomly decide to shift the image by a random amount between -5 and 5 pixels
            shift_x = np.random.randint(-5, 5)
            shift_y = np.random.randint(-5, 5)
            augments.append(shift(image, (shift_x, shift_y, 0)))

        return augments
    def load(self):
        print('loading training data...')

        # Check if the augmented data pickle file exists
        if not os.path.isfile(self.dataset_source_folder_path + 'CIFAR_augmented.npy'):
            # Create the augmented data
            with open(self.dataset_source_folder_path + self.dataset_file_name, 'rb') as f:
                self.data = pickle.load(f)

            train_X = np.array([instance["image"] for instance in self.data["train"]])
            train_y = np.array([instance["label"] for instance in self.data["train"]])

            test_X = np.array([instance["image"] for instance in self.data["test"]])
            test_y = np.array([instance["label"] for instance in self.data["test"]])

            # Create augmentations
            print('creating augmentations...')
            augmentations_X = []
            augmentations_y = []
            for image, label in tqdm(zip(train_X, train_y)):
                new_augments = self.create_augmentations(image)
                augmentations_X += new_augments
                augmentations_y += [label] * len(new_augments)
            
            # Print the number of augmented images
            print('number of augmented images:', len(augmentations_X))

            # Reshape the data
            transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
            augmentations_X = np.array([transforms(image).numpy() for image in augmentations_X])
            test_X = np.array([transforms(image).numpy() for image in test_X])

            # Save the augmented data
            print("saving augmented data...")
            with open(self.dataset_source_folder_path + 'CIFAR_augmented.npy', 'wb') as f:
                np.save(f, augmentations_X, allow_pickle=False)
                np.save(f, augmentations_y, allow_pickle=False)
                np.save(f, test_X, allow_pickle=False)
                np.save(f, test_y, allow_pickle=False)
        else:
            # Load the augmented data
            with open(self.dataset_source_folder_path + 'CIFAR_augmented.npy', 'rb') as f:
                augmentations_X = np.load(f, allow_pickle=False)
                augmentations_y = np.load(f, allow_pickle=False)
                test_X = np.load(f, allow_pickle=False)
                test_y = np.load(f, allow_pickle=False)
        
            print("Augmented training data size: ", len(augmentations_X))

        return { 
            'train': {
                "y": augmentations_y,
                "X": augmentations_X
            },
            'test': {
                "y": test_y,
                "X": test_X
            }
        }