'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from locale import normalize
from typing import Callable

from code.base_class.dataset import dataset
import pandas as pd


# Load a dataset with a user-defined training/testing split
class Split_Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    train_file_name = None
    test_file_name = None
    feature_selector: Callable = None
    normalize:bool = False
    
    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)
    
    # Z-score normalization
    # Normalize each column to contain values with mean 0 and std of 1
    # https://towardsdatascience.com/data-normalization-with-pandas-and-scikit-learn-7c1cc6ed6475
    def z_score(self, df: pd.DataFrame):
        df_std = df.copy()
        for column in df_std.columns:
            df_std[column] = (df_std[column] - df_std[column].mean()) / df_std[column].std()
        
        # Copy back the label column, which should not be normalized
        df_std["label"] = df["label"]
        return df_std

    def load(self):
        print('loading training data...')
        train = pd.read_csv(self.dataset_source_folder_path + self.train_file_name)
        test = pd.read_csv(self.dataset_source_folder_path + self.test_file_name)

        # Pick out features using the provided feature selector
        feature_columns = self.feature_selector(train)

        # Apply normalization
        if self.normalize:
            print("Applying Z-Score normalization")
            train = self.z_score(train)
            test = self.z_score(test)
        
        print(f'selected {len(feature_columns)}/{len(train.columns)} features')

        # Filter training and test datasets to the selected features
        train = train[pd.Index(["label"]).append(feature_columns)]
        test = test[pd.Index(["label"]).append(feature_columns)]

        return { 
            'train': {
                "y": train.iloc[:, 0],
                "X": train.iloc[:, 1:]
            },
            'test': {
                "y": test.iloc[:, 0],
                "X": test.iloc[:, 1:]
            }
        }