# Before any ML code runs we should enable multiprocessing
import json
import numpy as np
import pandas as pd
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
import torch

from code.stage_1_code.Result_Saver import Result_Saver
from code.stage_2_code.Evaluate_Classifier import Evaluate_Classifier
from code.stage_2_code.Method_MLP import Method_MLP
from code.stage_2_code.Setting_Pre_Split import Setting_Pre_Split
from code.stage_2_code.Split_Dataset_Loader import Split_Dataset_Loader

# We pick N features with the highest variance (low variance features are dropped)
NUM_FEATURES = 600
NUM_CATEGORIES = 10

# Return a list of the top N features from the dataset
def select_features(dataset: pd.DataFrame):
    targets = dataset.iloc[:, 1]
    features = dataset.iloc[:, 2:]

    selector = SelectKBest(chi2, k=NUM_FEATURES)
    selector.fit_transform(features, targets)

    return dataset.columns[selector.get_support(indices=True)]

#---- parameter section -------------------------------
np.random.seed(2)
torch.manual_seed(2)
#------------------------------------------------------

# ---- objection initialization section ---------------
data_obj = Split_Dataset_Loader('stage-2-dataset', '')
data_obj.dataset_source_folder_path = 'data/stage_2_data/'
data_obj.train_file_name = 'train.csv'
data_obj.test_file_name = 'test.csv'
data_obj.feature_selector = select_features
data_obj.normalize = False

method_obj = Method_MLP(NUM_FEATURES, NUM_CATEGORIES, 'multi-layer perceptron', '')

result_obj = Result_Saver('saver', '')
result_obj.result_destination_folder_path = 'result/stage_2_result/MLP_'
result_obj.result_destination_file_name = 'prediction_result'

setting_obj = Setting_Pre_Split('pre split', '')

evaluate_obj = Evaluate_Classifier('accuracy, precision, recall, f1 score', '')
# ------------------------------------------------------

# ---- running section ---------------------------------
print('************ Start ************')
setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
setting_obj.print_setup_summary()
classification_report = setting_obj.load_run_save_evaluate()
print('************ Overall Performance ************')
print(classification_report)
print('************ Finish ************')

# Write recorded training progress
with open('data/stage_2_data/training_report.json', 'w') as f:
    json.dump(method_obj.reports, f)