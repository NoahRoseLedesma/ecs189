
'''
Evaluate a classifier using accuracy, precision, recall, and F1 score
'''

from code.base_class.evaluate import evaluate
from sklearn.metrics import classification_report


class Evaluate_Classifier(evaluate):
    data = None
    
    def evaluate(self):
        print('evaluating performance...')
        return classification_report(self.data['true_y'], self.data['pred_y'], zero_division=1)
    
    def evaluate_to_dict(self) -> dict:
        return classification_report(self.data['true_y'], self.data['pred_y'], zero_division=1, output_dict=True)
        