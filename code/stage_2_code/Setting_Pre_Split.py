'''
Run against a pre-split dataset
'''

from code.base_class.setting import setting
import numpy as np

from code.stage_2_code.Split_Dataset_Loader import Split_Dataset_Loader

class Setting_Pre_Split(setting):
    fold = 3
    
    def load_run_save_evaluate(self):

        if not isinstance(self.dataset, Split_Dataset_Loader):
            raise ValueError("Split_Dataset_Loader must be used with Setting_Train_Test_Split")
        
        # load dataset
        loaded_data = self.dataset.load()

        # run MethodModule
        self.method.data = loaded_data
        learned_result = self.method.run()
            
        # save raw ResultModule
        self.result.data = learned_result
        self.result.save()
            
        self.evaluate.data = learned_result
        
        return self.evaluate.evaluate()

        