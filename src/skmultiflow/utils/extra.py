from collections import deque
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd


class EqualErrorRate:
   
    def __init__(self, pred_window_size = 100000):
        self._window_size = pred_window_size
        self._y_true = deque(maxlen=pred_window_size)
        self._y_pred = deque(maxlen=pred_window_size)
        
    
    def add_predictions(self, y_true, y_prob_pred):
        
        if isinstance(y_true, (int, float, complex)):
            y_true = [y_true]
        if isinstance(y_true, (int, float, complex)):
            y_prob_pred = [y_prob_pred]
        
        self._y_true.extend(y_true)
        self._y_pred.extend(y_prob_pred)
    
    
    def compute_detailed_eer(self):
        #return list(self.y_true_data), list(self.y_pred_data)
        fpr, tpr, thresholds = metrics.roc_curve(np.array(self._y_true) , np.array(self._y_pred), pos_label=1, drop_intermediate=False)
        fnr = 1-tpr
        abs_diffs = np.abs(fpr - fnr)
        min_index = np.argmin(abs_diffs)
        eer = np.mean((fpr[min_index], fnr[min_index]))
        
        return eer, thresholds[min_index], fnr, fpr, thresholds
    
    def compute_eer(self):
        eer, _, _, _, _ = self.compute_detailed_eer()
        return eer
    
