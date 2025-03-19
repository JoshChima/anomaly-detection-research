from typing import Literal
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.base import OutlierMixin

class STDDetector(OutlierMixin):
    def __init__(self, k=3, central_tendency_type:Literal['mean', 'median']='mean'):
        self.k = k
        self.central_tendency_type_ = central_tendency_type
        self.mean_ = None
        self.median_ = None
        self.std_ = None
    
    def _set_central_tendency(self, X):
        if self.central_tendency_type_ == 'mean':
            self.mean_ = np.mean(X, axis=0)
        elif self.central_tendency_type_ == 'median':
            self.median_ = np.median(X, axis=0)
        else:
            raise ValueError('Invalid central_tendency_type')
    
    def _get_central_tendency(self):
        if self.central_tendency_type_ == 'mean':
            return self.mean_
        elif self.central_tendency_type_ == 'median':
            return self.median_
        else:
            raise ValueError('Invalid central_tendency_type')
    
    def fit(self, X, y=None):
        X = np.asarray(X)
        self._set_central_tendency(X)
        self.std_ = np.std(X, axis=0)
        return self
    
    def decision_function(self, X):
        """
        Calculates the z-scores for each sample in X
        """
        X = np.asarray(X)
        return (X - self._get_central_tendency()) / self.std_

    def predict(self, X):
        self.fit(X, y=None)
        X = np.asarray(X)
        z_scores = self.decision_function(X)
        abs_zscores = np.abs(z_scores)
        return np.where(abs_zscores > self.k, -1, 0)
    
    def fit_predict(self, X, y=None):
        return self.predict(X)