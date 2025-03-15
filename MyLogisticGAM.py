import numpy as np
from pygam import LogisticGAM
class MyLogisticGAM(LogisticGAM):
    def predict_proba(self, X):
        prob_class1 = super().predict_proba(X)  # Get probability of class 1
        prob_class0 = 1 - prob_class1  # Compute probability of class 0
        return np.column_stack((prob_class0, prob_class1))  # Return 2D array