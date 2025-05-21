from sklearn.base import BaseEstimator, RegressorMixin
import torch
import numpy as np
import pandas as pd


class EstimatorWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.eval()  # Ensure model is in eval mode from the start

    def fit(self, X, y=None):
        """
        This method is for scikit-learn compatibility.
        The PyTorch model is assumed to be already trained.
        """
        # No actual fitting happens here as the PyTorch model is pre-trained
        return self

    def predict(self, X):
        """
        Makes predictions using the pre-trained PyTorch model.
        X is expected to be a pandas DataFrame or NumPy array.
        """
        self.model.eval()

        # Convert input X to tensor
        if isinstance(X, pd.DataFrame):
            X_np = X.values
        elif isinstance(X, np.ndarray):
            X_np = X
        else:
            raise TypeError("Input X to predict() must be a pandas DataFrame or a NumPy ndarray.")

        X_tensor = torch.tensor(X_np, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            predictions_tensor = self.model(X_tensor)

        # Ensure output is a 1D NumPy array for many scikit-learn scorers
        return predictions_tensor.cpu().numpy().ravel()
