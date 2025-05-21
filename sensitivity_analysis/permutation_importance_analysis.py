from sklearn.inspection import permutation_importance
import pandas as pd
from sklearn.base import BaseEstimator

class PermutationImportanceAnalysis:
    def __init__(self, model, random_state=42):
        self.model = model
        self.random_state = random_state
        self.feature_names_ = None
        self.importances_ = None

    def fit(self, X, y, scoring='neg_mean_squared_error', n_repeats=10):
        print("Calculating Permutation Importance...")
        result = permutation_importance(
            self.model,
            X,
            y,
            scoring=scoring,
            n_repeats=n_repeats,
            random_state=self.random_state,
            n_jobs=1
        )
        self.importances_ = result
        if hasattr(X, 'columns'):
            self.feature_names_ = X.columns
        return self

    def get_importances_df(self):
        if self.importances_ is None:
            raise RuntimeError("You must run fit() first!")

        importance_data = []
        # Reverse so the greater important features are first
        for i in self.importances_.importances_mean.argsort()[::-1]:
            name = self.feature_names_[i] if self.feature_names_ is not None else f"Feature {i}"
            importance_data.append({
                'feature': name,
                'importance_mean': self.importances_.importances_mean[i],
                'importance_std': self.importances_.importances_std[i]
            })
        return pd.DataFrame(importance_data)