import numpy as np
import pandas as pd


class FeatureEngineering:
    def __init__(self):
        """
        Initializes the FeatureEngineering class.
        Learned parameters will be stored as attributes.
        """
        self.max_actual_garage_age_ = None  # Learned from training data

    def _transform_age_features(self, df_input, to_transform_list):
        """
        Creates age-related features based on a sale year.
        Internal helper method.
        Args:
            df_input (pd.DataFrame): The DataFrame to transform.
            to_transform_list (list): A list of lists, where each inner list is
                                      ["OriginalYearColumnName", "NewAgeColumnName"].
        Returns:
            pd.DataFrame: The DataFrame with new age features added.
        """
        df = df_input.copy()

        if "YrSold" not in df.columns:
            raise ValueError("DataFrame must contain a 'YrSold' column for age calculations.")

        for original_year_col, new_age_col_name in to_transform_list:
            if original_year_col not in df.columns:
                print(f"Warning: Column '{original_year_col}' not found in DataFrame. Skipping '{new_age_col_name}'.")
                continue
            # Ensure columns are numeric before subtraction
            if pd.api.types.is_numeric_dtype(df[original_year_col]) and pd.api.types.is_numeric_dtype(df["YrSold"]):
                df[new_age_col_name] = df["YrSold"] - df[original_year_col]
            else:
                print(
                    f"Warning: Column '{original_year_col}' or 'YrSold' is not numeric. Skipping age calculation for '{new_age_col_name}'.")
                df[new_age_col_name] = np.nan  # Or handle as appropriate
        return df

    def fit(self, X, y=None):
        """
        Learns parameters needed for transformation from the training data X.
        In this case, it learns the maximum actual garage age.
        Args:
            X (pd.DataFrame): Training data.
            y (pd.Series, optional): Target variable. Not used in this transformer.
        Returns:
            self: The fitted transformer instance.
        """
        print("Fitting FeatureEngineering: Learning max_actual_garage_age_...")
        df_fit = X.copy()

        if 'GarageYrBlt' in df_fit.columns and 'YrSold' in df_fit.columns:
            has_garage_mask_fit = (df_fit['GarageYrBlt'] != -1) & (df_fit['GarageYrBlt'].notna())

            # Calculate temporary actual garage age
            temp_actual_garage_age = pd.Series(np.nan, index=df_fit.index)
            if has_garage_mask_fit.any():  # Ensure there are actual garages
                # Impute GarageYrBlt with YearBuilt if it's NaN but a garage exists (optional, good practice)
                # This should ideally be done in DataPreprocessor before this fit method
                # For simplicity, assuming GarageYrBlt is populated or correctly indicates -1
                temp_actual_garage_age.loc[has_garage_mask_fit] = df_fit.loc[has_garage_mask_fit, 'YrSold'] - \
                                                                  df_fit.loc[has_garage_mask_fit, 'GarageYrBlt']

            if temp_actual_garage_age.loc[has_garage_mask_fit].notna().any():
                self.max_actual_garage_age_ = temp_actual_garage_age.loc[has_garage_mask_fit].max()
            else:
                self.max_actual_garage_age_ = 0
        else:
            print("Warning: 'YrSold' or 'GarageYrBlt' not found in X during fit. Setting max_actual_garage_age_ to 0.")
            self.max_actual_garage_age_ = 0

        print(f"Learned max_actual_garage_age_: {self.max_actual_garage_age_}")
        return self

    def transform(self, X_input):
        """
        Applies all feature engineering transformations.
        Uses learned parameters (like max_actual_garage_age_).
        Args:
            X_input (pd.DataFrame): Data to transform (train, validation, or test).
        Returns:
            pd.DataFrame: Transformed DataFrame.
        """
        if 'GarageYrBlt' in X_input.columns and self.max_actual_garage_age_ is None:
            # This condition means fit() was not called, or it failed to learn the parameter.
            # For test/validation, we should not recalculate. Raise an error or use a default.
            raise RuntimeError(
                "FeatureEngineering.transform() called before fit() or max_actual_garage_age_ was not learned. Fit the transformer on training data first.")

        df = X_input.copy()

        # OverallQual squared
        if "OverallQual" in df.columns:
            df["OverallQual_sq"] = np.power(df["OverallQual"], 2)

        # PoolArea to HasPool
        if "PoolArea" in df.columns:
            df["HasPool"] = (df["PoolArea"] > 0).astype(int)
            df = df.drop("PoolArea", axis=1, errors='ignore')

        # MoSold cyclical features
        if "MoSold" in df.columns:
            df["MoSold_sin"] = np.sin(2 * np.pi * df["MoSold"] / 12)
            df["MoSold_cos"] = np.cos(2 * np.pi * df["MoSold"] / 12)
            # Consider dropping original 'MoSold' if these are the definitive replacements
            # df.drop("MoSold", axis=1, inplace=True, errors='ignore')

        # Age features (BuildingAge, RemodelAge)
        if "YrSold" in df.columns:
            age_specs = []
            if "YearBuilt" in df.columns:
                age_specs.append(["YearBuilt", "BuildingAgeAtSale"])
            if "YearRemodAdd" in df.columns:
                age_specs.append(["YearRemodAdd", "YearsSinceRemodelAtSale"])
            if age_specs:
                df = self._transform_age_features(df, age_specs)

        # Garage Ordinal Feature
        if 'GarageYrBlt' in df.columns and 'YrSold' in df.columns:
            has_garage_mask = (df['GarageYrBlt'] != -1) & (df['GarageYrBlt'].notna())

            temp_actual_garage_age = pd.Series(np.nan, index=df.index)
            if has_garage_mask.any():
                temp_actual_garage_age.loc[has_garage_mask] = df.loc[has_garage_mask, 'YrSold'] - \
                                                              df.loc[has_garage_mask, 'GarageYrBlt']

            df['GarageAgeAtSale'] = 0  # Default for "No Garage"

            # Use the max_age_to_use (learned from training data)
            # Ensure max_actual_garage_age_ is not None (i.e., fit has been called)
            max_age_to_use = self.max_actual_garage_age_ if self.max_actual_garage_age_ is not None else 0

            if has_garage_mask.any():  # Only apply to rows that have a garage
                df.loc[has_garage_mask, 'GarageAgeAtSale'] = \
                    (max_age_to_use - temp_actual_garage_age.loc[has_garage_mask]) + 1

            # df.drop(columns=['ActualGarageAge_temp'], inplace=True, errors='ignore') # Not needed as temp_actual_garage_age is local

        return df

    def fit_transform(self, X, y=None):
        """
        Convenience method to fit and then transform.
        """
        self.fit(X, y)
        return self.transform(X)
