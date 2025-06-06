import numpy as np


class FeatureEngineering:
    def __init__(self):
        self.max_garage_yr_built = None

    def transform_features(self, df):
        if self.max_garage_yr_built is None:
            raise EnvironmentError("Need to invoke learn_maximum() on the training data first!")

        # Keeping as many features arranged where: lower = worse for SalePrice, higher = better for SalePrice.
        df["BuildingNewnessScore"] = (df["YrSold"] - df["YearBuilt"]) * -1
        # Years remodelled since remodelled during sale.
        df["BuildingRemodNewnessScore"] = (df["YrSold"] - df["YearRemodAdd"]) * -1

        df = df.drop(["YearBuilt", "YearRemodAdd"], axis=1)

        # Logic: GarageYrBlt is -1 if no garage, we want a consistent structure where smaller = worse, larger = better.
        # We also want the GarageAgeAtSale instead. So first, for non -1 values, update them via YrSold - GarageYrBlt.
        # Then, set all -1 values as the max garage year +1, then have all multiplied by -1 to be managed by the
        # MinMaxScaler accordingly.
        mask_has_garage = (df["GarageYrBlt"] >= 0)
        df.loc[mask_has_garage, "GarageYrBlt"] = df.loc[mask_has_garage, "YrSold"] - df.loc[
            mask_has_garage, "GarageYrBlt"]

        df.loc[~mask_has_garage, "GarageYrBlt"] = self.max_garage_yr_built + 1
        df["GarageNewnessScore"] = df["GarageYrBlt"] * -1
        df = df.drop("GarageYrBlt", axis=1)

        df["MoSold_sin"] = np.sin(2 * np.pi * df["MoSold"] / 12)
        df["MoSold_cos"] = np.cos(2 * np.pi * df["MoSold"] / 12)
        df = df.drop("MoSold", axis=1, errors='ignore')

        df["HasPool"] = (df["PoolArea"] > 0).astype(int)
        df = df.drop("PoolArea", axis=1, errors='ignore')

        df["OverallQual_sq"] = np.power(df["OverallQual"], 2)

        # Inverting unemployment rate so higher is better
        df[['L1_A_UR', 'L1_A_UR_MA3', 'L1_A_UR_MA6', 'L1_I_UR', 'L1_I_UR_MA3', 'L1_I_UR_MA6']] *= -1
        return df

    def learn_maximum(self, df):
        if "GarageYrBlt" not in df.columns:
            raise ValueError("Expected column with GarageYrBuilt, this was not found!")
        self.max_garage_yr_built = max(df["GarageYrBlt"])
