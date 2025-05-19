# Ended up moving this into its own section, due to the overall complexity of my project.
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

import helpers
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
import numpy as np

from feature_engineering import FeatureEngineering


class DataPreprocessor:

    def __init__(self,
                 train_path='data/train.csv',
                 test_path='data/test.csv',
                 iowa_house_price_index_path='data/iowa_house_price_index.csv',
                 ames_unemployment_rate_path='data/ames_unemployment_rate.csv',
                 iowa_labour_participation_index_path='data/iowa_labour_participation_rate.csv',
                 iowa_unemployment_rate_index_path='data/iowa_unemployment_rate.csv'):

        sklearn.set_config(transform_output="pandas")
        self.train_df = pd.read_csv(train_path)
        self.test_df = pd.read_csv(test_path)

        self.iowa_house_price_index = pd.read_csv(iowa_house_price_index_path)
        self.ames_unemployment_rate = pd.read_csv(ames_unemployment_rate_path)
        self.iowa_labour_participation_rate = pd.read_csv(iowa_labour_participation_index_path)
        self.iowa_unemployment_rate = pd.read_csv(iowa_unemployment_rate_index_path)

        # Declare None values, to allow None check in "preprocess_data" to work.
        self.X_train = None
        self.X_test = None
        self.Y_train = None
        self.Y_test = None
        self.imputation_params = {}

    # 1. Set indexes. Data will be converted to date after contacting for greater efficiency.
    def _index_data(self):
        self.iowa_house_price_index = self.iowa_house_price_index.set_index("observation_date")
        self.ames_unemployment_rate = self.ames_unemployment_rate.set_index("observation_date")
        self.iowa_labour_participation_rate = self.iowa_labour_participation_rate.set_index("observation_date")
        self.iowa_unemployment_rate = self.iowa_unemployment_rate.set_index("observation_date")

    # 2. Merge data immediately for greater efficiency, abbreviate column names.
    def _merge_data(self):
        self.merged_df = pd.concat([self.ames_unemployment_rate,
                                    self.iowa_house_price_index,
                                    self.iowa_labour_participation_rate,
                                    self.iowa_unemployment_rate], axis=1)

        self.merged_df.columns = ["A_UR", "I_HPI", "I_PR", "I_UR"]

    # 3. Declare index as date objects, forward-fill due to house price indexes being quarterly.
    def _align_date_data(self):
        self.merged_df.index = pd.to_datetime(self.merged_df.index)

        # Forward filling house price indexes, as they are quarterly. Resampling not needed, due to pd.concat.
        self.merged_df = self.merged_df.ffill()

    # 4. Include 3 and 6 month rolling averages, lag all values by 1 month.
    def _fill_and_shift(self):
        for col in self.merged_df.columns:
            self.merged_df[f'{col}_MA3'] = self.merged_df[col].rolling(window=3).mean()
            self.merged_df[f'{col}_MA6'] = self.merged_df[col].rolling(window=6).mean()

        self.merged_df = self.merged_df.shift(periods=1)

        # Add "L1" for lag 1.
        rename_map = {col: f"L1_{col}" for col in self.merged_df.columns}
        self.merged_df = self.merged_df.rename(columns=rename_map)

    # 6. Complete the merging
    def _merge_with_training_data(self):
        self.merged_df["MoSold"] = self.merged_df.index.month
        self.merged_df["YrSold"] = self.merged_df.index.year
        self.train_df = pd.merge(self.train_df, self.merged_df, how='left', on=['MoSold', 'YrSold'])

    # 7. Split test and training data
    def _split_data(self):
        target_col = "SalePrice"
        x_train = self.train_df.drop(columns=[target_col])
        y_train = np.log1p(self.train_df[target_col])

        self.X_train, self.X_test, self.Y_train, self.Y_test \
            = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

    # 9. Learn relevant missing electrical data based on train data only
    def _learn_missing_params(self):
        self.imputation_params["Electrical"] = self.X_train["Electrical"].mode()[0]
        indexed_scales, global_value = helpers.learn_scaling_factors(self.X_train)
        self.imputation_params["LotFrontageScaleFactors"] = indexed_scales
        self.imputation_params["LotFrontageGlobalScaleFactor"] = global_value

    # 10. Impute missing values based on learned parameters.
    def _impute_missing_params(self, threshold):
        mode_electrical = self.imputation_params["Electrical"]
        indexed_scales = self.imputation_params["LotFrontageScaleFactors"]
        global_value = self.imputation_params["LotFrontageGlobalScaleFactor"]
        self.X_train["Electrical"] = self.X_train["Electrical"].fillna(mode_electrical)
        self.X_test["Electrical"] = self.X_test["Electrical"].fillna(mode_electrical)
        self.X_train["LotFrontage"] = helpers.fill_na_lotfrontage(self.X_train, indexed_scales, threshold, global_value)
        self.X_test["LotFrontage"] = helpers.fill_na_lotfrontage(self.X_test, indexed_scales, threshold, global_value)

    def preprocess_data(self, lot_frontage_threshold, use_ohe=True):
        if self.X_train is not None or self.Y_train is not None or self.X_test is not None or self.Y_test is not None:
            return self.X_train, self.X_test, self.Y_train, self.Y_test
        self._index_data()
        self._merge_data()
        self._align_date_data()

        self._fill_and_shift()

        # 5. added this step here to be more explicit, it doesn't fit well anywhere else.
        self.train_df = helpers.init_fill_na(self.train_df)

        self._merge_with_training_data()

        self._split_data()

        # 8 Impute SqrtLotArea needed for helpers lot frontage & scaling factor methods.
        self.X_train["SqrtLotArea"] = np.sqrt(self.X_train["LotArea"])
        self.X_test["SqrtLotArea"] = np.sqrt(self.X_test["LotArea"])

        self._learn_missing_params()
        self._impute_missing_params(lot_frontage_threshold)

        # Removing SqrtLotArea as it's redundant due to LotArea being log transformed.
        self.X_train = self.X_train.drop(columns=["SqrtLotArea"], axis=1)
        self.X_test = self.X_test.drop(columns=["SqrtLotArea"], axis=1)

        if use_ohe:
            encoder = ColumnTransformer([
                ('ohe', OneHotEncoder(drop='first', sparse_output=False), helpers.get_categorical_cols_nominal())],
                remainder='passthrough', sparse_threshold=1, verbose_feature_names_out=_custom_feature_names)
            encoder.set_output(transform="pandas")

            encoder.fit(pd.concat([self.X_train, self.X_test], axis=0))
            self.X_train = encoder.transform(self.X_train)
            self.X_test = encoder.transform(self.X_test)
            self.X_train = self.X_train.drop("Id", axis=1)
            self.X_test = self.X_test.drop("Id", axis=1)
            feature_engineer = FeatureEngineering()
            feature_engineer.learn_maximum(self.X_train)
            self.X_train = feature_engineer.transform_features(self.X_train)
            self.X_test = feature_engineer.transform_features(self.X_test)
        return self.X_train, self.X_test, self.Y_train, self.Y_test


def _custom_feature_names(transformer_name, feature_name):
    if transformer_name != "ohe":
        return feature_name
    else:
        return f"{transformer_name}__{feature_name}"
