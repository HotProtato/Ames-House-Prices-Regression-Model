# Ended up moving this into its own section, due to the overall complexity of my project.
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from utils import helpers
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
import numpy as np

from data_management import feature_engineering


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
        self.merged_df = self.merged_df.ffill().bfill()

    # 4. Include 3 and 6 month rolling averages, lag all values by 1 month.
    def _fill_and_shift(self):
        for col in self.merged_df.columns:
            self.merged_df[f'{col}_MA3'] = self.merged_df[col].rolling(window=3).mean()
            self.merged_df[f'{col}_MA6'] = self.merged_df[col].rolling(window=6).mean()

        self.merged_df = self.merged_df.shift(periods=1)

        # Add "L1" for lag 1.
        rename_map = {col: f"L1_{col}" for col in self.merged_df.columns}
        self.merged_df = self.merged_df.rename(columns=rename_map)
        self.merged_df["MoSold"] = self.merged_df.index.month
        self.merged_df["YrSold"] = self.merged_df.index.year

    # 6. Complete the merging
    def _merge_econ_with_df(self, df):
        df = pd.merge(df, self.merged_df, how='left', on=['MoSold', 'YrSold'])
        return df

    # 7. Split test and training data
    def _split_data(self):
        target_col = "SalePrice"
        x_train = self.train_df.drop(columns=[target_col])
        y_train = np.log1p(self.train_df[target_col])
        y_bins = pd.qcut(y_train, q=10, labels=False)

        #self.X_train, self.X_test, self.Y_train, self.Y_test \
        #    = train_test_split(x_train, y_train, test_size=0.05, random_state=42, stratify=y_bins)

        self.X_train, self.X_test, self.Y_train, self.Y_test \
            = train_test_split(x_train, y_train, test_size=0.05, random_state=42, stratify=y_bins)

    # 9. Learn relevant missing electrical data based on train data only
    def _learn_missing_params(self):
        self.imputation_params["Electrical"] = self.X_train["Electrical"].mode()[0]
        indexed_scales, global_value = helpers.learn_scaling_factors(self.X_train)
        self.imputation_params["LotFrontageScaleFactors"] = indexed_scales
        self.imputation_params["LotFrontageGlobalScaleFactor"] = global_value
        self.imputation_params["KitchenQual"] = self.X_train["KitchenQual"].mode()[0]

    # 10. Impute missing values based on learned parameters.
    def _impute_missing_params(self, threshold):
        kitchen_qual = self.imputation_params["KitchenQual"]
        mode_electrical = self.imputation_params["Electrical"]
        indexed_scales = self.imputation_params["LotFrontageScaleFactors"]
        global_value = self.imputation_params["LotFrontageGlobalScaleFactor"]
        self.X_train["Electrical"] = self.X_train["Electrical"].fillna(mode_electrical)
        self.X_test["Electrical"] = self.X_test["Electrical"].fillna(mode_electrical)
        self.test_df["Electrical"] = self.test_df["Electrical"].fillna(mode_electrical)
        self.test_df["KitchenQual"] = self.test_df["KitchenQual"].fillna(kitchen_qual)
        self.X_train["LotFrontage"] = helpers.fill_na_lotfrontage(self.X_train, indexed_scales, threshold, global_value)
        self.X_test["LotFrontage"] = helpers.fill_na_lotfrontage(self.X_test, indexed_scales, threshold, global_value)
        self.test_df["LotFrontage"] = helpers.fill_na_lotfrontage(self.test_df, indexed_scales, threshold, global_value)

    def preprocess_data(self, lot_frontage_threshold, use_ohe=True):
        if self.X_train is not None or self.Y_train is not None or self.X_test is not None or self.Y_test is not None:
            return self.X_train, self.X_test, self.Y_train, self.Y_test
        self._index_data()
        self._merge_data()
        self._align_date_data()

        self._fill_and_shift()

        # 5. added this step here to be more explicit, it doesn't fit well anywhere else.
        self.train_df = helpers.init_fill_na(self.train_df)
        self.test_df = helpers.init_fill_na(self.test_df)

        self.train_df = self._merge_econ_with_df(self.train_df)
        self.test_df = self._merge_econ_with_df(self.test_df)

        self._split_data()

        # Given "150" is not in train_df, and there's exactly 1 in test_df, but it's a townhouse,
        # resolving this to 160 as the closest match.
        self.test_df.loc[self.test_df["MSSubClass"] == 150, "MSSubClass"] = 160

        # 8 Impute SqrtLotArea needed for helpers lot frontage & scaling factor methods.
        self.X_train["SqrtLotArea"] = np.sqrt(self.X_train["LotArea"])
        self.X_test["SqrtLotArea"] = np.sqrt(self.X_test["LotArea"])
        self.test_df["SqrtLotArea"] = np.sqrt(self.test_df["LotArea"])

        self._learn_missing_params()
        self._impute_missing_params(lot_frontage_threshold)

        # Removing SqrtLotArea as it's redundant due to LotArea being log transformed.
        self.X_train = self.X_train.drop(columns=["SqrtLotArea"], axis=1)
        self.X_test = self.X_test.drop(columns=["SqrtLotArea"], axis=1)
        self.test_df = self.test_df.drop(columns=["SqrtLotArea"], axis=1)

        if 'FullBath' in self.test_df.columns and 'FullBath' in self.X_train.columns:
            # 1. Ensure both columns are numeric before comparing
            train_full_bath = pd.to_numeric(self.X_train['FullBath'], errors='coerce').fillna(0)
            test_full_bath = pd.to_numeric(self.test_df['FullBath'], errors='coerce').fillna(0)

            # 2. Get the max value from the training data
            max_full_bath_train = int(train_full_bath.max())

            # 3. Cap the test data at the max value
            test_full_bath.loc[test_full_bath > max_full_bath_train] = max_full_bath_train

            # 4. Assign the cleaned, integer-converted data back to the dataframe
            self.test_df['FullBath'] = test_full_bath.astype(int)

        # Apply the same robust logic for BsmtFullBath
        if 'BsmtFullBath' in self.test_df.columns and 'BsmtFullBath' in self.X_train.columns:
            train_bsmt_full = pd.to_numeric(self.X_train['BsmtFullBath'], errors='coerce').fillna(0)
            test_bsmt_full = pd.to_numeric(self.test_df['BsmtFullBath'], errors='coerce').fillna(0)

            max_bsmt_full_bath_train = int(train_bsmt_full.max())

            test_bsmt_full.loc[test_bsmt_full > max_bsmt_full_bath_train] = max_bsmt_full_bath_train

            self.test_df['BsmtFullBath'] = test_bsmt_full.astype(int)

        # And for BsmtHalfBath
        if 'BsmtHalfBath' in self.test_df.columns and 'BsmtHalfBath' in self.X_train.columns:
            train_bsmt_half = pd.to_numeric(self.X_train['BsmtHalfBath'], errors='coerce').fillna(0)
            test_bsmt_half = pd.to_numeric(self.test_df['BsmtHalfBath'], errors='coerce').fillna(0)

            max_bsmt_half_bath_train = int(train_bsmt_half.max())

            test_bsmt_half.loc[test_bsmt_half > max_bsmt_half_bath_train] = max_bsmt_half_bath_train

            self.test_df['BsmtHalfBath'] = test_bsmt_half.astype(int)

        self.X_train = self.X_train.drop(columns=["Utilities"], axis=1)
        self.X_test = self.X_test.drop(columns=["Utilities"], axis=1)
        self.test_df = self.test_df.drop(columns=["Utilities"], axis=1)

        if use_ohe:
            encoder = ColumnTransformer([
                ('ohe', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), helpers.get_categorical_cols_nominal())],
                remainder='passthrough', sparse_threshold=1, verbose_feature_names_out=_custom_feature_names)
            encoder.set_output(transform="pandas")

            encoder.fit(pd.concat([self.X_train, self.X_test], axis=0))
            self.X_train = encoder.transform(self.X_train)
            self.X_test = encoder.transform(self.X_test)
            self.test_df = encoder.transform(self.test_df)
            self.X_train = self.X_train.drop("Id", axis=1)
            self.X_test = self.X_test.drop("Id", axis=1)
            feature_engineer = feature_engineering.FeatureEngineering()
            feature_engineer.learn_maximum(self.X_train)
            self.X_train = feature_engineer.transform_features(self.X_train)
            self.X_test = feature_engineer.transform_features(self.X_test)
            self.test_df = feature_engineer.transform_features(self.test_df)

            # These features sum to <= 1, and results improved with their removal.

            to_remove = ['ohe__Neighborhood_Blueste', 'ohe__Condition1_RRNe',
                         'ohe__Condition2_PosA', 'ohe__Condition2_RRAe', 'ohe__Condition2_RRAn', 'ohe__Condition2_RRNn',
                         'ohe__RoofMatl_Membran', 'ohe__RoofMatl_Metal', 'ohe__RoofMatl_Roll',
                         'ohe__Exterior1st_AsphShn', 'ohe__Exterior1st_CBlock', 'ohe__Exterior1st_ImStucc',
                         'ohe__Exterior1st_Stone', 'ohe__Exterior2nd_CBlock', 'ohe__Exterior2nd_Other',
                         'ohe__Electrical_Mix',
                         'ohe__MiscFeature_TenC']  # Removed based on <= 1 total non-zero appearances

            #self.X_train = self.X_train.drop(to_remove, axis=1)
            #self.X_test = self.X_test.drop(to_remove, axis=1)
            #self.test_df = self.test_df.drop(to_remove, axis=1)
        return self.X_train, self.X_test, self.Y_train, self.Y_test, self.test_df


def _custom_feature_names(transformer_name, feature_name):
    if transformer_name != "ohe":
        return feature_name
    else:
        return f"{transformer_name}__{feature_name}"