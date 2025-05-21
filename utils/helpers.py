import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
import numpy as np
import torch
from torch import nn

def _custom_feature_names(transformer_name, feature_name):
    if transformer_name == "remainder" and feature_name.startswith("ohe"):
        return feature_name
    else:
        return f"{transformer_name}__{feature_name}"

def generate_preprocessor(ordinal_cats_ordered, categorical_cols_ordinal, numerical_cols, sparse=False):

    return ColumnTransformer(
        transformers=[
            ('ord', OrdinalEncoder(categories=ordinal_cats_ordered), categorical_cols_ordinal),
            ('num', StandardScaler(), numerical_cols)
        ],
        remainder='passthrough',
        sparse_threshold=(0 if sparse else 1),
        verbose_feature_names_out=_custom_feature_names
    )

def get_log_minmax_cols():
    return ['LotArea', 'LotFrontage', '2ndFlrSF', 'MasVnrArea', '1stFlrSF', 'OpenPorchSF', 'TotalBsmtSF',
            'LowQualFinSF', 'GrLivArea', 'GarageArea', 'WoodDeckSF', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',
            '3SsnPorch', 'ScreenPorch', 'EnclosedPorch', 'MiscVal']

def get_minmax_cols():
    return ['OverallQual', 'OverallCond', 'YrSold', 'OverallQual_sq', 'BuildingNewnessScore',
            'BuildingRemodNewnessScore', 'GarageCars', 'Fireplaces', 'TotRmsAbvGrd', 'YrSold', 'L1_A_UR',
            'L1_A_UR_MA3', 'L1_A_UR_MA6', 'L1_I_HPI', 'L1_I_HPI_MA3', 'L1_I_HPI_MA6', 'L1_I_PR', 'L1_I_PR_MA3',
            'L1_I_PR_MA6', 'L1_I_UR', 'L1_I_UR_MA3', 'L1_I_UR_MA6', 'OverallQual_sq', 'GarageNewnessScore']
# Separated macroeconomic fields below, easier to distinguish.
# A_UR", "I_HPI", "I_PR", "I_UR


def get_categorical_cols_nominal():
    return ['MSSubClass', 'MSZoning', 'Street', 'Alley', 'LandContour',
            'Utilities', 'LotConfig', 'Neighborhood', 'Condition1', 'Condition2',
            'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
            'Exterior2nd', 'MasVnrType', 'Foundation', 'BsmtExposure', 'BsmtFinType1',
            'BsmtFinType2', 'Heating', 'CentralAir', 'Electrical', 'GarageType',
            'MiscFeature', 'SaleType', 'SaleCondition', 'BsmtFullBath', 'BsmtHalfBath',
            'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr']


def get_categorical_cols_ordinal():
    return ['LotShape', 'LandSlope', 'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual',
            'Functional', 'FireplaceQu', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence']


def get_ordinal_cats_ordered():
    return [
        ['IR3', 'IR2', 'IR1', 'Reg'],  # LotShape
        ['Sev', 'Mod', 'Gtl'],  # LandSlope
        ['Po', 'Fa', 'TA', 'Gd', 'Ex'],  # ExterQual
        ['Po', 'Fa', 'TA', 'Gd', 'Ex'],  # ExterCond
        ['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],  # BsmtQual
        ['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],  # BsmtCond
        ['Po', 'Fa', 'TA', 'Gd', 'Ex'],  # HeatingQC
        ['Po', 'Fa', 'TA', 'Gd', 'Ex'],  # KitchenQual
        ['Sal', 'Sev', 'Maj2', 'Maj1', 'Mod', 'Min2', 'Min1', 'Typ'],  # Functional
        ['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],  # FireplaceQu
        ['None', 'Unf', 'RFn', 'Fin'],  # GarageFinish
        ['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],  # GarageQual
        ['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],  # GarageCond
        ['N', 'P', 'Y'],  # PavedDrive
        ['None', 'Fa', 'TA', 'Gd', 'Ex'],  # PoolQC
        ['None', 'MnWw', 'GdWo', 'MnPrv', 'GdPrv']  # Fence
    ]


def init_fill_na(df):
    cols_fill_none_cat = ['Alley', 'BsmtQual', 'BsmtCond', 'BsmtExposure',
                          'BsmtFinType1', 'BsmtFinType2', 'FireplaceQu',
                          'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
                          'PoolQC', 'Fence', 'MiscFeature', 'MasVnrType']
    for col in cols_fill_none_cat:
        df[col] = df[col].fillna("None")
    df["MasVnrArea"] = df["MasVnrArea"].fillna(0)
    # After much deliberation here's the plan for GarageYrBlt:
    # I want it GarageAgeDuringSale, but there could be legit "0" values.. I also want it so
    # it's not just as low as possible except 0 is "good" with respect to the sale price.
    # So, GarageYrBlt will be -1, then filtered for >= 0, inversed, with 1 added, so the greater thev alue
    # the better, resolving the contradiction.
    df["GarageYrBlt"] = df["GarageYrBlt"].fillna(-1)
    return df


def learn_scaling_factors(df):
    required_cols = ["BldgType", "MSZoning", "LotShape", "LotFrontage", "SqrtLotArea"]

    if not all(col in df.columns for col in required_cols):
        # Check specifically which are missing for a better error message
        missing = [col for col in required_cols if col not in df.columns]
        raise ValueError(f"Input df missing required columns: {missing}")

    df_calc = df[required_cols].copy()
    df_calc = df_calc.dropna(subset=['LotFrontage', 'SqrtLotArea'])
    df_calc = df_calc[df_calc['SqrtLotArea'] > 0]

    if df_calc.empty:
        raise ValueError("No valid rows with non-missing LotFrontage and positive SqrtLotArea found.")

    df_calc['ScalingFactor'] = df_calc['LotFrontage'] / df_calc['SqrtLotArea']
    df_calc.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_calc.dropna(subset=['ScalingFactor'], inplace=True)

    if df_calc.empty:
        raise ValueError("Ratio calculation resulted in no valid values.")

    global_median_ratio = df_calc['ScalingFactor'].median()
    global_count = len(df_calc)
    print(f"Calculated Global Median Ratio: {global_median_ratio:.4f} (from {global_count} samples)")

    groups_to_calculate = {
        '3way': ['MSZoning', 'BldgType', 'LotShape'],
        '2way_ZS': ['MSZoning', 'LotShape'],
        '2way_ZB': ['MSZoning', 'BldgType'],
        '2way_BS': ['BldgType', 'LotShape'],
        '1way_Z': ['MSZoning'],
        '1way_B': ['BldgType'],
        '1way_S': ['LotShape']
    }

    grouped_rules_dict = {}
    for level_name, grouping_cols in groups_to_calculate.items():
        print(f"Calculating for group level: {level_name} ({grouping_cols})")
        grouped = df_calc.groupby(grouping_cols)['ScalingFactor']

        rules_for_level = grouped.agg(
            Median_Ratio="median",
            SampleCount="count"
        )
        grouped_rules_dict[level_name] = rules_for_level
        print(f" -> Found {len(rules_for_level)} groups for {level_name}")

    return grouped_rules_dict, global_median_ratio


def _fill_na_lotfrontage_helper(row, rules, min_samples, global_value):
    if row["LotFrontage"] is None:
        return row["LotFrontage"]
    sqrt_area = row["SqrtLotArea"]
    if pd.isna(sqrt_area) or sqrt_area <= 0:
        print(f"Warning: Invalid SqrtLotArea for row index {row.name}.")
        return None
    key3 = (row['MSZoning'], row['BldgType'], row['LotShape'])
    # Define potential 2-way keys (add others if needed)
    key2_zone_shape = (row['MSZoning'], row['LotShape'])
    key2_zone_bldg = (row['MSZoning'], row['BldgType'])
    key2_bldg_shape = (row['BldgType'], row['LotShape'])
    # Define potential 1-way keys
    key1_zone = (row['MSZoning'],)
    key1_shape = (row['LotShape'],)
    key1_bldg = (row['BldgType'],)

    # Rules

    if key3 in rules and rules.loc[key3, 'Count'] >= min_samples:
        return rules.loc[key3, 'ScaleFactor'] * sqrt_area

    possible_2way = [key2_zone_shape, key2_zone_bldg, key2_bldg_shape]
    best_2way_ratio = None
    max_2way_count = -1
    for key2 in possible_2way:
        if key2 in rules and rules.loc[key2, 'SampleCount'] >= min_samples:
            if rules[key2]['SampleCount'] > max_2way_count:
                max_2way_count = rules.loc[key2, 'SampleCount']
                best_2way_ratio = rules.loc[key2, 'ScaleFactor'] * sqrt_area
    if best_2way_ratio is not None:
        return best_2way_ratio

    # 3. Try 1-way keys (similar prioritization logic)
    possible_1way = [key1_zone, key1_bldg, key1_shape]
    best_1way_ratio = None
    max_1way_count = -1
    for key1 in possible_1way:
        if key1 in rules and rules.loc[key1, 'SampleCount'] >= min_samples:
            if rules.loc[key1, 'SampleCount'] > max_1way_count:
                max_1way_count = rules.loc[key1, 'SampleCount']
                best_1way_ratio = rules.loc[key1, 'ScaleFactor'] * sqrt_area
    if best_1way_ratio is not None:
        return best_1way_ratio

    # 4. Fallback to global
    return global_value * sqrt_area


def fill_na_lotfrontage(df_in, indexed_scales, threshold, global_value):
    df = df_in.copy()

    if 'SqrtLotArea' not in df.columns:
        if 'LotArea' in df.columns:
            df['SqrtLotArea'] = np.sqrt(df['LotArea']).fillna(0)  # Create if missing, handle NaNs
        else:
            raise ValueError("Cannot impute LotFrontage without LotArea/SqrtLotArea.")

    required_cols = ["BldgType", "MSZoning", "LotShape", "LotFrontage", "SqrtLotArea"]
    df = df[required_cols]
    return df.apply(_fill_na_lotfrontage_helper, args=(indexed_scales, threshold, global_value), axis=1)

def get_model_and_optim(in_features):
    model = nn.Sequential(
        nn.Linear(in_features, 188),
        nn.ELU(),
        nn.Linear(188, 47),
        nn.SiLU(),
        nn.Linear(47, 11),
        nn.SiLU(),
        nn.Linear(11, 1)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001171529841287367,
                                 betas=(0.9419019618038571, 0.9699586645558991),
                                 eps=1.2250425262180856e-08)
    return model, optimizer