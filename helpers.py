import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
import numpy as np


def generate_preprocessor(drop, sparse_output,
                          ordinal_cats_ordered, categorical_cols_ordinal, numerical_cols,
                          categorical_cols_nominal):
    # Many parameters, as this is used during both EDA and production.

    return ColumnTransformer(
        transformers=[
            ('ord', OrdinalEncoder(categories=ordinal_cats_ordered), categorical_cols_ordinal),
            ('num', StandardScaler(), numerical_cols),
            ('ohe', OneHotEncoder(drop=drop,
                                  handle_unknown='ignore',
                                  sparse_output=sparse_output),  # Set sparse=False for now
             categorical_cols_nominal)
        ],
        remainder='passthrough'  # Keep any columns not specified, or use 'drop'
    )


def get_numeric_cols():
    return ['LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt',
            'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',
            'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea',
            'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr',
            'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars',
            'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',
            'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold']
# TO CONSIDER: I could remove LotFrontage and just use the LotFrontageRatio.


def get_categorical_cols_nominal():
    return ['MSSubClass', 'MSZoning', 'Street', 'Alley', 'LandContour',
            'Utilities', 'LotConfig', 'Neighborhood', 'Condition1', 'Condition2',
            'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
            'Exterior2nd', 'MasVnrType', 'Foundation', 'BsmtExposure', 'BsmtFinType1',
            'BsmtFinType2', 'Heating', 'CentralAir', 'Electrical', 'GarageType',
            'MiscFeature', 'SaleType', 'SaleCondition']


def get_categorical_cols_ordinal():
    return ['LotShape', 'LandSlope', 'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual',
            'Functional', 'FireplaceQu', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence']


def get_ordinal_cats_ordered():
    ordinal_cats_ordered = [
        ['Reg', 'IR1', 'IR2', 'IR3'],  # LotShape
        ['Gtl', 'Mod', 'Sev'],  # LandSlope
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
    return ordinal_cats_ordered


def init_fill_na(df):
    cols_fill_none_cat = ['Alley', 'BsmtQual', 'BsmtCond', 'BsmtExposure',
                          'BsmtFinType1', 'BsmtFinType2', 'FireplaceQu',
                          'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
                          'PoolQC', 'Fence', 'MiscFeature', 'MasVnrType']
    for col in cols_fill_none_cat:
        df[col] = df[col].fillna("None")
    for col in ["MasVnrArea", "GarageYrBlt"]:
        df[col] = df[col].fillna(0)
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
        # Use .agg() to get both median and count cleanly
        rules_for_level = grouped.agg(
            Median_Ratio="median",
            SampleCount="count"
        )
        grouped_rules_dict[level_name] = rules_for_level
        print(f" -> Found {len(rules_for_level)} groups for {level_name}")

    # --- Return the package ---
    print("Finished learning rules.")
    # Return dict of DataFrames (your 'indexed_scales') and the global median separately
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
            df['SqrtLotArea'] = np.sqrt(df['LotArea']).fillna(0) # Create if missing, handle NaNs
        else:
            raise ValueError("Cannot impute LotFrontage without LotArea/SqrtLotArea.")

    required_cols = ["BldgType", "MSZoning", "LotShape", "LotFrontage", "SqrtLotArea"]
    df = df[required_cols]
    return df.apply(_fill_na_lotfrontage_helper, args=(indexed_scales, threshold, global_value), axis=1)
