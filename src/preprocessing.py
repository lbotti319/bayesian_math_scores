"""
Handle preprocessing of the data, such as converting string columns to numeric
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def prepare_data(df):
    """
    Splits out the target variable from the rest
    Converts categorical values to numeric with onehote encoding
    """
    df = df.copy()
    
    y = df['G3']
    df.drop('G3', axis=1, inplace=True)
    
    cat_cols = df.dtypes[df.dtypes=='object'].index
    int_cols = df.dtypes[df.dtypes=='int64'].index

#     scaler = StandardScaler()
#     X = pd.DataFrame(scaler.fit_transform(df[int_cols]), columns = int_cols)
    X = df[int_cols].copy()
    X['intercept'] = 1

    # drop='if_binary' ensures only one dummy column for binary variables
    transform = OneHotEncoder(drop='if_binary')
    encoded_cats = transform.fit_transform(df[cat_cols])
    encoded_cat_features = transform.get_feature_names(cat_cols)

    df_enc = pd.DataFrame.sparse.from_spmatrix(encoded_cats, columns = encoded_cat_features)
    df_enc[encoded_cat_features] = df_enc[encoded_cat_features].sparse.to_dense()
    
    joined = X.join(df_enc)
    final_features = joined[['intercept', 'age', 'sex_M', 'failures', 'higher_yes', 'Medu', 'absences', 'G2']]

    return final_features, y


def prepare_data_no_standardizing(df):
    """
    Splits out the target variable from the rest
    Converts categorical values to numeric with onehote encoding
    """
    y = df['G3']
    df.drop('G3', axis=1, inplace=True)

    X = pd.get_dummies(df, drop_first=True)
    X['intercept'] = 1

    return X, y


def MAR_data_deletion(df, percent_missing_feature1, percent_missing_feature2, missing_feature1, missing_feature2):
    """
    missing_feature1 is higher_yes
    missing_feature2 is either G2 or absences

    Example for missing features higher_yes and G2/absences:
    Higher_yes: Younger than 18, the less likely they are to know if they'll pursue higher ed
    G2/absences: The older they are (18 - 22),
    the more likely to get permission to skip mid semester grades
    The older the are (18 - 22), the more likely they are to skip/be absent from class
    """
    df = df.copy()

    underage_ind = np.where(df["age"] < 18)[0]
    overage_ind = np.where(df["age"] >= 18)[0]

    n = df.shape[0]

    nanidx_missing_feature1 = np.random.choice(underage_ind, int(n * percent_missing_feature1))
    nanidx_missing_feature2 = np.random.choice(overage_ind, int(n * percent_missing_feature2))

    df.loc[nanidx_missing_feature1, missing_feature1] = np.NaN
    df.loc[nanidx_missing_feature2, missing_feature2] = np.NaN

    return df


def mcar_removal(df, proportion):
    # Copy the dataframe as to not modify the old one
    df = df.copy()
    n = df.shape[0]
    remove = int(n*proportion)
    print(f"removing {remove} entries for both features")
    abs_remove = np.random.choice(n, remove, replace=False)
    replacements = np.empty(remove)
    replacements.fill(np.nan)
    df.loc[abs_remove, 'absences'] = replacements
    
    g2_remove = np.random.choice(n, remove, replace=False)
    df.loc[abs_remove, 'G2'] = replacements
    return df
