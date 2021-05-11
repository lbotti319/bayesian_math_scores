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
    
    joined = X.join(df_enc)
    final_features = joined[['intercept', 'age', 'sex_M', 'failures', 'higher_yes', 'Medu', 'absences', 'G2']]

    return final_features, y

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