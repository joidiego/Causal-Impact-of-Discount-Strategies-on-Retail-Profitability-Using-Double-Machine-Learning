# src/features/encode.py
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

def one_hot_encode(df, columns):
    """
    One-hot encode kolom kategorikal.
    """
    encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
    encoded = encoder.fit_transform(df[columns])
    feature_names = encoder.get_feature_names_out(columns)
    encoded_df = pd.DataFrame(encoded, columns=feature_names, index=df.index)
    df_remaining = df.drop(columns, axis=1)
    return pd.concat([df_remaining, encoded_df], axis=1), encoder