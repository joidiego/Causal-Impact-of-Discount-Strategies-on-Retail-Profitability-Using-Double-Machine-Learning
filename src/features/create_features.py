# src/features/create_features.py
from src.features.encode import one_hot_encode
import pandas as pd

def create_causal_features(df):
    cat_cols = ['Category', 'Sub-Category', 'Region', 'Segment', 'Ship_Mode']
    num_cols = [
        'Sales', 'Quantity', 'month', 'year', 'day_of_week', 
        'shipping_duration', 'discount_amount', 'profit_margin'
    ]

    # One-hot encode
    df_encoded, _ = one_hot_encode(df, cat_cols)

    # Ambil hanya kolom numerik yang valid
    X = df_encoded[num_cols + [col for col in df_encoded.columns if col.startswith(tuple(cat_cols))]]
    T = df['Discount'].values
    Y = df['Profit'].values

    return X, T, Y