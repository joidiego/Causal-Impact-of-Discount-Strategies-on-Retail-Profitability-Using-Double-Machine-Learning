# src/data/clean.py
import pandas as pd
import numpy as np

def load_and_clean_data(path):
    """
    Load and clean raw Superstore data.
    """
    df = pd.read_csv(path, encoding='latin1', parse_dates=['Order Date', 'Ship Date'])
    
    # Normalize column names
    df.columns = [col.replace(" ", "_") for col in df.columns]
    df = df.rename(columns={
        "Order_Date": "order_date",
        "Ship_Date": "ship_date",
        "Customer_ID": "customer_id",
        "Product_ID": "product_id"
    })

    # Feature engineering
    df['discount_amount'] = df['Sales'] * df['Discount']
    df['profit_margin'] = df['Profit'] / (df['Sales'] + 1e-8)
    df['month'] = df['order_date'].dt.month          # 1-12
    df['year'] = df['order_date'].dt.year            # 2014, 2015, ...
    df['day_of_week'] = df['order_date'].dt.dayofweek
    df['shipping_duration'] = (df['ship_date'] - df['order_date']).dt.days

    # Handle missing
    df.dropna(subset=['Profit', 'Sales'], inplace=True)

    return df