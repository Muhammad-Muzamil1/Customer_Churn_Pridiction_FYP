import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
import sys

# Ensure Unicode prints correctly on Windows
sys.stdout.reconfigure(encoding='utf-8')

def create_customer_features(df, churn_days=90):
    """
    Create customer features from raw transaction data
    """
    # Clean data and force copies to avoid SettingWithCopyWarning
    df = df.copy()
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
    df = df.dropna(subset=['CustomerID']).copy()
    df['CustomerID'] = pd.to_numeric(df['CustomerID'], errors='coerce').astype(int)
    df = df[~df['InvoiceNo'].astype(str).str.startswith('C')].copy()
    df = df[df['Quantity'] > 0].copy()
    df['LineTotal'] = df['Quantity'] * df['UnitPrice']

    # Reference date for recency
    REF_DATE = df['InvoiceDate'].max() + pd.Timedelta(days=1)

    # Customer aggregates
    # Use invoice-level sum for avg_order_value
    invoice_sum = df.groupby(['CustomerID', 'InvoiceNo'])['LineTotal'].sum()
    cust_agg = df.groupby('CustomerID').agg(
        recency_days=('InvoiceDate', lambda x: (REF_DATE - x.max()).days),
        frequency=('InvoiceNo', 'nunique'),
        monetary=('LineTotal', 'sum'),
        avg_order_value=('CustomerID', lambda x: invoice_sum.loc[x.name].mean()),
        unique_products=('StockCode', 'nunique'),
        total_items=('Quantity', 'sum')
    ).reset_index()

    # Mean days between orders
    def mean_days_between(g):
        g = g.sort_values()
        if len(g) <= 1:
            return np.nan
        return g.diff().dt.days.mean()

    mdays = df.groupby('CustomerID')['InvoiceDate'].apply(mean_days_between)
    cust_agg['mean_days_between_orders'] = mdays.fillna(9999)

    # Churn label
    cust_agg['churn'] = (cust_agg['recency_days'] > churn_days).astype(int)

    return cust_agg

def create_recommendation_data(df):
    """
    Create recommendation system matrices
    """
    df = df.copy()
    # User-item matrix
    user_item = df.groupby(['CustomerID', 'StockCode'])['Quantity'].sum().unstack(fill_value=0)

    # Item similarity matrix (item-based collaborative filtering)
    item_user = user_item.T
    item_sim = cosine_similarity(item_user)
    item_sim_df = pd.DataFrame(item_sim, index=item_user.index, columns=item_user.index)

    # Product lookup
    product_lookup = df[['StockCode', 'Description']].drop_duplicates()
    product_lookup = product_lookup.set_index('StockCode')['Description']

    return user_item, item_sim_df, product_lookup

if __name__ == "__main__":
    try:
        # Load data
        df = pd.read_csv("data_30000.csv", encoding='latin-1')
    except FileNotFoundError:
        print("[ERROR] File 'data_30000.csv' not found. Check the path.")
        exit()

    # Create customer features
    cust_agg = create_customer_features(df)

    # Save features
    cust_agg.to_csv('customer_features.csv', index=False)

    # Create recommendation data
    user_item, item_sim_df, product_lookup = create_recommendation_data(df)
    user_item.to_pickle('user_item_matrix.pkl')
    item_sim_df.to_pickle('item_similarity.pkl')
    product_lookup.to_csv('product_lookup.csv')

    print(f"✅ Created customer features with {len(cust_agg)} customers")
    print(f"✅ Features: {', '.join(cust_agg.columns)}")
    print(f"✅ Churn distribution: {cust_agg['churn'].value_counts().to_dict()}")
