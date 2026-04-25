"""
Data loading and sequence building for modern TS models.
"""

import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download
from sklearn.preprocessing import StandardScaler
import torch
import config

def load_master_data():
    print(f"Downloading {config.HF_DATA_FILE} from {config.HF_DATA_REPO}...")
    path = hf_hub_download(
        repo_id=config.HF_DATA_REPO, filename=config.HF_DATA_FILE,
        repo_type="dataset", token=config.HF_TOKEN, cache_dir="./hf_cache"
    )
    df = pd.read_parquet(path)
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index().rename(columns={'index': 'Date'})
    df['Date'] = pd.to_datetime(df['Date'])
    return df

def prepare_returns_matrix(df_wide, tickers):
    available = [t for t in tickers if t in df_wide.columns]
    df_long = df_wide.melt(id_vars=['Date'], value_vars=available,
                           var_name='ticker', value_name='price')
    df_long = df_long.sort_values(['ticker', 'Date'])
    df_long['log_return'] = df_long.groupby('ticker')['price'].transform(
        lambda x: np.log(x / x.shift(1))
    )
    df_long = df_long.dropna(subset=['log_return'])
    return df_long.pivot(index='Date', columns='ticker', values='log_return')[available].dropna()

def prepare_macro(df_wide):
    macro_cols = [c for c in config.MACRO_COLS if c in df_wide.columns]
    macro_df = df_wide[['Date'] + macro_cols].copy()
    macro_df = macro_df.set_index('Date').ffill().dropna()
    return macro_df

def build_sequences(returns, macro):
    """
    Returns:
        X: (num_samples, CONTEXT_LEN, num_features)  where features = ETFs + macro
        y: (num_samples, num_etfs)
        scalers
    """
    common = returns.index.intersection(macro.index)
    returns = returns.loc[common]
    macro = macro.loc[common]

    tickers = returns.columns.tolist()
    n_etfs = len(tickers)
    n_features = n_etfs + len(macro.columns)

    # Scale all features globally
    all_data = np.concatenate([returns.values, macro.values], axis=1)
    scaler = StandardScaler().fit(all_data)
    scaled = scaler.transform(all_data)

    X_list, y_list = [], []
    for i in range(len(scaled) - config.CONTEXT_LEN - config.TARGET_HORIZON + 1):
        X_list.append(scaled[i:i+config.CONTEXT_LEN])
        # target = next-day return for each ETF (first n_etfs columns of next row)
        y_list.append(scaled[i+config.CONTEXT_LEN, :n_etfs])

    X = np.stack(X_list, axis=0)  # (samples, CONTEXT_LEN, n_features)
    y = np.stack(y_list, axis=0)  # (samples, n_etfs)
    return X, y, scaler, tickers
