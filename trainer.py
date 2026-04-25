"""
Parallel training of all 4 modern TS models.
"""

import json
import pandas as pd
import numpy as np
import torch
from concurrent.futures import ProcessPoolExecutor
import config
import data_manager
from models.patchtst import PatchTST
from models.timesnet import TimesNet
from models.tsmixer import TSMixer
from models.film import FiLM
import push_results

def train_one_model(model_name, X, y, scaler, tickers):
    print(f"[{model_name}] Training on {len(X)} samples...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_etfs = len(tickers)
    n_features = X.shape[2]

    if model_name == "PatchTST":
        model = PatchTST(n_features, config.CONTEXT_LEN, config.PATCH_LEN,
                         config.PATCH_STRIDE, config.PATCH_HIDDEN,
                         config.PATCH_HEADS, config.PATCH_LAYERS, n_etfs).to(device)
    elif model_name == "TimesNet":
        model = TimesNet(config.CONTEXT_LEN, n_features, config.TIMES_HIDDEN,
                         config.TIMES_TOP_K, n_etfs).to(device)
    elif model_name == "TSMixer":
        model = TSMixer(config.CONTEXT_LEN, n_features, config.TSMIXER_HIDDEN,
                        config.TSMIXER_BLOCKS, n_etfs).to(device)
    elif model_name == "FiLM":
        model = FiLM(config.CONTEXT_LEN, n_features, config.FILM_COND_DIM,
                     config.FILM_HIDDEN, n_etfs).to(device)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE,
                                 weight_decay=config.WEIGHT_DECAY)
    criterion = torch.nn.MSELoss()

    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)

    # For FiLM, we need macro condition: last CONTEXT_LEN row of macro features (which are the last columns)
    # macro is the last config.FILM_COND_DIM columns of X
    cond_idx = n_features - config.FILM_COND_DIM
    cond_data = X_t[:, -1, cond_idx:] if model_name == "FiLM" else None

    dataset = torch.utils.data.TensorDataset(X_t, y_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)

    model.train()
    for epoch in range(config.EPOCHS):
        epoch_loss = 0.0
        for batch_X, batch_y in loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            if model_name == "FiLM":
                # Take condition from the batch
                cond = batch_X[:, -1, cond_idx:]
                pred = model(batch_X, cond)
            else:
                pred = model(batch_X)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_X.size(0)
        if (epoch + 1) % 20 == 0:
            print(f"    [{model_name}] Epoch {epoch+1}/{config.EPOCHS} - Loss: {epoch_loss/len(X):.6f}")

    # Predict on the most recent sample
    model.eval()
    latest_X = X_t[-1:].to(device)
    with torch.no_grad():
        if model_name == "FiLM":
            cond = latest_X[:, -1, cond_idx:]
            preds = model(latest_X, cond).cpu().numpy().flatten()
        else:
            preds = model(latest_X).cpu().numpy().flatten()

    forecasts = {tickers[i]: float(preds[i]) for i in range(len(tickers))}
    return model_name, forecasts

def run_modern_ts():
    print(f"=== P2-ETF-MODERN-TS Run: {config.TODAY} ===")
    df_master = data_manager.load_master_data()
    df_master = df_master[df_master['Date'] >= config.TRAIN_START]
    macro = data_manager.prepare_macro(df_master)

    all_results = {}

    for universe_name, tickers in config.UNIVERSES.items():
        print(f"\n--- Processing Universe: {universe_name} ---")
        returns = data_manager.prepare_returns_matrix(df_master, tickers)
        if len(returns) < config.MIN_OBSERVATIONS:
            continue
        X, y, scaler, etf_tickers = data_manager.build_sequences(returns, macro)

        # Train all 4 models in parallel
        with ProcessPoolExecutor(max_workers=4) as executor:
            futures = []
            for model_name in ["PatchTST", "TimesNet", "TSMixer", "FiLM"]:
                futures.append(executor.submit(train_one_model, model_name, X, y, scaler, etf_tickers))

            for future in futures:
                model_name, forecasts = future.result()
                if model_name not in all_results:
                    all_results[model_name] = {}
                all_results[model_name][universe_name] = forecasts

    # Build output payload
    output = {"run_date": config.TODAY, "config": {}}
    for model_name in ["PatchTST", "TimesNet", "TSMixer", "FiLM"]:
        output[model_name] = {
            "universes": all_results.get(model_name, {}),
            "top_picks": {}
        }
        for uni, forecasts in all_results.get(model_name, {}).items():
            sorted_items = sorted(forecasts.items(), key=lambda x: x[1], reverse=True)
            output[model_name]["top_picks"][uni] = [
                {"ticker": t, "forecast": f} for t, f in sorted_items[:3]
            ]

    push_results.push_daily_result(output)
    print("\n=== Run Complete ===")

if __name__ == "__main__":
    run_modern_ts()
