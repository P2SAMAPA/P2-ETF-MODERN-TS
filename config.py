"""
Configuration for P2-ETF-MODERN-TS engine.
"""

import os
from datetime import datetime

HF_DATA_REPO = "P2SAMAPA/fi-etf-macro-signal-master-data"
HF_DATA_FILE = "master_data.parquet"
HF_OUTPUT_REPO = "P2SAMAPA/p2-etf-modern-ts-results"

FI_COMMODITIES_TICKERS = ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV"]
EQUITY_SECTORS_TICKERS = [
    "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV",
    "XLI", "XLY", "XLP", "XLU", "GDX", "XME",
    "IWF", "XSD", "XBI", "IWM"
]
ALL_TICKERS = list(set(FI_COMMODITIES_TICKERS + EQUITY_SECTORS_TICKERS))
UNIVERSES = {
    "FI_COMMODITIES": FI_COMMODITIES_TICKERS,
    "EQUITY_SECTORS": EQUITY_SECTORS_TICKERS,
    "COMBINED": ALL_TICKERS
}

MACRO_COLS = ["VIX", "DXY", "T10Y2Y", "TBILL_3M"]

# Sequence parameters
CONTEXT_LEN = 60                 # past trading days used as input
TARGET_HORIZON = 1               # predict next day
BATCH_SIZE = 64
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
EPOCHS = 80
RANDOM_SEED = 42
MIN_OBSERVATIONS = 252
TRAIN_START = "2008-01-01"

# PatchTST
PATCH_LEN = 12
PATCH_STRIDE = 12
PATCH_HIDDEN = 128
PATCH_HEADS = 8
PATCH_LAYERS = 3

# TimesNet
TIMES_HIDDEN = 64
TIMES_TOP_K = 3               # periods to use

# TSMixer
TSMIXER_HIDDEN = 64
TSMIXER_BLOCKS = 4

# FiLM
FILM_HIDDEN = 128
FILM_COND_DIM = len(MACRO_COLS)    # number of macro variables

TODAY = datetime.now().strftime("%Y-%m-%d")
HF_TOKEN = os.environ.get("HF_TOKEN", None)
