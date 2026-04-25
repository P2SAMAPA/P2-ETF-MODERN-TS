# P2-ETF-MODERN-TS

**Modern Time‑Series Forecasting – PatchTST, TimesNet, TSMixer, and FiLM**

[![Daily Run](https://github.com/P2SAMAPA/P2-ETF-MODERN-TS/actions/workflows/daily_run.yml/badge.svg)](https://github.com/P2SAMAPA/P2-ETF-MODERN-TS/actions/workflows/daily_run.yml)
[![Hugging Face Dataset](https://img.shields.io/badge/🤗%20Dataset-p2--etf--modern--ts--results-blue)](https://huggingface.co/datasets/P2SAMAPA/p2-etf-modern-ts-results)

## Overview

`P2-ETF-MODERN-TS` integrates four state‑of‑the‑art time‑series models:

- **PatchTST** – Patch‑based Transformer with channel independence.
- **TimesNet** – 2D convolution on time‑frequency representations.
- **TSMixer** – Lightweight all‑MLP with time and feature mixing.
- **FiLM** – Feature‑wise Linear Modulation conditioned on macro variables.

All models train in parallel on the full 2008‑2026 dataset and predict next‑day ETF returns. The dashboard displays each model's forecasts in a separate tab.

## Methodology

- **Shared input**: rolling window of 60 days of ETF returns + macro features.
- **Training**: 80 epochs per model, parallel across 4 workers.
- **Inference**: latest sequence → next‑day return for each ETF.

## Usage

```bash
pip install -r requirements.txt
python trainer.py
streamlit run streamlit_app.py
