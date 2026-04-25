"""
Streamlit Dashboard for Modern TS Engine (4 + 1 tabs).
"""

import streamlit as st
import pandas as pd
import numpy as np
from huggingface_hub import HfApi, hf_hub_download
import json
import config
from us_calendar import USMarketCalendar

st.set_page_config(page_title="P2Quant Modern TS", page_icon="⏰", layout="wide")

st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: 600; color: #1f77b4; }
    .hero-card { background: linear-gradient(135deg, #1f77b4 0%, #2C5282 100%); border-radius: 16px; padding: 2rem; color: white; text-align: center; }
    .hero-ticker { font-size: 4rem; font-weight: 800; }
    .metric-positive { color: #28a745; font-weight: 600; }
    .metric-negative { color: #dc3545; font-weight: 600; }
    .combined-header { font-size: 1.8rem; font-weight: 600; margin-top: 2rem; }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def load_latest_results():
    try:
        api = HfApi(token=config.HF_TOKEN)
        files = api.list_repo_files(repo_id=config.HF_OUTPUT_REPO, repo_type="dataset")
        json_files = sorted([f for f in files if f.startswith("modern_ts_") and f.endswith('.json')], reverse=True)
        if not json_files:
            return None
        path = hf_hub_download(repo_id=config.HF_OUTPUT_REPO, filename=json_files[0],
                               repo_type="dataset", token=config.HF_TOKEN, cache_dir="./hf_cache")
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return None

def safe_pct(val):
    try:
        return f"{float(val)*100:.2f}%"
    except:
        return "N/A"

def display_model_tab(model_data, model_name):
    if not model_data:
        st.warning(f"No {model_name} data available.")
        return
    top_picks = model_data.get('top_picks', {})
    universes = model_data.get('universes', {})
    subtabs = st.tabs(["📊 Combined", "📈 Equity Sectors", "💰 FI/Commodities"])
    keys = ["COMBINED", "EQUITY_SECTORS", "FI_COMMODITIES"]
    for subtab, key in zip(subtabs, keys):
        with subtab:
            top = top_picks.get(key, [])
            universe = universes.get(key, {})
            if top:
                p = top[0]
                st.markdown(f"""
                <div class="hero-card">
                    <div style="font-size: 1.2rem; opacity: 0.8;">{model_name} TOP PICK</div>
                    <div class="hero-ticker">{p['ticker']}</div>
                    <div>Forecast: {safe_pct(p['forecast'])}</div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("### Top 3 Picks")
                rows = [{"Ticker": p['ticker'], "Forecast": safe_pct(p['forecast'])} for p in top]
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

                st.markdown("### All ETFs")
                all_rows = [{"Ticker": t, "Forecast": safe_pct(d)} for t, d in universe.items()]
                df_all = pd.DataFrame(all_rows).sort_values("Forecast", ascending=False)
                st.dataframe(df_all, use_container_width=True, hide_index=True)

def display_combined_tab(data):
    if not data:
        st.warning("No data available.")
        return
    model_names = ["PatchTST", "TimesNet", "TSMixer", "FiLM"]
    universes = ["COMBINED", "EQUITY_SECTORS", "FI_COMMODITIES"]
    subtabs = st.tabs(["📊 Combined", "📈 Equity Sectors", "💰 FI/Commodities"])
    for subtab, uni in zip(subtabs, universes):
        with subtab:
            st.markdown(f"### {uni} – Combined Forecasts (Normalized)")
            # Collect forecasts per model per ETF
            tickers = set()
            model_forecasts = {}
            for model in model_names:
                model_data = data.get(model, {}).get('universes', {}).get(uni, {})
                if model_data:
                    tickers.update(model_data.keys())
                    model_forecasts[model] = model_data

            if not tickers:
                st.info(f"No data for {uni}.")
                continue

            # Build normalized table
            rows = []
            for t in sorted(tickers):
                row = {'Ticker': t}
                raw = []
                for model in model_names:
                    val = model_forecasts.get(model, {}).get(t, None)
                    row[f"{model} (raw)"] = f"{val*100:.3f}%" if val is not None else "N/A"
                    if val is not None:
                        raw.append(val)
                # Compute z-scores for this ETF across models that have a value
                if raw:
                    raw_arr = np.array(raw)
                    mean = raw_arr.mean()
                    std = raw_arr.std()
                    if std > 0:
                        z_scores = (raw_arr - mean) / std
                    else:
                        z_scores = np.zeros_like(raw_arr)
                    # Add z-scores to row
                    for i, model in enumerate(model_names):
                        if model_forecasts.get(model, {}).get(t) is not None:
                            row[f"{model} (z)"] = f"{z_scores[i]:.2f}"
                        else:
                            row[f"{model} (z)"] = "N/A"
                    # Consensus score = average z-score
                    row['Consensus (avg z)'] = f"{z_scores.mean():.2f}"
                else:
                    row['Consensus (avg z)'] = "N/A"
                rows.append(row)

            df = pd.DataFrame(rows)
            # Sort by consensus z-score descending
            df_sorted = df.sort_values("Consensus (avg z)", ascending=False)
            st.dataframe(df_sorted, use_container_width=True, hide_index=True)

            # Highlight top pick based on consensus
            top_row = df_sorted.iloc[0]
            top_ticker = top_row['Ticker']
            top_consensus = top_row['Consensus (avg z)']
            st.markdown(f"""
            <div class="hero-card">
                <div style="font-size: 1.2rem; opacity: 0.8;">🏆 CONSENSUS TOP PICK</div>
                <div class="hero-ticker">{top_ticker}</div>
                <div>Avg Z‑Score: {top_consensus}</div>
            </div>
            """, unsafe_allow_html=True)

# --- Sidebar ---
st.sidebar.markdown("## ⚙️ Configuration")
calendar = USMarketCalendar()
st.sidebar.markdown(f"**📅 Next Trading Day:** {calendar.next_trading_day().strftime('%Y-%m-%d')}")
data = load_latest_results()
if data:
    st.sidebar.markdown(f"**Run Date:** {data.get('run_date', 'Unknown')}")

st.markdown('<div class="main-header">⏰ P2Quant Modern TS</div>', unsafe_allow_html=True)
st.markdown('<div>Modern Time‑Series Models: PatchTST, TimesNet, TSMixer, FiLM + Combined View</div>', unsafe_allow_html=True)

if data is None:
    st.warning("No data available.")
    st.stop()

tab1, tab2, tab3, tab4, tab5 = st.tabs(["🧩 PatchTST", "🌊 TimesNet", "⚙️ TSMixer", "🎛️ FiLM", "📊 Combined"])

with tab1:
    display_model_tab(data.get("PatchTST"), "PatchTST")
with tab2:
    display_model_tab(data.get("TimesNet"), "TimesNet")
with tab3:
    display_model_tab(data.get("TSMixer"), "TSMixer")
with tab4:
    display_model_tab(data.get("FiLM"), "FiLM")
with tab5:
    display_combined_tab(data)
