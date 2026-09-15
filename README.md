# algo

Algo trading bot — BTC/USDT. Roadmap: data -> strategi -> backtest -> ML -> paper trading -> live.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Fase 1: data pipeline

```bash
python fetch_data.py                      # default: BTC/USDT 1d, 730 hari
python explore_data.py                    # statistik + plot
```

## Struktur

- `fetch_data.py` — ambil + validasi klines dari Binance (ccxt)
- `explore_data.py` — statistik & visualisasi
- `data/` — hasil fetch (csv) & plot
