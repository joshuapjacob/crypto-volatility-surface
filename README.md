# [Crypto Volatility Surface](https://joshuapjacob.com/crypto-volatility-surface)

A dashboard to visualize cryptocurrency implied volatility surfaces constructed with option data from Binance.

![Volatility Surface](./thumbnail.png)

## Run Locally

```bash
uv run streamlit run app.py
```

If you are in a restricted location (like the United States) and use a proxy to access Binance, set a `BINANCE_PROXY` environment variable.

```bash
# For example:
BINANCE_PROXY="socks5://localhost:5000"
```
