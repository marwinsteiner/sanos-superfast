"""Fetch SPX option chain from Yahoo Finance and dump as JSON for C++ consumption."""
import json
import yfinance as yf
from datetime import datetime

def fetch_spx_chain():
    ticker = yf.Ticker("^SPX")
    spot = ticker.info.get("regularMarketPrice") or ticker.fast_info.get("lastPrice", 0)

    # Get all available expiry dates
    expirations = ticker.options
    print(f"SPX spot: {spot}")
    print(f"Available expiries: {len(expirations)}")

    # Limit to first 20 expiries for manageable size
    expirations = expirations[:20]

    result = {
        "symbol": "SPX",
        "spot": spot,
        "fetch_time": datetime.now().isoformat(),
        "expiries": []
    }

    for exp_str in expirations:
        try:
            chain = ticker.option_chain(exp_str)
        except Exception as e:
            print(f"  Skipping {exp_str}: {e}")
            continue

        calls = chain.calls
        puts = chain.puts

        # Filter: only keep options with nonzero bid and ask
        calls = calls[(calls["bid"] > 0) & (calls["ask"] > 0)].copy()

        if len(calls) < 5:
            print(f"  Skipping {exp_str}: only {len(calls)} valid calls")
            continue

        exp_data = {
            "expiration": exp_str,
            "n_calls": len(calls),
            "strikes": calls["strike"].tolist(),
            "bids": calls["bid"].tolist(),
            "asks": calls["ask"].tolist(),
            "last_prices": calls["lastPrice"].tolist(),
            "implied_vols": calls["impliedVolatility"].tolist(),
            "volumes": calls["volume"].fillna(0).astype(int).tolist(),
            "open_interest": calls["openInterest"].fillna(0).astype(int).tolist(),
        }

        result["expiries"].append(exp_data)
        print(f"  {exp_str}: {len(calls)} calls, strikes [{calls['strike'].min():.0f} - {calls['strike'].max():.0f}]")

    out_path = "data/spx_options.json"
    import os
    os.makedirs("data", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nWritten to {out_path}")
    print(f"Total expiries: {len(result['expiries'])}")
    total_opts = sum(e['n_calls'] for e in result['expiries'])
    print(f"Total options: {total_opts}")

if __name__ == "__main__":
    fetch_spx_chain()
