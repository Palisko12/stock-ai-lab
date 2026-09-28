"""Independent, transparent stock momentum research baseline. No broker orders."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import yfinance as yf

DEFAULT = "AAPL,MSFT,NVDA,AMZN,GOOGL,META,TSLA,AMD,AVGO,JPM,XOM,UNH"
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--tickers",default=DEFAULT)
    p.add_argument("--capital",type=float,default=10000)
    a=p.parse_args()
    if a.capital<=0: p.error("capital must be positive")
    tickers=list(dict.fromkeys(x.strip().upper() for x in a.tickers.split(",") if x.strip()))
    if len(tickers)<2: p.error("need at least two tickers")
    results=[]
    for symbol in tickers:
        try:
            bars=yf.download(symbol,period="8mo",interval="1d",auto_adjust=True,progress=False,threads=False)
            if len(bars)<90: continue
            close=bars["Close"]
            if hasattr(close,"columns"): close=close.iloc[:,0]
            close=close.dropna()
            if len(close)<90: continue
            m1=float(close.iloc[-1]/close.iloc[-22]-1)
            m3=float(close.iloc[-1]/close.iloc[-64]-1)
            ma50=float(close.iloc[-50:].mean())
            price=float(close.iloc[-1])
            if not all(np.isfinite(x) for x in (m1,m3,ma50,price)): continue
            results.append({"ticker":symbol,"close":round(price,4),"momentum_1m_pct":round(100*m1,2),"momentum_3m_pct":round(100*m3,2),"above_sma50":price>ma50,"ranking_score":round(.4*m1+.6*m3,6)})
        except Exception as e:
            print(f"SKIP {symbol}: {type(e).__name__}: {e}")
    ranked=sorted((r for r in results if r["above_sma50"] and r["momentum_3m_pct"]>0),key=lambda r:r["ranking_score"],reverse=True)
    selected=ranked[:5]
    allocation=round(100/len(selected),2) if selected else 0
    for r in selected:r["target_weight_pct"]=allocation
    report={"timestamp_utc":datetime.now(timezone.utc).isoformat(),"mode":"RESEARCH_ONLY_NO_ORDERS","initial_virtual_capital_usd":a.capital,"universe":tickers,"selected":selected,"all_analyzed":results,"notes":"Signal snapshot only. Not a historical backtest, not paper execution, no PnL; survivorship bias possible."}
    Path("research_output").mkdir(exist_ok=True)
    Path("research_output/latest.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))
    if len(results)<2:raise SystemExit("Insufficient valid market data")
if __name__=="__main__":main()
