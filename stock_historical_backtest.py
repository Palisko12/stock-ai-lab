"""Reproducible, no-orders daily-close stock momentum backtest.

Signal uses completed daily closes at t. Trades execute at t+1 open.
Portfolio is equal weighted across up to five qualifying stocks, remaining cash.
Includes costs, same fixed universe (survivorship bias), no dividends/corporate taxes.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf

DEFAULT = "AAPL,MSFT,NVDA,AMZN,GOOGL,META,TSLA,AMD,AVGO,JPM,XOM,UNH"
def get_bars(symbol,start,end):
    d=yf.download(symbol,start=start,end=end,auto_adjust=True,progress=False,threads=False)
    if d.empty:return None
    if isinstance(d.columns,pd.MultiIndex):d.columns=d.columns.get_level_values(0)
    d=d[["Open","Close"]].dropna().sort_index()
    d.index=pd.DatetimeIndex(d.index).tz_localize(None).normalize()
    return d if len(d)>=100 else None

def weights_at(bars,decision_day,tickers):
    candidates=[]
    for sym in tickers:
        df=bars[sym]
        ix=df.index.searchsorted(decision_day,side="right")-1
        # Only trade when latest data is actually the decision day's close.
        if ix<90 or df.index[ix]!=decision_day:continue
        close=df["Close"]
        p=float(close.iloc[ix])
        m1=p/float(close.iloc[ix-21])-1
        m3=p/float(close.iloc[ix-63])-1
        ma=float(close.iloc[ix-49:ix+1].mean())
        if all(np.isfinite(z) for z in (p,m1,m3,ma)) and p>ma and m3>0:
            candidates.append((.4*m1+.6*m3,sym))
    top=[s for _,s in sorted(candidates,reverse=True)[:5]]
    return {s:1/len(top) for s in top} if top else {}

def main():
    a=argparse.ArgumentParser()
    a.add_argument("--start",default="2023-01-01")
    a.add_argument("--end",default="2026-09-28")
    a.add_argument("--tickers",default=DEFAULT)
    a.add_argument("--fee-bps",type=float,default=10)
    a.add_argument("--rebalance-days",type=int,default=21)
    a.add_argument("--capital",type=float,default=10000)
    args=a.parse_args()
    if args.fee_bps<0 or args.rebalance_days<1 or args.capital<=0:a.error("invalid fee, rebalance days, or capital")
    tickers=list(dict.fromkeys(x.strip().upper() for x in args.tickers.split(",") if x.strip()))
    start=pd.Timestamp(args.start);end=pd.Timestamp(args.end)
    if end<=start:a.error("end must be after start")
    # Warmup includes at least 100 completed sessions.
    download_start=(start-pd.Timedelta(days=250)).strftime("%Y-%m-%d")
    download_end=(end+pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    bars={}
    for s in tickers+["SPY"]:
        try:
            d=get_bars(s,download_start,download_end)
            if d is not None:bars[s]=d
        except Exception as e:print("SKIP",s,repr(e))
    if "SPY" not in bars:raise SystemExit("SPY data missing")
    tickers=[s for s in tickers if s in bars and s!="SPY"]
    if len(tickers)<2:raise SystemExit("Too few securities")
    # Use only sessions with prices for the entire universe and SPY, so no forward fill.
    dates=bars["SPY"].index
    for s in tickers:dates=dates.intersection(bars[s].index)
    dates=dates[(dates>=start)&(dates<=end)]
    if len(dates)<100:raise SystemExit("Too few common sessions")
    positions={}
    cash=float(args.capital)
    close_equities=[]
    trades=0;turnover=0.0
    total_fee_dollars=0.0
    cost_rate=args.fee_bps/10000
    peak=float(args.capital);max_dd=0.0
    # Initial signal is previous completed session, execution at next session open.
    prev=bars["SPY"].index[bars["SPY"].index < dates[0]]
    if len(prev)==0:raise SystemExit("No pre-start session")
    signal_day=prev[-1]
    target=weights_at(bars,signal_day,tickers)
    for i,day in enumerate(dates):
        opens={s:float(bars[s].loc[day,"Open"]) for s in tickers}
        closes={s:float(bars[s].loc[day,"Close"]) for s in tickers}
        if i % args.rebalance_days==0:
            equity_open=cash+sum(q*opens[s] for s,q in positions.items())
            old={s:q*opens[s]/equity_open for s,q in positions.items()} if equity_open>0 else {}
            turnover+=sum(abs(target.get(s,0)-old.get(s,0)) for s in set(old)|set(target))
            # Net trades only: preserve unchanged positions and rebalance differences.
            # Target notional uses pre-fee equity. Sells happen before buys; buys scale to cash.
            desired={sym:equity_open*weight for sym,weight in target.items()}
            for sym,qty in list(positions.items()):
                current=qty*opens[sym]
                sell_notional=max(0.0,current-desired.get(sym,0.0))
                if sell_notional>1e-8:
                    sell_qty=min(qty,sell_notional/opens[sym])
                    proceeds=sell_qty*opens[sym]
                    cash+=proceeds*(1-cost_rate)
                    total_fee_dollars+=proceeds*cost_rate
                    positions[sym]=qty-sell_qty
                    trades+=1
                    if positions[sym]<1e-10:del positions[sym]
            buys={sym:max(0.0,wanted-positions.get(sym,0.0)*opens[sym])
                  for sym,wanted in desired.items()}
            required=sum(buys.values())*(1+cost_rate)
            scale=min(1.0,max(0.0,cash)/required) if required>0 else 0.0
            for sym,notional in buys.items():
                actual=notional*scale
                if actual<=1e-8:continue
                qty=actual/opens[sym]
                positions[sym]=positions.get(sym,0.0)+qty
                cash-=actual*(1+cost_rate)
                total_fee_dollars+=actual*cost_rate
                trades+=1
            if cash< -1e-6:raise AssertionError("negative cash after rebalance")
        equity_close=cash+sum(q*closes[s] for s,q in positions.items())
        peak=max(peak,equity_close)
        max_dd=min(max_dd,equity_close/peak-1)
        close_equities.append(equity_close)
        # Make next decision at TODAY's close; do not use tomorrow's close.
        if (i+1)%args.rebalance_days==0:
            target=weights_at(bars,day,tickers)
    ending=close_equities[-1]
    first=dates[0];last=dates[-1]
    spy=bars["SPY"]
    spy_return=float(spy.loc[last,"Close"]/spy.loc[first,"Open"]-1)
    benchmark_equity=[args.capital*float(spy.loc[day,"Close"])/float(spy.loc[first,"Open"]) for day in dates]
    years=max((last-first).days/365.25,1/365.25)
    r={"mode":"BACKTEST_NO_ORDERS","start":str(first.date()),"end":str(last.date()),"sessions":len(dates),
       "universe":tickers,"fee_bps_each_side":args.fee_bps,"rebalance_days":args.rebalance_days,
       "initial_capital":args.capital,"ending_equity":round(ending,2),
       "total_return_pct":round(100*(ending/args.capital-1),2),
       "annualized_return_pct":round(100*((ending/args.capital)**(1/years)-1),2),
       "max_drawdown_pct":round(100*max_dd,2),"SPY_return_pct":round(100*spy_return,2),
       "orders_simulated":trades,"turnover_weight_sum":round(turnover,2),
       "total_fees_usd":round(total_fee_dollars,2),
       "caveats":"Research-only. Fixed 2026 universe creates survivorship/selection bias; no dividends, slippage beyond fees, taxes, liquidity, or broker fills. Partial shares; possible corporate-action distortions."}
    Path("backtest_output").mkdir(exist_ok=True)
    Path("backtest_output/results.json").write_text(json.dumps(r,indent=2),encoding="utf-8")
    pd.DataFrame({"date":[str(d.date()) for d in dates],"equity":close_equities,"spy_equity":benchmark_equity}).to_csv("backtest_output/equity.csv",index=False)
    print(json.dumps(r,indent=2))
if __name__=="__main__":main()
