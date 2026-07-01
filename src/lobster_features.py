"""Tier B: LOBSTER message data -> daily order-flow predictors.

For each deep-dive stock we stream the LOBSTER message files (one per stock-year,
object data.<TICKER>.<year>$message.raw), reconstruct the top of book day-by-day,
and emit three daily microstructure predictors:

    ofi      : Order-Flow Imbalance (Cont, Kukanov & Stoikov 2014), daily sum of
               best-quote changes -- needs the reconstructed top of book.
    rel_spread: mean relative quoted spread (ask-bid)/mid over the session.
    signed_imb: signed trade imbalance = sum(-Direction*Size) / sum(Size) over
               executions (trade-initiator sign; no book needed).

LOBSTER conventions used:
    Event Type 1=add, 2=partial cancel, 3=delete, 4=visible exec, 5=hidden exec,
               6=cross/auction (skipped).
    Direction  +1 = bid-side (buy) limit order, -1 = ask-side (sell) limit order.
               An execution of a buy-limit (Direction +1) is seller-initiated,
               so the trade sign is -Direction.
    Time       seconds after midnight; continuous session 34200..57600 (09:30-16:00).
    Price      already in dollars.

Output: one CSV per stock (date, asset, ofi, rel_spread, signed_imb, trade_vol),
which the stock loader merges as lagged predictors for the deep-dive panel.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from sortedcontainers import SortedDict

SESSION_OPEN = 34200      # 09:30:00
SESSION_CLOSE = 57600     # 16:00:00


_MSG_COLS = ['Time', 'Event Type', 'Order ID', 'Size', 'Price', 'Direction', 'Dates']


def read_message_df(path: Path) -> pd.DataFrame:
    """Read a LOBSTER message file -> DataFrame with the standard columns.

    Supports .parquet (preferred for big stocks; columnar + low memory) and the
    original nested .RData (fine for small stocks). For .RData the object is
    data.<TICKER>.<year>$message.raw.
    """
    path = Path(path)
    if path.suffix == '.parquet':
        df = pd.read_parquet(path, columns=_MSG_COLS)
        df.columns = [str(c) for c in df.columns]
        return df
    import rdata
    parsed = rdata.read_rda(str(path))
    key = next((k for k in parsed if str(k).startswith('data.')), list(parsed)[0])
    obj = parsed[key]
    if isinstance(obj, dict):
        mk = next((k for k in obj if 'message' in str(k).lower()), list(obj)[0])
        df = obj[mk]
    else:
        df = obj
    df = df.copy()
    df.columns = [str(c) for c in df.columns]
    return df


def _downcast(df):
    """Keep only needed columns and shrink dtypes to cut memory."""
    df = df[[c for c in _MSG_COLS if c in df.columns]].copy()
    df['Time'] = df['Time'].astype('float64')          # keep precision for sort
    for c in ['Event Type', 'Direction']:
        df[c] = df[c].astype('int16')
    for c in ['Size', 'Dates']:
        df[c] = pd.to_numeric(df[c], downcast='integer')
    df['Order ID'] = df['Order ID'].astype('int64')
    df['Price'] = df['Price'].astype('float64')
    return df


def _aggregate_day(g):
    """Sort one day's messages and return the daily order-flow record."""
    g = g.sort_values('Time')
    ofi, spr, imb, vol = _process_day(
        g['Time'].values, g['Event Type'].values.astype(int), g['Order ID'].values,
        g['Size'].values.astype(float), g['Price'].values.astype(float),
        g['Direction'].values.astype(int))
    day = pd.to_datetime(int(g['Dates'].iloc[0]), unit='D').normalize()
    return {'date': day, 'ofi': ofi, 'rel_spread': spr, 'signed_imb': imb, 'trade_vol': vol}


def _year_rows_streaming(path, asset, batch_size=3_000_000):
    """Stream a .parquet year file in row-group batches; process complete days.

    Memory stays flat (~one batch + one day) regardless of file size, so even
    AAPL's multi-GB years are safe on a laptop.
    """
    import pyarrow.parquet as pq
    import gc
    pf = pq.ParquetFile(path)
    leftover = None
    rows = []
    for batch in pf.iter_batches(batch_size=batch_size, columns=_MSG_COLS):
        t = _downcast(batch.to_pandas())
        t = t[(t['Time'] >= SESSION_OPEN) & (t['Time'] <= SESSION_CLOSE)]
        if leftover is not None and len(leftover):
            t = pd.concat([leftover, t], ignore_index=True)
        if not len(t):
            leftover = None; continue
        uniq = np.unique(t['Dates'].values)
        last = uniq[-1]                                  # last day may continue next batch
        for d in uniq[:-1]:
            rows.append(_aggregate_day(t[t['Dates'].values == d]))
        leftover = t[t['Dates'].values == last].copy()
        del t; gc.collect()
    if leftover is not None and len(leftover):
        for d in np.unique(leftover['Dates'].values):
            rows.append(_aggregate_day(leftover[leftover['Dates'].values == d]))
    return rows


def _year_rows_inmemory(path, asset):
    """Full-load a (small) .RData year file and aggregate per day."""
    df = read_message_df(path).rename(columns={'Event.Type': 'Event Type', 'Order.ID': 'Order ID'})
    df = _downcast(df)
    df = df[(df['Time'] >= SESSION_OPEN) & (df['Time'] <= SESSION_CLOSE)]
    rows = [_aggregate_day(g) for _, g in df.groupby('Dates', sort=True)]
    return rows


def build_lobster_daily(ticker: str, msg_dir: Path, years, file_pattern='{ticker}_raw_{year}.RData',
                        out_path: Path | None = None, resume=True, verbose=True) -> pd.DataFrame:
    """Process a stock's year files -> daily order-flow CSV, per-year and resumable.

    For each year it prefers a .parquet file (streamed, memory-flat) and falls
    back to .RData (full-load, fine for small stocks). Each completed year is
    appended to out_path immediately, so a kill loses at most the in-progress
    year; rerun with resume=True to continue.
    """
    import gc
    asset = f'{ticker}_STOCK_RV5'
    done_years = set()
    if out_path is not None and Path(out_path).exists() and resume:
        prev = pd.read_csv(out_path, parse_dates=['date'])
        done_years = set(pd.DatetimeIndex(prev['date']).year)
        if verbose and done_years:
            print(f'  [resume] {ticker}: years already done {sorted(done_years)}')

    all_rows = []
    for yr in years:
        if yr in done_years:
            continue
        base = Path(msg_dir) / file_pattern.format(ticker=ticker, year=yr)
        pq_path = base.with_suffix('.parquet')
        if pq_path.exists():
            src, stream = pq_path, True
        elif base.exists():
            src, stream = base, False
        else:
            if verbose:
                print(f'  [skip] {base.name} / .parquet not found')
            continue
        if verbose:
            print(f'  [LOBSTER] {ticker} {yr} ({"parquet/stream" if stream else "rdata"}) ...', flush=True)
        rows = _year_rows_streaming(src, asset) if stream else _year_rows_inmemory(src, asset)
        for r in rows:
            r['asset'] = asset
        ydf = pd.DataFrame(rows).sort_values('date')
        if out_path is not None and len(ydf):
            header = not Path(out_path).exists()
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            ydf.to_csv(out_path, mode='a', header=header, index=False)
            if verbose:
                print(f'    -> appended {len(ydf)} days for {yr}')
        all_rows.append(ydf)
        gc.collect()

    if out_path is not None and Path(out_path).exists():
        return pd.read_csv(out_path, parse_dates=['date']).sort_values('date').reset_index(drop=True)
    return pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()


def _process_day(time, etype, oid, size, price, direction):
    """Reconstruct top of book for one day; return (ofi, rel_spread, signed_imb, vol)."""
    bid = SortedDict()      # price -> total resting size (buy side)
    ask = SortedDict()      # price -> total resting size (sell side)
    orders = {}             # order_id -> (side, price, size)

    ofi = 0.0
    spread_sum = 0.0
    spread_n = 0
    signed_vol = 0.0
    tot_vol = 0.0
    prev = None             # (Pb, qb, Pa, qa) previous top of book

    n = len(time)
    for i in range(n):
        et = etype[i]
        d = direction[i]
        p = price[i]
        s = size[i]
        o = oid[i]
        changed = False

        if et == 1:                                   # new limit order
            book = bid if d == 1 else ask
            book[p] = book.get(p, 0.0) + s
            orders[o] = (d, p, s)
            # Uncross: without the LOBSTER orderbook snapshot, stale opposite
            # quotes accumulate. A resting limit cannot sit through a new order
            # on the crossing side, so clear stale opposite levels.
            if d == 1:                                 # buy -> drop stale asks <= p
                while ask and ask.peekitem(0)[0] <= p:
                    ask.popitem(0)
            else:                                      # sell -> drop stale bids >= p
                while bid and bid.peekitem(-1)[0] >= p:
                    bid.popitem(-1)
            changed = True
        elif et in (2, 3, 4):                         # reduce a resting order
            if o in orders:
                d0, p0, s0 = orders[o]
                book = bid if d0 == 1 else ask
                dec = s0 if et == 3 else s             # full delete vs partial/exec size
                rem = book.get(p0, 0.0) - dec
                if rem <= 0:
                    book.pop(p0, None)
                else:
                    book[p0] = rem
                ns = s0 - dec
                if ns <= 0:
                    orders.pop(o, None)
                else:
                    orders[o] = (d0, p0, ns)
                if et == 4:                            # visible execution = trade
                    signed_vol += -d0 * s
                    tot_vol += s
                changed = True
            elif et == 4:                              # exec of an order not in book
                signed_vol += -d * s
                tot_vol += s
        elif et == 5:                                  # hidden execution = trade only
            signed_vol += -d * s
            tot_vol += s

        if changed and bid and ask:
            Pb, qb = bid.peekitem(-1)                  # best bid = highest buy price
            Pa, qa = ask.peekitem(0)                   # best ask = lowest sell price
            if Pa > 0 and Pb > 0:
                mid = 0.5 * (Pa + Pb)
                spread_sum += (Pa - Pb) / mid
                spread_n += 1
                if prev is not None:
                    Pb0, qb0, Pa0, qa0 = prev
                    eb = (qb if Pb >= Pb0 else 0.0) - (qb0 if Pb <= Pb0 else 0.0)
                    ea = (qa if Pa <= Pa0 else 0.0) - (qa0 if Pa >= Pa0 else 0.0)
                    ofi += eb - ea
                prev = (Pb, qb, Pa, qa)

    rel_spread = spread_sum / spread_n if spread_n else np.nan
    signed_imb = signed_vol / tot_vol if tot_vol > 0 else np.nan
    return ofi, rel_spread, signed_imb, tot_vol


if __name__ == '__main__':
    import sys
    src = sys.argv[1] if len(sys.argv) > 1 else '/mnt/user-data/uploads/1781968737774_AGEN_raw_2012.RData'
    df = read_message_df(Path(src)).rename(columns={'Event Type': 'EventType', 'Order ID': 'OrderID'})
    df = df[(df['Time'] >= SESSION_OPEN) & (df['Time'] <= SESSION_CLOSE)]
    df['date'] = pd.to_datetime(df['Dates'], unit='D').dt.normalize()
    out = []
    for day, g in df.groupby('date', sort=True):
        g = g.sort_values('Time')
        ofi, spr, imb, vol = _process_day(
            g['Time'].values, g['EventType'].values.astype(int), g['OrderID'].values,
            g['Size'].values.astype(float), g['Price'].values.astype(float),
            g['Direction'].values.astype(int))
        out.append({'date': day, 'ofi': ofi, 'rel_spread': spr, 'signed_imb': imb, 'trade_vol': vol})
    r = pd.DataFrame(out)
    print(f'AGEN 2012: {len(r)} trading days')
    print(r[['rel_spread', 'signed_imb', 'ofi', 'trade_vol']].describe().to_string())
    print('\nfirst rows:'); print(r.head().to_string(index=False))
