#!/usr/bin/env python3
"""Interactive PSX index / ETF SIP planner and market insights using psxdata.

Author: Salim Ali Khan · Version 1.1.0 · October 2026

Privacy: this script does not upload your holdings or plans. It only reads/writes
files you choose locally and fetches public PSX market data (see README).
"""

from __future__ import annotations

__version__ = "1.1.0"
__author__ = "Salim Ali Khan"
__date__ = "October 2026"

import os
import sys


def _bootstrap_dependencies() -> None:
    """Install psxdata/pandas only when missing (curl | python friendly)."""
    if os.environ.get("PSX_SIP_SKIP_BOOTSTRAP", "").strip().lower() in (
        "1",
        "yes",
        "true",
    ):
        return

    import importlib.util
    import subprocess

    specs = [
        ("pandas", "pandas>=2.0.0"),
        ("psxdata", "psxdata>=1.2.0"),
    ]
    missing = [spec for mod, spec in specs if importlib.util.find_spec(mod) is None]
    if not missing:
        return

    if os.environ.get("PSX_SIP_NO_PIP", "").strip().lower() in ("1", "yes", "true"):
        print(
            "Missing Python packages. Install once:\n"
            f"  {sys.executable} -m pip install {' '.join(missing)}",
            file=sys.stderr,
        )
        sys.exit(1)

    in_venv = sys.prefix != sys.base_prefix or hasattr(sys, "real_prefix")
    cmd = [sys.executable, "-m", "pip", "install", "-q"]
    if not in_venv:
        cmd.append("--user")
    cmd.extend(missing)
    print(
        "psx-sip: installing missing packages (one-time): "
        + ", ".join(missing),
        file=sys.stderr,
    )
    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError:
        print(
            "Could not install dependencies. Try:\n"
            f"  {sys.executable} -m pip install -r requirements.txt",
            file=sys.stderr,
        )
        sys.exit(1)


_bootstrap_dependencies()

import json
import logging
import math
import re
from datetime import datetime, timedelta, timezone

import pandas as pd
import psxdata
from bs4 import BeautifulSoup
from psxdata.constants import BASE_URL, COLUMN_MAP, INDEX_NAMES
from psxdata.parsers.html import parse_html_table
from psxdata.parsers.normalizers import coerce_numeric
from psxdata.scrapers.base import BaseScraper


def _configure_psxdata() -> None:
    """Map newer PSX headers and hide expected psxdata warnings on noisy index history."""
    # PSX /indices snapshot uses title case; psxdata 1.2.0 only maps "CHANGE".
    COLUMN_MAP.setdefault("Change", "change")
    if os.environ.get("PSX_SIP_VERBOSE", "").strip().lower() not in (
        "1",
        "yes",
        "true",
    ):
        logging.getLogger("psxdata").setLevel(logging.ERROR)


_configure_psxdata()

BOARD_LOT = 500
DEFAULT_COST_BUFFER_PCT = 0.5
HOLDINGS_SNAPSHOT_VERSION = "1"
MISSING_WEIGHT_ALERT_PCT = 3.0
MISSING_TOP_N_ALERT = 10
MISSING_TOTAL_WEIGHT_ALERT_PCT = 15.0
_ETF_BASKET_UNIT_RE = re.compile(r"Per\s+([\d,]+)\s+ETF\s+Units", re.I)
_ETF_CASH_RE = re.compile(r"Cash Component:\s*Rs\.?\s*([\d,]+(?:\.\d+)?)", re.I)
_ETF_CASH_PCT_RE = re.compile(
    r"%\s*Cash Component:\s*([\d,]+(?:\.\d+)?)\s*%?", re.I
)


class _EtfPageScraper(BaseScraper):
    """Minimal scraper for PSX ETF creation-unit pages."""

    def fetch_page(self, etf_symbol: str) -> str:
        url = f"{BASE_URL}/etf/{etf_symbol.upper()}"
        return self._request("GET", url).text


_etf_scraper: _EtfPageScraper | None = None
_etf_frame_cache: dict[str, tuple[pd.DataFrame, dict]] = {}
_etf_catalog: dict[str, dict] | None = None


def _get_etf_scraper() -> _EtfPageScraper:
    global _etf_scraper
    if _etf_scraper is None:
        _etf_scraper = _EtfPageScraper()
    return _etf_scraper


def _parse_num(raw: str) -> float:
    cleaned = raw.replace(",", "").strip()
    if not cleaned or cleaned == "-":
        raise ValueError("missing number")
    return float(cleaned)


def list_etf_symbols() -> pd.DataFrame:
    syms = psxdata.symbols()
    if "is_etf" not in syms.columns:
        return pd.DataFrame(columns=["symbol", "name"])
    etfs = syms[syms["is_etf"] == True].copy()  # noqa: E712
    return etfs.sort_values("symbol").reset_index(drop=True)


def print_etf_basket_policy() -> None:
    print("\nHow often ETF composition changes (PSX / fund manager):")
    print(
        "  • PSX shows the official creation-unit basket (with an “as of” date on each ETF page)."
    )
    print(
        "  • That basket is supplied by the AMC — typically updated at day-end; it is not "
        "rebuilt on every price tick."
    )
    print(
        "  • iNAV on PSX uses that basket with live prices intraday; the list of names "
        "and share counts change when the fund rebalances or the AMC publishes a new unit."
    )
    print(
        "  • During rebalancing, PSX may pause iNAV; holdings can shift after that. "
        "Re-run this planner before a SIP if you want the latest published basket."
    )


def _sector_by_symbol_map() -> dict[str, str]:
    symbols_meta = psxdata.symbols()
    if "sector_name" not in symbols_meta.columns:
        return {}
    return (
        symbols_meta.set_index("symbol")["sector_name"].astype(str).to_dict()
    )


def _market_price_index() -> tuple[pd.DataFrame, str]:
    screener = psxdata.screener()
    if screener.empty or "symbol" not in screener.columns:
        print("Could not load market prices (screener empty).")
        sys.exit(1)
    price_col = "price" if "price" in screener.columns else "current"
    return screener.set_index("symbol"), price_col


def _lookup_unit_price(
    symbol: str, scr: pd.DataFrame, price_col: str
) -> float | None:
    sym = symbol.upper()
    if sym not in scr.index:
        return None
    try:
        return parse_price(scr.loc[sym][price_col])
    except ValueError:
        return None


def _format_etf_unit_price(price: float | None) -> str:
    if price is None:
        return "ETF price: N/A"
    return f"ETF price: Rs. {price:,.2f}"


def _parse_etf_page_quote(html: str) -> float | None:
    el = BeautifulSoup(html, "html.parser").select_one(".quote__close")
    if el is None:
        return None
    text = el.get_text(strip=True).replace("Rs.", "").replace(",", "").strip()
    try:
        return float(text)
    except ValueError:
        return None


def _fetch_etf_parsed(etf_symbol: str) -> tuple[dict | None, str]:
    html = _get_etf_scraper().fetch_page(etf_symbol)
    soup = BeautifulSoup(html, "html.parser")
    return _parse_etf_equity_basket(soup), html


def _resolve_etf_unit_price(
    etf_symbol: str,
    html: str,
    scr: pd.DataFrame,
    price_col: str,
) -> float | None:
    price = _parse_etf_page_quote(html)
    if price is not None:
        return price
    return _lookup_unit_price(etf_symbol, scr, price_col)


def _short_sector_label(sector: str, max_len: int = 30) -> str:
    sector = sector.strip()
    if len(sector) <= max_len:
        return sector
    return sector[: max_len - 1].rstrip() + "…"


def _composition_summary_line(df: pd.DataFrame, meta: dict, top_n: int = 3) -> str:
    sectors = sector_weight_table(df)
    parts = [
        f"{_short_sector_label(row['sector_name'])} {row['sector_weight']:.0f}%"
        for _, row in sectors.head(top_n).iterrows()
    ]
    line = ", ".join(parts) if parts else "No sector breakdown"
    if meta.get("cash_pct"):
        line += f" | cash {meta['cash_pct']:.0f}%"
    if meta.get("as_of"):
        line += f" (basket {meta['as_of']})"
    return line


def _build_etf_frame_from_parsed(
    etf_symbol: str,
    parsed: dict,
    scr: pd.DataFrame,
    sector_by_symbol: dict[str, str],
    price_col: str,
    *,
    quiet: bool = False,
) -> tuple[pd.DataFrame, dict]:
    rows: list[dict] = []
    missing_prices: list[str] = []
    for h in parsed["holdings"]:
        sym = h["symbol"]
        if sym not in scr.index:
            missing_prices.append(sym)
            continue
        scr_row = scr.loc[sym]
        try:
            price = parse_price(scr_row[price_col])
        except ValueError:
            missing_prices.append(sym)
            continue
        sector = sector_by_symbol.get(sym, "")
        value = h["basket_shares"] * price
        rows.append(
            {
                "symbol": sym,
                "name": h["name"],
                "sector_name": sector or "UNKNOWN",
                "price": price,
                "basket_shares": h["basket_shares"],
                "basket_value_pkr": value,
            }
        )

    if missing_prices and not quiet:
        print(f"  Warning: no live price for: {', '.join(missing_prices)}")

    if not rows:
        raise ValueError("no priced holdings")

    df = pd.DataFrame(rows)
    cash_pkr = float(parsed["cash_pkr"])
    equity_total = float(df["basket_value_pkr"].sum())
    basket_total = equity_total + cash_pkr
    if basket_total <= 0:
        raise ValueError("zero basket value")

    df["idx_weight"] = df["basket_value_pkr"] / basket_total * 100.0
    meta = {
        "etf_symbol": etf_symbol,
        "basket_units": parsed["basket_units"],
        "cash_pkr": cash_pkr,
        "cash_pct": parsed["cash_pct"]
        if parsed["cash_pct"] is not None
        else (cash_pkr / basket_total * 100.0),
        "as_of": parsed.get("as_of", ""),
        "title": parsed["title"],
        "missing_price_symbols": list(missing_prices),
        "parsed_holdings": list(parsed["holdings"]),
    }
    return (
        df.sort_values("idx_weight", ascending=False).reset_index(drop=True),
        meta,
    )


def _load_etf_catalog() -> dict[str, dict]:
    global _etf_catalog
    if _etf_catalog is not None:
        return _etf_catalog

    etfs = list_etf_symbols()
    symbols = etfs["symbol"].astype(str).tolist()
    print("\nLoading published ETF baskets from PSX (one request per ETF)...")
    scr, price_col = _market_price_index()
    sector_by_symbol = _sector_by_symbol_map()

    catalog: dict[str, dict] = {}
    for sym in symbols:
        parsed, html = _fetch_etf_parsed(sym)
        etf_price = _resolve_etf_unit_price(sym, html, scr, price_col)
        price_line = _format_etf_unit_price(etf_price)
        if parsed is None:
            catalog[sym] = {
                "kind": "unknown",
                "line": "Basket not available on PSX",
                "price_line": price_line,
                "etf_unit_price": etf_price,
            }
            continue
        if parsed["kind"] == "treasury":
            catalog[sym] = {
                "kind": "treasury",
                "line": "Treasury / T-bills + cash (buy the ETF; not equity SIP here)",
                "price_line": price_line,
                "etf_unit_price": etf_price,
                "parsed": parsed,
                "html": html,
            }
            continue
        try:
            df, meta = _build_etf_frame_from_parsed(
                sym,
                parsed,
                scr,
                sector_by_symbol,
                price_col,
                quiet=True,
            )
        except ValueError:
            catalog[sym] = {
                "kind": "unknown",
                "line": "Could not price basket holdings",
                "price_line": price_line,
                "etf_unit_price": etf_price,
            }
            continue
        meta["etf_unit_price"] = etf_price
        _etf_frame_cache[sym] = (df, meta)
        catalog[sym] = {
            "kind": "equity",
            "line": _composition_summary_line(df, meta),
            "price_line": price_line,
            "etf_unit_price": etf_price,
            "parsed": parsed,
            "html": html,
        }

    _etf_catalog = catalog
    return catalog


def _fmt_insight_num(value: float | None, decimals: int = 2) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    return f"{value:,.{decimals}f}"


def _fmt_insight_pct(value: float | None) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    return f"{value:+.2f}%"


def _pct_below_high(current: float | None, period_high: float | None) -> float | None:
    if (
        current is None
        or period_high is None
        or period_high <= 0
        or (isinstance(current, float) and math.isnan(current))
    ):
        return None
    return (current / period_high - 1.0) * 100.0


def _fetch_indices_market_snapshot() -> pd.DataFrame:
    resp = _get_etf_scraper()._request("GET", f"{BASE_URL}/indices")
    rows = parse_html_table(resp.text)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "index_name" in df.columns:
        df = df.rename(columns={"index_name": "symbol"})
    for col in ("high", "low", "current", "change"):
        if col in df.columns:
            df[col] = df[col].apply(coerce_numeric)
    if "change_pct" in df.columns:
        df["change_pct"] = (
            df["change_pct"].astype(str).str.replace("%", "", regex=False).apply(coerce_numeric)
        )
    return df


def _historical_ohlc(symbol: str) -> pd.DataFrame:
    try:
        df = psxdata.stocks(symbol.upper())
    except Exception:
        return pd.DataFrame()
    if df.empty or "date" not in df.columns:
        return pd.DataFrame()
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    if "is_anomaly" in out.columns:
        clean = out[~out["is_anomaly"].fillna(False)]
        if not clean.empty:
            out = clean
    return out.sort_values("date").reset_index(drop=True)


def _period_highs_from_ohlc(df: pd.DataFrame) -> dict[str, float | None]:
    if df.empty:
        return {
            "current": None,
            "week_high": None,
            "month_high": None,
            "year_high": None,
            "day_change_pct": None,
        }
    end = df["date"].max()

    def _max_high(since: pd.Timestamp) -> float | None:
        sl = df[df["date"] >= since]
        if sl.empty:
            return None
        return float(sl["high"].max())

    current = float(df.iloc[-1]["close"])
    day_change_pct: float | None = None
    if len(df) >= 2:
        prev = float(df.iloc[-2]["close"])
        if prev:
            day_change_pct = (current / prev - 1.0) * 100.0

    return {
        "current": current,
        "day_high": float(df.iloc[-1]["high"]),
        "week_high": _max_high(end - timedelta(days=7)),
        "month_high": _max_high(end - timedelta(days=31)),
        "year_high": _max_high(end - timedelta(days=365)),
        "day_change_pct": day_change_pct,
    }


def _insights_row(
    symbol: str,
    label: str,
    snapshot: dict[str, float | None],
    periods: dict[str, float | None],
) -> dict:
    current = snapshot.get("current")
    if current is None:
        current = periods.get("current")
    change_pct = snapshot.get("change_pct")
    if change_pct is None:
        change_pct = periods.get("day_change_pct")
    day_high = snapshot.get("high")
    if day_high is None:
        day_high = periods.get("day_high")
    return {
        "symbol": symbol,
        "name": label,
        "current": current,
        "change_pct": change_pct,
        "day_high": day_high,
        "week_high": periods.get("week_high"),
        "month_high": periods.get("month_high"),
        "year_high": periods.get("year_high"),
        "vs_week_high": _pct_below_high(current, periods.get("week_high")),
        "vs_month_high": _pct_below_high(current, periods.get("month_high")),
        "vs_year_high": _pct_below_high(current, periods.get("year_high")),
    }


def _print_insights_market_table(title: str, rows: list[dict]) -> None:
    print(f"\n{'=' * 100}")
    print(f"  {title}")
    print(f"{'=' * 100}\n")
    if not rows:
        print("  No data available.")
        return

    table = pd.DataFrame(rows)
    show = table[
        [
            "symbol",
            "current",
            "change_pct",
            "day_high",
            "week_high",
            "month_high",
            "year_high",
            "vs_week_high",
            "vs_month_high",
            "vs_year_high",
        ]
    ].copy()
    show["current"] = show["current"].map(lambda v: _fmt_insight_num(v))
    show["change_pct"] = show["change_pct"].map(_fmt_insight_pct)
    show["day_high"] = show["day_high"].map(lambda v: _fmt_insight_num(v))
    show["week_high"] = show["week_high"].map(lambda v: _fmt_insight_num(v))
    show["month_high"] = show["month_high"].map(lambda v: _fmt_insight_num(v))
    show["year_high"] = show["year_high"].map(lambda v: _fmt_insight_num(v))
    for col in ("vs_week_high", "vs_month_high", "vs_year_high"):
        show[col] = show[col].map(_fmt_insight_pct)
    show = show.rename(
        columns={
            "change_pct": "day_chg%",
            "day_high": "day_high",
            "week_high": "wk_high",
            "month_high": "mo_high",
            "year_high": "yr_high",
            "vs_week_high": "vs_wk_hi",
            "vs_month_high": "vs_mo_hi",
            "vs_year_high": "vs_yr_hi",
        }
    )
    print(show.to_string(index=False))
    print(
        "\n  vs_* columns: how far current level is below that period's high "
        "(0% = at the high, negative = below)."
    )


def _sector_performance_insights() -> tuple[pd.DataFrame, pd.DataFrame]:
    scr = psxdata.screener()
    sym_meta = psxdata.symbols()
    if scr.empty or sym_meta.empty:
        return pd.DataFrame(), pd.DataFrame()

    merged = scr.merge(
        sym_meta[["symbol", "sector_name"]],
        on="symbol",
        how="left",
    )
    merged = merged[merged["sector_name"].notna()]
    merged = merged[merged["sector_name"].astype(str).str.len() > 0]

    stock_stats = (
        merged.groupby("sector_name", as_index=False)
        .agg(
            stocks=("symbol", "count"),
            avg_day_chg=("change_pct", "mean"),
            avg_1y_chg=("change_1y_pct", "mean"),
            advancers=("change_pct", lambda s: int((s > 0).sum())),
            decliners=("change_pct", lambda s: int((s < 0).sum())),
        )
        .sort_values("avg_day_chg", ascending=False)
    )
    stock_stats = stock_stats[stock_stats["stocks"] >= 3].reset_index(drop=True)

    psx_sectors = psxdata.sectors()
    breadth = pd.DataFrame()
    if not psx_sectors.empty and "sector_name" in psx_sectors.columns:
        breadth = psx_sectors.copy()
        total = breadth["advance"].fillna(0) + breadth["decline"].fillna(0)
        breadth["breadth_pct"] = (
            breadth["advance"].fillna(0) / total.replace(0, float("nan")) * 100.0
        )
        breadth = breadth.sort_values("breadth_pct", ascending=False).reset_index(
            drop=True
        )

    return stock_stats, breadth


def _print_sector_insights(
    stock_stats: pd.DataFrame, breadth: pd.DataFrame
) -> None:
    print(f"\n{'=' * 100}")
    print("  Sector pulse — which areas are doing well?")
    print(f"{'=' * 100}\n")

    if stock_stats.empty:
        print("  Could not compute sector moves from the screener.")
        return

    print("  By average stock move today (sectors with 3+ listed names):\n")
    leaders = stock_stats.head(8).copy()
    laggards = stock_stats.tail(5).sort_values("avg_day_chg").copy()

    def _display(sec_df: pd.DataFrame) -> pd.DataFrame:
        out = sec_df[
            ["sector_name", "stocks", "avg_day_chg", "avg_1y_chg", "advancers", "decliners"]
        ].copy()
        out["avg_day_chg"] = out["avg_day_chg"].map(lambda v: _fmt_insight_pct(v))
        out["avg_1y_chg"] = out["avg_1y_chg"].map(lambda v: _fmt_insight_pct(v))
        return out.rename(
            columns={
                "sector_name": "sector",
                "avg_day_chg": "avg_today",
                "avg_1y_chg": "avg_1y",
            }
        )

    print("  Leaders (highest average day change):")
    print(_display(leaders).to_string(index=False))
    print("\n  Laggards (lowest average day change):")
    print(_display(laggards).to_string(index=False))

    if not breadth.empty:
        print("\n  PSX sector breadth (% of names up vs down in each sector today):\n")
        bshow = breadth.head(8)[
            ["sector_name", "advance", "decline", "unchanged", "breadth_pct"]
        ].copy()
        bshow["breadth_pct"] = bshow["breadth_pct"].map(
            lambda v: _fmt_insight_pct(v) if pd.notna(v) else "—"
        )
        bshow = bshow.rename(columns={"sector_name": "sector", "breadth_pct": "pct_up"})
        print(bshow.to_string(index=False))

    top = stock_stats.iloc[0]
    print(
        f"\n  Snapshot: strongest sector by average move today is "
        f"{top['sector_name']} ({top['avg_day_chg']:+.2f}% across {int(top['stocks'])} names)."
    )


def _snapshot_by_symbol() -> dict[str, dict[str, float | None]]:
    snapshot_df = _fetch_indices_market_snapshot()
    snap_by_symbol: dict[str, dict[str, float | None]] = {}
    if snapshot_df.empty:
        return snap_by_symbol
    for _, row in snapshot_df.iterrows():
        sym = str(row["symbol"]).upper()
        snap_by_symbol[sym] = {
            "current": row.get("current"),
            "change_pct": row.get("change_pct"),
            "high": row.get("high"),
            "low": row.get("low"),
        }
    return snap_by_symbol


def _insights_for_index(name: str, snap_by_symbol: dict[str, dict[str, float | None]]) -> dict:
    print(f"  Index history: {name}...")
    periods = _period_highs_from_ohlc(_historical_ohlc(name))
    snap = snap_by_symbol.get(name, {})
    return _insights_row(name, name, snap, periods)


def _insights_for_etf(sym: str, label: str) -> dict:
    print(f"  ETF history: {sym}...")
    periods = _period_highs_from_ohlc(_historical_ohlc(sym))
    return _insights_row(sym, label, {}, periods)


def choose_insights_scope() -> str:
    print("\nWhat insights do you want?\n")
    print("  1. Full snapshot (all indices, all ETFs, sector pulse)")
    print("  2. Selected PSX indices (one or more, not all)")
    print("  3. Selected ETFs (one or more, not all)")
    print("  4. Custom mix (any indices and/or ETFs you choose)")
    print("  5. Sector pulse only (screener — no index/ETF history)")
    print()
    while True:
        choice = prompt("Select [1]: ") or "1"
        if choice in ("1", "full", "all", "a"):
            return "full"
        if choice in ("2", "index", "indices", "i"):
            return "indices"
        if choice in ("3", "etf", "etfs", "e"):
            return "etfs"
        if choice in ("4", "mix", "custom", "both", "m"):
            return "mix"
        if choice in ("5", "sectors", "sector", "s"):
            return "sectors"
        print("  Enter 1, 2, 3, 4, or 5.")


def run_insights_flow() -> None:
    scope = choose_insights_scope()
    print("\nPSX Insights — live PSX data")

    if scope == "sectors":
        print("Fetching sector screener and breadth...\n")
        stock_stats, breadth = _sector_performance_insights()
        _print_sector_insights(stock_stats, breadth)
        return

    if scope in ("indices", "etfs", "mix"):
        index_names: list[str] = []
        etf_picks: list[tuple[str, str]] = []
        if scope in ("indices", "mix"):
            index_names = choose_indices_multi(
                allow_empty=(scope == "mix"),
            )
        if scope in ("etfs", "mix"):
            etf_picks = choose_etfs_multi(
                allow_empty=(scope == "mix"),
            )
        if scope == "mix" and not index_names and not etf_picks:
            print("  No indices or ETFs selected.")
            return

        print("\nFetching levels and period highs for your selection...\n")
        snap_by_symbol = _snapshot_by_symbol() if index_names else {}

        index_rows = [
            _insights_for_index(name, snap_by_symbol) for name in index_names
        ]
        etf_rows = [
            _insights_for_etf(sym, label) for sym, label in etf_picks
        ]

        if index_rows:
            title = "PSX indices — selected (levels and period highs)"
            if len(index_rows) == 1:
                title = f"PSX index — {index_rows[0]['symbol']} (levels and period highs)"
            _print_insights_market_table(title, index_rows)
        if etf_rows:
            title = "PSX ETFs — selected (levels and period highs)"
            if len(etf_rows) == 1:
                title = f"PSX ETF — {etf_rows[0]['symbol']} (levels and period highs)"
            _print_insights_market_table(title, etf_rows)
        return

    print("Fetching market snapshot and historical highs (this may take a minute)...\n")
    snap_by_symbol = _snapshot_by_symbol()

    index_rows: list[dict] = []
    for name in INDEX_NAMES:
        index_rows.append(_insights_for_index(name, snap_by_symbol))

    etf_rows: list[dict] = []
    for _, etf in list_etf_symbols().iterrows():
        sym = str(etf["symbol"]).upper()
        etf_rows.append(_insights_for_etf(sym, str(etf["name"])))

    stock_stats, breadth = _sector_performance_insights()

    _print_insights_market_table("PSX indices — levels and period highs", index_rows)
    _print_insights_market_table("PSX ETFs — prices and period highs", etf_rows)
    _print_sector_insights(stock_stats, breadth)


def _local_state_enabled() -> bool:
    return os.environ.get("PSX_SIP_NO_LOCAL_STATE", "").strip().lower() not in (
        "1",
        "yes",
        "true",
    )


def _user_state_file() -> str:
    return os.path.join(os.path.expanduser("~"), ".psx-sip", "user_state.json")


def load_user_state() -> dict:
    if not _local_state_enabled():
        return {}
    path = _user_state_file()
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def save_user_state(**fields: str) -> None:
    if not _local_state_enabled():
        return
    state = load_user_state()
    state.update(fields)
    state["updated_at"] = _utc_now_iso()
    os.makedirs(os.path.dirname(_user_state_file()), exist_ok=True)
    with open(_user_state_file(), "w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=2)


def print_usage_tips() -> None:
    print(
        """
Quick reference (type h or ? at any menu):
  • Index & ETF (menus 1 & 2): new plan, top-up, scope, exclusions, snapshots.
  • Next SIP: pick 1 or 2 → SIP mode 2 Top-up → enter CSV path (any name).
  • Match check (menu 4): Index or ETF → CSV or SYMBOL:qty → resemblance %.
  • Paths: relative (from cwd), absolute, or ~/file.csv — all work.
  • Env: PSX_SIP_HOLDINGS=/path/to.csv skips the load menu in top-up.
  • Lost file? Top-up menu 2 (plan CSV) or 3 (SYMBOL:qty, e.g. FFC:1500).
  • Local reminder only: ~/.psx-sip/user_state.json (last file path). Set
    PSX_SIP_NO_LOCAL_STATE=1 to disable. Nothing is sent to the internet.
""".strip()
    )


def print_startup_reminders() -> None:
    print("  Menus: type h or ? for shortcuts.")
    if not _local_state_enabled():
        return
    state = load_user_state()
    last = state.get("last_holdings_path", "")
    if last and os.path.isfile(last):
        sid = state.get("last_source_id", "")
        tag = f" ({sid})" if sid else ""
        print(f"  Last holdings file{tag}: {last}")
        print(f"  Next top-up: menu 2, or PSX_SIP_HOLDINGS={last!r}")


def _is_help_choice(choice: str) -> bool:
    return choice.strip().lower() in ("h", "?", "help")


def choose_product_type() -> str:
    print("\nWhat would you like to plan?\n")
    print(
        "  1. PSX Index SIP (new / top-up / scope — same features as ETF)"
    )
    print(
        "  2. ETF SIP (new / top-up / scope — creation-unit basket)"
    )
    print("  3. Insights (indices, ETFs, sector leaders — no SIP plan)")
    print(
        "  4. Compare my holdings to an index/ETF "
        "(CSV or shares — how close am I?)"
    )
    print("  h. Help (shortcuts)")
    print()
    while True:
        choice = prompt("Select [1]: ") or "1"
        if _is_help_choice(choice):
            print_usage_tips()
            continue
        if choice in ("1", "index", "i"):
            return "index"
        if choice in ("2", "etf", "e"):
            return "etf"
        if choice in ("3", "insights", "insight"):
            return "insights"
        if choice in ("4", "compare", "match", "align", "holdings"):
            return "align"
        print("  Enter 1, 2, 3, 4, or h.")


def prompt_return_to_menu() -> bool:
    raw = prompt("\nReturn to main menu? [Y/n]: ").lower()
    return raw not in ("n", "no")


def choose_etf() -> str:
    etfs = list_etf_symbols()
    if etfs.empty:
        print("No ETFs found in PSX symbol list.")
        sys.exit(1)

    print_etf_basket_policy()
    catalog = _load_etf_catalog()

    print("\nAvailable PSX ETFs (sector mix from latest PSX creation unit):\n")
    symbols = etfs["symbol"].astype(str).tolist()
    for i, row in etfs.iterrows():
        sym = str(row["symbol"])
        info = catalog.get(sym, {})
        price_line = info.get("price_line", _format_etf_unit_price(None))
        print(f"  {i + 1:2}. {sym} — {row['name']}  |  {price_line}")
        print(f"      {info.get('line', '')}")
    print()
    valid = set(symbols)
    while True:
        raw = prompt("Select ETF (number or symbol, e.g. MZNPETF): ")
        if raw.isdigit():
            idx = int(raw)
            if 1 <= idx <= len(symbols):
                return symbols[idx - 1]
            print("  Invalid selection.")
            continue
        key = raw.upper().strip()
        if key in valid:
            return key
        print(f"  Unknown ETF. Pick 1–{len(symbols)} or a symbol from the list.")


def _parse_etf_equity_basket(soup: BeautifulSoup) -> dict | None:
    modal = soup.select_one(".etfCub__modal")
    if modal is None:
        return None
    title = modal.find("h1")
    title_text = title.get_text(" ", strip=True) if title else ""
    unit_match = _ETF_BASKET_UNIT_RE.search(title_text)
    basket_units = float(_parse_num(unit_match.group(1))) if unit_match else 10_000.0

    headers = [th.get_text(strip=True) for th in modal.select("thead th")]
    if headers and "Component Asset" in " ".join(headers):
        return {"kind": "treasury", "title": title_text}

    h3_text = modal.find("h3").get_text(" ", strip=True) if modal.find("h3") else ""
    cash_pkr = 0.0
    cash_pct: float | None = None
    m_cash = _ETF_CASH_RE.search(h3_text)
    if m_cash:
        cash_pkr = _parse_num(m_cash.group(1))
    m_pct = _ETF_CASH_PCT_RE.search(h3_text)
    if m_pct:
        cash_pct = _parse_num(m_pct.group(1))

    as_of = ""
    date_match = re.search(r"\b([A-Za-z]{3}\s+\d{1,2},\s+\d{4})\b", h3_text)
    if date_match:
        as_of = date_match.group(1)

    holdings: list[dict] = []
    for tr in modal.select("tbody tr"):
        cells = tr.find_all("td")
        if len(cells) < 3:
            continue
        symbol = cells[0].get_text(strip=True).upper()
        name = cells[1].get_text(strip=True)
        shares = _parse_num(cells[2].get_text(strip=True))
        holdings.append({"symbol": symbol, "name": name, "basket_shares": shares})

    if not holdings:
        return None

    return {
        "kind": "equity",
        "title": title_text,
        "basket_units": basket_units,
        "cash_pkr": cash_pkr,
        "cash_pct": cash_pct,
        "as_of": as_of,
        "holdings": holdings,
    }


def _print_treasury_etf_basket(etf_symbol: str, basket: dict) -> None:
    print(f"\nFetching {etf_symbol} basket from PSX...")
    soup = BeautifulSoup(basket["page_html"], "html.parser")
    modal = soup.select_one(".etfCub__modal")
    print(f"\n{'=' * 72}")
    print(f"  {etf_symbol} — {basket.get('title', 'underlying basket')}")
    print(f"{'=' * 72}\n")
    if modal is None:
        print("No basket data on PSX for this ETF.")
        sys.exit(1)
    rows = []
    for tr in modal.select("tbody tr"):
        cells = tr.find_all("td")
        if len(cells) < 4:
            continue
        rows.append(
            {
                "component": cells[0].get_text(strip=True),
                "description": cells[1].get_text(strip=True),
                "asset_value": cells[2].get_text(strip=True),
                "weight": cells[3].get_text(strip=True),
            }
        )
    if rows:
        print(pd.DataFrame(rows).to_string(index=False))
    print(
        "\nThis ETF tracks fixed-income / treasury components, not listed equities. "
        "Use the ETF symbol on PSX to invest; this tool's share SIP math applies "
        "to equity baskets only."
    )
    sys.exit(0)


def load_etf_frame(etf_symbol: str) -> tuple[pd.DataFrame, dict]:
    etf_symbol = etf_symbol.upper()
    if etf_symbol in _etf_frame_cache:
        return _etf_frame_cache[etf_symbol]

    catalog = _etf_catalog or {}
    entry = catalog.get(etf_symbol)
    if entry and entry.get("kind") == "treasury":
        _print_treasury_etf_basket(
            etf_symbol,
            {**entry["parsed"], "page_html": entry["html"]},
        )

    print(f"\nFetching {etf_symbol} creation unit from PSX...")
    parsed, html = _fetch_etf_parsed(etf_symbol)
    scr, price_col = _market_price_index()
    if parsed is None:
        print(f"No underlying equity basket found for {etf_symbol}.")
        sys.exit(1)
    if parsed["kind"] == "treasury":
        _print_treasury_etf_basket(etf_symbol, {**parsed, "page_html": html})

    sector_by_symbol = _sector_by_symbol_map()
    try:
        df, meta = _build_etf_frame_from_parsed(
            etf_symbol,
            parsed,
            scr,
            sector_by_symbol,
            price_col,
        )
    except ValueError:
        print("No priced holdings; cannot build ETF allocation.")
        sys.exit(1)

    if meta.get("etf_unit_price") is None:
        meta = {
            **meta,
            "etf_unit_price": _resolve_etf_unit_price(
                etf_symbol, html, scr, price_col
            ),
        }
    _etf_frame_cache[etf_symbol] = (df, meta)
    return df, meta


def print_etf_overview(df: pd.DataFrame, etf_symbol: str, meta: dict) -> None:
    print(f"\n{'=' * 72}")
    print(f"  {etf_symbol} — underlying equities (PSX creation unit weights)")
    unit_price = meta.get("etf_unit_price")
    if unit_price is not None:
        print(f"  {_format_etf_unit_price(unit_price)} (PSX quote)")
    if meta.get("as_of"):
        print(f"  Basket as of {meta['as_of']} ({meta.get('title', '')})")
    print(f"{'=' * 72}\n")

    show = df.copy()
    show.insert(0, "rank", range(1, len(show) + 1))
    display = show[
        ["rank", "symbol", "name", "sector_name", "price", "idx_weight"]
    ].copy()
    display["price"] = display["price"].map(lambda p: f"{p:,.2f}")
    display["idx_weight"] = display["idx_weight"].map(lambda w: f"{w:.2f}%")
    print(display.to_string(index=False))
    equity_w = df["idx_weight"].sum()
    print(
        f"\nEquity portion: {equity_w:.2f}% of basket | "
        f"Cash component: {meta['cash_pct']:.2f}% "
        f"(Rs. {meta['cash_pkr']:,.2f} per {meta['basket_units']:,.0f} ETF units)"
    )
    print(
        f"Holdings: {len(df)} stocks | "
        f"Weight sum (equities + cash in basket): {equity_w + meta['cash_pct']:.2f}%"
    )

    sectors = sector_weight_table(df)
    print(f"\n{'=' * 72}")
    print(f"  {etf_symbol} — sector weights (equity portion only)")
    print(f"{'=' * 72}\n")
    sec_display = sectors.copy()
    sec_display.insert(0, "rank", range(1, len(sec_display) + 1))
    sec_display["sector_weight"] = sec_display["sector_weight"].map(
        lambda w: f"{w:.2f}%"
    )
    sec_display = sec_display.rename(
        columns={
            "sector_name": "sector",
            "sector_weight": "weight",
            "stocks": "#stocks",
            "top_symbol": "heaviest_symbol",
        }
    )
    print(sec_display.to_string(index=False))
    print(
        f"\nNote: SIP amounts below target the equity slice only; "
        f"~{meta['cash_pct']:.2f}% of a real creation unit is cash."
    )
    print(
        "Basket composition changes when the AMC publishes a new creation unit "
        f"(see date above); prices refresh daily."
    )


def _read_prompt_line(text: str) -> str:
    """Read one line; use the controlling TTY when stdin is a pipe (curl | python -)."""
    sys.stdout.write(text)
    sys.stdout.flush()
    if sys.stdin.isatty():
        return sys.stdin.readline().strip()
    with open("/dev/tty", encoding="utf-8") as tty:
        return tty.readline().strip()


def prompt(text: str) -> str:
    try:
        return _read_prompt_line(text)
    except (EOFError, KeyboardInterrupt, OSError):
        print("\nCancelled.")
        sys.exit(0)


def prompt_positive_float(label: str) -> float:
    while True:
        raw = prompt(f"{label}: ")
        try:
            value = float(raw.replace(",", ""))
        except ValueError:
            print("  Enter a valid number.")
            continue
        if value <= 0:
            print("  Amount must be greater than zero.")
            continue
        return value


def prompt_non_negative_float(label: str, default: float) -> float:
    raw = prompt(f"{label} [{default}]: ")
    if not raw:
        return default
    try:
        value = float(raw.replace(",", ""))
    except ValueError:
        print(f"  Using default {default}%.")
        return default
    if value < 0:
        print(f"  Using default {default}%.")
        return default
    return value


_INDEX_ALIASES = {
    "KSE100": "KSE100",
    "KSE100PR": "KSE100PR",
    "KSE30": "KSE30",
    "KMI30": "KMI30",
}


def _resolve_index_token(token: str) -> str | None:
    key = token.upper().replace("-", "").replace(" ", "")
    candidate = _INDEX_ALIASES.get(key, key)
    if candidate in INDEX_NAMES:
        return candidate
    return None


def _parse_index_selection(raw: str) -> list[str]:
    selected: list[str] = []
    seen: set[str] = set()
    parts = [p.strip() for p in raw.replace(";", ",").split(",") if p.strip()]
    for part in parts:
        if part.isdigit():
            idx = int(part)
            if 1 <= idx <= len(INDEX_NAMES):
                name = INDEX_NAMES[idx - 1]
            else:
                print(f"  Skipping invalid index number: {idx}")
                continue
        else:
            name = _resolve_index_token(part)
            if not name:
                print(f"  Unknown index: {part}")
                continue
        if name not in seen:
            selected.append(name)
            seen.add(name)
    return selected


def choose_indices_multi(*, allow_empty: bool = False) -> list[str]:
    print("\nAvailable PSX indices:\n")
    for i, name in enumerate(INDEX_NAMES, start=1):
        print(f"  {i:2}. {name}")
    print(
        "\n  Tip: comma-separate for several (e.g. 1,3,6 or KSE100,KMI30)."
    )
    while True:
        raw = prompt("Select indices (numbers or names): ")
        picked = _parse_index_selection(raw)
        if picked:
            print(f"  Selected {len(picked)} index(es): {', '.join(picked)}")
            return picked
        if allow_empty and not raw.strip():
            return []
        print("  Pick at least one index (or leave blank only in a custom mix).")


def choose_etfs_multi(*, allow_empty: bool = False) -> list[tuple[str, str]]:
    etfs = list_etf_symbols()
    if etfs.empty:
        print("No ETFs found in PSX symbol list.")
        return []

    print("\nAvailable PSX ETFs:\n")
    symbols: list[str] = []
    names_by_symbol: dict[str, str] = {}
    for i, row in etfs.iterrows():
        sym = str(row["symbol"]).upper()
        symbols.append(sym)
        names_by_symbol[sym] = str(row["name"])
        print(f"  {len(symbols):2}. {sym} — {row['name']}")
    valid = set(symbols)
    print(
        "\n  Tip: comma-separate for several (e.g. 1,2 or MZNPETF,UBLPETF)."
    )
    while True:
        raw = prompt("Select ETFs (numbers or symbols): ")
        if allow_empty and not raw.strip():
            return []

        picked_syms: list[str] = []
        seen: set[str] = set()
        parts = [p.strip() for p in raw.replace(";", ",").split(",") if p.strip()]
        for part in parts:
            if part.isdigit():
                idx = int(part)
                if 1 <= idx <= len(symbols):
                    sym = symbols[idx - 1]
                else:
                    print(f"  Skipping invalid ETF number: {idx}")
                    continue
            else:
                sym = part.upper()
                if sym not in valid:
                    print(f"  Unknown ETF: {part}")
                    continue
            if sym not in seen:
                picked_syms.append(sym)
                seen.add(sym)

        if picked_syms:
            print(f"  Selected {len(picked_syms)} ETF(s): {', '.join(picked_syms)}")
            return [(s, names_by_symbol[s]) for s in picked_syms]
        print("  Pick at least one ETF (or leave blank only in a custom mix).")


def choose_index() -> str:
    print("\nAvailable PSX indices:\n")
    for i, name in enumerate(INDEX_NAMES, start=1):
        print(f"  {i:2}. {name}")
    print()
    while True:
        raw = prompt("Select index (number or name, e.g. KMI30): ")
        if raw.isdigit():
            idx = int(raw)
            if 1 <= idx <= len(INDEX_NAMES):
                return INDEX_NAMES[idx - 1]
            print("  Invalid selection.")
            continue
        name = _resolve_index_token(raw)
        if name:
            return name
        print(f"  Unknown index. Pick 1–{len(INDEX_NAMES)} or a name from the list.")


def parse_price(value) -> float:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        raise ValueError("missing price")
    if isinstance(value, str):
        cleaned = value.replace(",", "").strip()
        if not cleaned or cleaned == "-":
            raise ValueError("missing price")
        return float(cleaned)
    return float(value)


def load_index_frame(index_name: str) -> tuple[pd.DataFrame, dict]:
    print(f"\nFetching {index_name} constituents from PSX (cached when available)...")
    raw = psxdata.indices(index_name)
    if raw is None or raw.empty:
        print(f"No data returned for index {index_name}.")
        sys.exit(1)

    symbols_meta = psxdata.symbols()
    if "sector_name" in symbols_meta.columns:
        raw = raw.merge(
            symbols_meta[["symbol", "sector_name"]],
            on="symbol",
            how="left",
        )
    else:
        raw["sector_name"] = ""

    parsed_holdings: list[dict] = []
    missing_price_symbols: list[str] = []
    priced_rows: list[dict] = []

    for _, row in raw.iterrows():
        sym = str(row["symbol"]).upper()
        name = str(row.get("name", sym))
        parsed_holdings.append(
            {"symbol": sym, "name": name, "basket_shares": None}
        )
        weight = pd.to_numeric(row.get("idx_weight"), errors="coerce")
        try:
            price = parse_price(row["current"])
        except ValueError:
            missing_price_symbols.append(sym)
            continue
        if weight is None or (isinstance(weight, float) and math.isnan(weight)):
            missing_price_symbols.append(sym)
            continue
        priced_rows.append(
            {
                "symbol": sym,
                "name": name,
                "sector_name": str(row.get("sector_name", "") or "UNKNOWN"),
                "price": float(price),
                "idx_weight": float(weight),
            }
        )

    if missing_price_symbols:
        print(
            f"  Warning: no live price/weight for {len(missing_price_symbols)} "
            f"constituent(s): {', '.join(missing_price_symbols[:12])}"
            + (" …" if len(missing_price_symbols) > 12 else "")
        )

    if not priced_rows:
        print(f"No priced constituents for index {index_name}.")
        sys.exit(1)

    df = pd.DataFrame(priced_rows).sort_values("idx_weight", ascending=False)
    df["sector_name"] = df["sector_name"].fillna("UNKNOWN").astype(str)
    meta = {
        "index_name": index_name,
        "as_of": "",
        "title": f"{index_name} index constituents",
        "parsed_holdings": parsed_holdings,
        "missing_price_symbols": missing_price_symbols,
    }
    return df.reset_index(drop=True), meta


def sector_weight_table(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for sector, grp in df.groupby("sector_name"):
        top = grp.loc[grp["idx_weight"].idxmax()]
        rows.append(
            {
                "sector_name": sector,
                "sector_weight": float(grp["idx_weight"].sum()),
                "stocks": len(grp),
                "top_symbol": top["symbol"],
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values("sector_weight", ascending=False)
        .reset_index(drop=True)
    )


def print_overview_tables(df: pd.DataFrame, index_name: str) -> None:
    print(f"\n{'=' * 72}")
    print(f"  {index_name} — companies (sorted by index weight)")
    print(f"{'=' * 72}\n")

    companies = df.copy()
    companies.insert(0, "rank", range(1, len(companies) + 1))
    display = companies[
        ["rank", "symbol", "name", "sector_name", "price", "idx_weight"]
    ].copy()
    display["price"] = display["price"].map(lambda p: f"{p:,.2f}")
    display["idx_weight"] = display["idx_weight"].map(lambda w: f"{w:.2f}%")
    print(display.to_string(index=False))
    print(
        f"\nTotal: {len(companies)} companies | "
        f"Index weight sum: {df['idx_weight'].sum():.2f}%"
    )

    sectors = sector_weight_table(df)
    print(f"\n{'=' * 72}")
    print(f"  {index_name} — sector weights")
    print(f"{'=' * 72}\n")
    sec_display = sectors.copy()
    sec_display.insert(0, "rank", range(1, len(sec_display) + 1))
    sec_display["sector_weight"] = sec_display["sector_weight"].map(
        lambda w: f"{w:.2f}%"
    )
    sec_display = sec_display.rename(
        columns={
            "sector_name": "sector",
            "sector_weight": "weight",
            "stocks": "#stocks",
            "top_symbol": "heaviest_symbol",
        }
    )
    print(sec_display.to_string(index=False))


def parse_symbol_list(raw: str, valid: set[str]) -> set[str]:
    if not raw:
        return set()
    tokens = [t.strip().upper() for t in raw.replace(";", ",").split(",") if t.strip()]
    unknown = [t for t in tokens if t not in valid]
    if unknown:
        print(f"  Ignoring unknown symbols: {', '.join(unknown)}")
    return {t for t in tokens if t in valid}


def apply_exclusions(
    df: pd.DataFrame,
    exclude_sectors: set[str],
    exclude_symbols: set[str],
) -> pd.DataFrame:
    out = df.copy()
    if exclude_sectors:
        out = out[~out["sector_name"].str.upper().isin(exclude_sectors)]
    if exclude_symbols:
        out = out[~out["symbol"].isin(exclude_symbols)]
    return out.sort_values("idx_weight", ascending=False).reset_index(drop=True)


def pick_sector_leaders(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.groupby("sector_name")["idx_weight"].idxmax()
    return df.loc[idx].sort_values("idx_weight", ascending=False).reset_index(drop=True)


def pick_top_n(df: pd.DataFrame, n: int) -> pd.DataFrame:
    if n <= 0:
        return df.iloc[0:0].copy()
    return df.head(min(n, len(df))).copy()


def prompt_exclusions(
    df: pd.DataFrame,
    *,
    universe_label: str = "index",
) -> tuple[set[str], set[str]]:
    valid_symbols = set(df["symbol"].astype(str))
    sector_map = {
        s.upper(): s for s in sorted(df["sector_name"].unique(), key=str.casefold)
    }

    print("\n--- Exclusions (optional) ---")
    print("Leave blank to include everything.\n")

    sector_list = list(sector_map.values())
    print(f"Sectors in this {universe_label}:")
    for i, sector in enumerate(sector_list, start=1):
        count = (df["sector_name"] == sector).sum()
        w = df.loc[df["sector_name"] == sector, "idx_weight"].sum()
        print(f"  {i:2}. {sector} ({count} stocks, {w:.2f}% weight)")

    sector_raw = prompt("\nExclude sectors (numbers or names, comma-separated): ")
    exclude_sectors: set[str] = set()
    if sector_raw:
        parts = [p.strip() for p in sector_raw.replace(";", ",").split(",") if p.strip()]
        for part in parts:
            if part.isdigit():
                num = int(part)
                if 1 <= num <= len(sector_list):
                    exclude_sectors.add(sector_list[num - 1].upper())
                else:
                    print(f"  Skipping invalid sector number: {num}")
            else:
                key = part.upper()
                if key in sector_map:
                    exclude_sectors.add(key)
                else:
                    print(f"  Unknown sector: {part}")

    symbol_raw = prompt(
        "Exclude companies by symbol (comma-separated, e.g. TOMCL,ENGRO): "
    )
    exclude_symbols = parse_symbol_list(symbol_raw, valid_symbols)
    return exclude_sectors, exclude_symbols


def prompt_portfolio_scope(
    df: pd.DataFrame,
    benchmark_name: str,
    *,
    universe_label: str = "index",
    weight_label: str = "index",
) -> pd.DataFrame:
    hint = ""
    if universe_label == "index" and benchmark_name == "KMI30":
        hint = " (common for KMI30 SIP: top 5 ≈ 49% index weight, top 10 ≈ 79%)"
    if universe_label == "ETF basket":
        hint = " (partial ETF replication: top names cover most basket weight)"

    print(f"\n--- Portfolio scope ({benchmark_name}) ---{hint}\n")
    print(f"  1. All holdings in the {universe_label}")
    print(f"  2. Top 5 by {weight_label} weight")
    print(f"  3. Top 10 by {weight_label} weight")
    print("  4. Custom top N")
    print("  5. One stock per sector (highest weight in each sector)")
    print()

    while True:
        choice = prompt("Select scope [1]: ") or "1"
        if choice not in {"1", "2", "3", "4", "5"}:
            print("  Enter 1–5.")
            continue

        working = df.copy()

        leaders_answer = prompt(
            "Before top-N, keep only the heaviest stock in each sector? [y/N]: "
        ).lower()
        if leaders_answer in ("y", "yes"):
            before = len(working)
            working = pick_sector_leaders(working)
            print(
                f"  Sector leaders: {len(working)} stocks "
                f"(removed {before - len(working)} lower-weight peers)."
            )

        if choice == "1":
            selected = working
            label = "all stocks"
        elif choice == "2":
            selected = pick_top_n(working, 5)
            label = "top 5"
        elif choice == "3":
            selected = pick_top_n(working, 10)
            label = "top 10"
        elif choice == "4":
            while True:
                raw = prompt(f"How many top stocks (1–{len(working)})? ")
                if raw.isdigit():
                    n = int(raw)
                    if 1 <= n <= len(working):
                        selected = pick_top_n(working, n)
                        label = f"top {n}"
                        break
                print("  Enter a valid number in range.")
        else:
            selected = pick_sector_leaders(working)
            label = "sector leaders (one per sector)"

        if selected.empty:
            print("  Selection is empty; try again.")
            continue

        raw_w = selected["idx_weight"].sum()
        print(
            f"\nSelected {label}: {len(selected)} stocks covering "
            f"{raw_w:.2f}% of {weight_label} weight (before re-normalizing for SIP)."
        )
        show = selected[["symbol", "sector_name", "price", "idx_weight"]].copy()
        show["price"] = show["price"].map(lambda p: f"{p:,.2f}")
        show["idx_weight"] = show["idx_weight"].map(lambda w: f"{w:.2f}%")
        print(show.to_string(index=False))
        confirm = prompt("\nUse this selection? [Y/n]: ").lower()
        if confirm in ("", "y", "yes"):
            return selected.reset_index(drop=True)
        print("  Let's pick again.\n")


def _initial_share_counts(
    target_pkr: pd.Series, effective_price: pd.Series, use_board_lot: bool
) -> pd.Series:
    if use_board_lot:
        lots = (target_pkr / (effective_price * BOARD_LOT)).apply(math.floor)
        return (lots * BOARD_LOT).astype(int)
    return (target_pkr / effective_price).apply(math.floor).astype(int)


def _redeploy_cash_to_weights(
    plan: pd.DataFrame,
    investment_pkr: float,
    use_board_lot: bool,
) -> pd.DataFrame:
    """Spend remaining PKR on whole shares, favoring names furthest below target weight."""
    out = plan.copy()
    lot = BOARD_LOT if use_board_lot else 1
    shares = out["shares"].to_numpy(dtype=float)
    prices = out["effective_price"].to_numpy(dtype=float)
    targets = out["target_pkr"].to_numpy(dtype=float)

    def invested_total() -> float:
        return float((shares * prices).sum())

    while True:
        cash = investment_pkr - invested_total()
        if cash <= 0:
            break

        best_row: int | None = None
        best_deficit = -1.0
        for i in range(len(out)):
            cost = prices[i] * lot
            if cost > cash + 1e-9:
                continue
            deficit = targets[i] - shares[i] * prices[i]
            if deficit > best_deficit:
                best_deficit = deficit
                best_row = i

        if best_row is None:
            break
        shares[best_row] += lot

    out["shares"] = shares.astype(int)
    return out


def compute_sip_plan(
    df: pd.DataFrame,
    investment_pkr: float,
    use_board_lot: bool,
    cost_buffer_pct: float,
) -> pd.DataFrame:
    if df.empty:
        return df

    total_weight = df["idx_weight"].sum()
    if total_weight <= 0:
        print("Index weights are zero; cannot allocate.")
        sys.exit(1)

    plan = df.copy()
    plan["effective_price"] = plan["price"] * (1.0 + cost_buffer_pct / 100.0)
    plan["weight_pct"] = plan["idx_weight"] / total_weight * 100.0
    plan["target_pkr"] = investment_pkr * (plan["weight_pct"] / 100.0)
    plan["shares"] = _initial_share_counts(
        plan["target_pkr"], plan["effective_price"], use_board_lot
    )
    plan = _redeploy_cash_to_weights(plan, investment_pkr, use_board_lot)

    plan["invested_pkr"] = plan["shares"] * plan["effective_price"]
    total_invested = plan["invested_pkr"].sum()
    if total_invested > 0:
        plan["actual_weight_pct"] = plan["invested_pkr"] / total_invested * 100.0
    else:
        plan["actual_weight_pct"] = 0.0
    plan["leftover_pkr"] = plan["target_pkr"] - plan["invested_pkr"]
    return plan.sort_values("weight_pct", ascending=False).reset_index(drop=True)


def print_plan(
    plan: pd.DataFrame,
    investment_pkr: float,
    use_board_lot: bool,
    cost_buffer_pct: float,
) -> None:
    active = plan[plan["shares"] > 0].copy()
    zero = plan[plan["shares"] == 0]

    mode = f"board lot ({BOARD_LOT} shares)" if use_board_lot else "whole shares"
    print(f"\n=== SIP allocation ({mode}) ===\n")
    print(f"Monthly investment: PKR {investment_pkr:,.2f}")
    print(
        f"Cost buffer: +{cost_buffer_pct:.2f}% on market price "
        f"(used as effective buy price)"
    )
    print(f"Stocks with at least 1 share/lot: {len(active)} / {len(plan)}")

    if active.empty:
        print(
            "\nNo shares could be bought at this investment level with current prices."
        )
        print(
            "Try a higher SIP amount, fewer exclusions, or disable board-lot rounding."
        )
        return

    show = active[
        [
            "symbol",
            "name",
            "sector_name",
            "price",
            "effective_price",
            "weight_pct",
            "actual_weight_pct",
            "target_pkr",
            "shares",
            "invested_pkr",
        ]
    ].copy()
    show["price"] = show["price"].map(lambda p: f"{p:,.2f}")
    show["effective_price"] = show["effective_price"].map(lambda p: f"{p:,.2f}")
    show["weight_pct"] = show["weight_pct"].map(lambda w: f"{w:.2f}%")
    show["actual_weight_pct"] = show["actual_weight_pct"].map(lambda w: f"{w:.2f}%")
    show["target_pkr"] = show["target_pkr"].map(lambda v: f"{v:,.2f}")
    show["invested_pkr"] = show["invested_pkr"].map(lambda v: f"{v:,.2f}")
    show = show.rename(columns={"effective_price": "buy_price_buffered"})
    print(show.to_string(index=False))

    total_invested = plan["invested_pkr"].sum()
    cash_left = investment_pkr - total_invested
    print(f"\nTotal invested (this month, incl. buffer): PKR {total_invested:,.2f}")
    print(f"Unallocated cash (keep for next SIP / fees): PKR {cash_left:,.2f}")

    lot = BOARD_LOT if use_board_lot else 1
    min_cost = (plan["effective_price"] * lot).min()
    if cash_left > 0.01:
        pct = cash_left / investment_pkr * 100
        print(
            f"\nNote: {pct:.1f}% remains because only whole "
            f"{'board lots' if use_board_lot else 'shares'} can be bought "
            f"(cheapest next purchase ≈ PKR {min_cost:,.2f} at buffered price)."
        )

    if not zero.empty:
        print(
            f"\n{len(zero)} names still have zero shares "
            f"(could not reach even one {'lot' if use_board_lot else 'share'} "
            f"within your budget at buffered prices)."
        )


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _parse_snapshot_header_line(line: str) -> tuple[str, str] | None:
    line = line.strip()
    if not line.startswith("#"):
        return None
    body = line.lstrip("#").strip()
    if "=" not in body:
        return None
    key, _, val = body.partition("=")
    return key.strip(), val.strip()


def build_holdings_snapshot_rows(
    plan: pd.DataFrame,
    live_df: pd.DataFrame,
    *,
    source_type: str,
    source_id: str,
    parsed_holdings: list[dict] | None,
    missing_price_symbols: list[str],
    prior_shares: dict[str, int] | None,
) -> pd.DataFrame:
    """One row per basket line; includes names we could not price on PSX."""
    missing_set = {s.upper() for s in missing_price_symbols}
    live_by_sym = live_df.set_index("symbol") if not live_df.empty else pd.DataFrame()
    plan_by_sym = plan.set_index("symbol") if not plan.empty else pd.DataFrame()
    prior = prior_shares or {}

    def _row(
        symbol: str,
        name: str,
        sector_name: str,
        basket_shares: float | None,
        price: float | None,
        idx_weight: float | None,
        data_status: str,
        notes: str,
    ) -> dict:
        sym = symbol.upper()
        shares_this = int(plan_by_sym.loc[sym, "shares"]) if sym in plan_by_sym.index else 0
        invested_this = (
            float(plan_by_sym.loc[sym, "invested_pkr"]) if sym in plan_by_sym.index else 0.0
        )
        if sym in prior:
            shares_held = int(prior[sym]) + shares_this
        else:
            shares_held = shares_this
        return {
            "symbol": sym,
            "name": name,
            "sector_name": sector_name or "",
            "basket_shares": basket_shares,
            "price": price,
            "idx_weight_pct": idx_weight,
            "shares_held": shares_held,
            "shares_this_run": shares_this,
            "invested_pkr_this_run": invested_this,
            "data_status": data_status,
            "notes": notes,
        }

    rows: list[dict] = []
    if parsed_holdings:
        for h in parsed_holdings:
            sym = str(h["symbol"]).upper()
            name = str(h.get("name", ""))
            b_sh = float(h.get("basket_shares", 0))
            if sym in missing_set:
                sector = ""
                rows.append(
                    _row(
                        sym,
                        name,
                        sector,
                        b_sh,
                        None,
                        None,
                        "no_live_price",
                        "Could not fetch live price from PSX screener",
                    )
                )
                continue
            if sym not in live_by_sym.index:
                rows.append(
                    _row(
                        sym,
                        name,
                        "",
                        b_sh,
                        None,
                        None,
                        "not_in_live_basket",
                        "Listed in creation unit but missing from priced frame",
                    )
                )
                continue
            lr = live_by_sym.loc[sym]
            rows.append(
                _row(
                    sym,
                    str(lr.get("name", name)),
                    str(lr.get("sector_name", "")),
                    b_sh,
                    float(lr["price"]),
                    float(lr["idx_weight"]),
                    "ok",
                    "",
                )
            )
        seen = {r["symbol"] for r in rows}
        for sym, sh in prior.items():
            if sh <= 0 or sym in seen:
                continue
            rows.append(
                _row(
                    sym,
                    "",
                    "",
                    None,
                    None,
                    None,
                    "held_not_in_basket",
                    "Shares from your file; not in current published basket",
                )
            )
        return pd.DataFrame(rows)

    for _, lr in live_df.iterrows():
        sym = str(lr["symbol"]).upper()
        rows.append(
            _row(
                sym,
                str(lr.get("name", "")),
                str(lr.get("sector_name", "")),
                None,
                float(lr["price"]),
                float(lr["idx_weight"]),
                "ok",
                "",
            )
        )
    if prior:
        seen = {r["symbol"] for r in rows}
        for sym, sh in prior.items():
            if sh <= 0 or sym in seen:
                continue
            rows.append(
                _row(
                    sym,
                    "",
                    "",
                    None,
                    None,
                    None,
                    "held_not_in_basket",
                    "Shares from your file; not in current index/scope",
                )
            )
    return pd.DataFrame(rows)


def write_holdings_snapshot(
    path: str,
    table: pd.DataFrame,
    header: dict[str, str],
) -> None:
    lines = [
        f"# psx-sip-holdings snapshot_version={HOLDINGS_SNAPSHOT_VERSION}",
    ]
    for key in sorted(header.keys()):
        val = str(header[key]).replace("\n", " ")
        lines.append(f"# {key}={val}")
    lines.append(
        "# Edit shares_held between runs if you bought outside this planner."
    )
    lines.append(
        "# Pass this file on the next SIP as your last holdings snapshot."
    )
    lines.append(
        "# If you lose this file: top-up mode also accepts an old plan CSV or manual SYMBOL:qty."
    )
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
        table.to_csv(fh, index=False)


def load_holdings_snapshot(path: str) -> tuple[dict[str, str], pd.DataFrame]:
    if not os.path.isfile(path):
        print(f"  File not found: {path}")
        sys.exit(1)
    header: dict[str, str] = {}
    data_lines: list[str] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#"):
                parsed = _parse_snapshot_header_line(line)
                if parsed:
                    header[parsed[0]] = parsed[1]
                continue
            data_lines.append(line)
    if header.get("snapshot_version") not in (None, HOLDINGS_SNAPSHOT_VERSION):
        print(
            f"  Warning: snapshot version {header.get('snapshot_version')} "
            f"(planner expects {HOLDINGS_SNAPSHOT_VERSION})."
        )
    from io import StringIO

    if not data_lines or not data_lines[0].strip():
        print("  Holdings file has no data rows.")
        sys.exit(1)
    df = pd.read_csv(StringIO("".join(data_lines)))
    required = {"symbol", "shares_held"}
    if not required.issubset(df.columns):
        print(f"  Holdings file must include columns: {', '.join(sorted(required))}")
        sys.exit(1)
    df["symbol"] = df["symbol"].astype(str).str.upper()
    df["shares_held"] = pd.to_numeric(df["shares_held"], errors="coerce").fillna(0).astype(int)
    return header, df


def maybe_save_holdings_snapshot(
    plan: pd.DataFrame,
    live_df: pd.DataFrame,
    *,
    source_type: str,
    source_id: str,
    investment_pkr: float,
    cost_buffer_pct: float,
    use_board_lot: bool,
    meta: dict | None,
    prior_shares: dict[str, int] | None,
    default_name: str,
) -> None:
    answer = prompt(
        "\nSave holdings snapshot for your next top-up SIP? [Y/n]: "
    ).lower()
    if answer not in ("", "y", "yes"):
        return
    print("  Save as relative or absolute path (any filename you like).")
    path_raw = prompt(f"Holdings file path [{default_name}]: ") or default_name
    path = resolve_user_path(path_raw)
    parsed_holdings = (meta or {}).get("parsed_holdings")
    missing = (meta or {}).get("missing_price_symbols") or []
    table = build_holdings_snapshot_rows(
        plan,
        live_df,
        source_type=source_type,
        source_id=source_id,
        parsed_holdings=parsed_holdings,
        missing_price_symbols=missing,
        prior_shares=prior_shares,
    )
    snap_header = {
        "planner_version": __version__,
        "generated_at": _utc_now_iso(),
        "source_type": source_type,
        "source_id": source_id,
        "investment_pkr_this_run": f"{investment_pkr:.2f}",
        "cost_buffer_pct": f"{cost_buffer_pct:.2f}",
        "board_lot": "yes" if use_board_lot else "no",
        "basket_as_of": str((meta or {}).get("as_of", "")),
    }
    write_holdings_snapshot(path, table, snap_header)
    abs_path = os.path.abspath(path)
    print(f"Saved holdings snapshot: {abs_path}")
    save_user_state(
        last_holdings_path=abs_path,
        last_source_type=source_type,
        last_source_id=source_id,
    )
    print(
        "  Next run: ETF/Index → 2 Top-up (this file is remembered locally), "
        f"or PSX_SIP_HOLDINGS={abs_path!r}"
    )
    n_bad = (table["data_status"] != "ok").sum() if "data_status" in table.columns else 0
    if n_bad:
        print(
            f"  {n_bad} row(s) have data_status other than ok "
            f"(e.g. no_live_price) — fix or ignore when editing shares_held."
        )


def _benchmark_target_pct(live_df: pd.DataFrame) -> pd.Series:
    live = live_df.copy()
    live["symbol"] = live["symbol"].astype(str).str.upper()
    total = float(live["idx_weight"].sum())
    if total <= 0:
        return pd.Series(dtype=float)
    s = live.set_index("symbol")["idx_weight"] / total
    return s


def report_holdings_alignment(
    prior_shares: dict[str, int],
    live_df: pd.DataFrame,
    source_type: str,
    source_id: str,
) -> None:
    """Compare user share counts to live index/ETF weights (no trades)."""
    target = _benchmark_target_pct(live_df)
    if target.empty:
        print("  No benchmark weights available.")
        return

    live = live_df.copy()
    live["symbol"] = live["symbol"].astype(str).str.upper()
    by_sym = live.set_index("symbol")

    held_rows: list[dict] = []
    orphan_rows: list[dict] = []
    total_value = 0.0

    for sym, sh in prior_shares.items():
        if sh <= 0:
            continue
        sym = sym.upper()
        if sym not in by_sym.index:
            orphan_rows.append({"symbol": sym, "shares": sh})
            continue
        price = float(by_sym.loc[sym, "price"])
        value = sh * price
        total_value += value
        held_rows.append(
            {
                "symbol": sym,
                "name": str(by_sym.loc[sym].get("name", "")),
                "shares": sh,
                "price": price,
                "value_pkr": value,
            }
        )

    if total_value <= 0:
        print("  Could not value any in-basket holdings (check symbols/shares).")
        if orphan_rows:
            print(
                "  Symbols not in basket: "
                + ", ".join(r["symbol"] for r in orphan_rows)
            )
        return

    user_vec = pd.Series(0.0, index=target.index)
    for r in held_rows:
        user_vec[r["symbol"]] = r["value_pkr"] / total_value

    l1 = float((user_vec - target).abs().sum())
    resemblance_pct = max(0.0, (1.0 - l1 / 2.0) * 100.0)
    held_symbols = {r["symbol"] for r in held_rows}
    coverage_weight = float(target.loc[list(held_symbols & set(target.index))].sum()) * 100.0
    overlap_weight = float(pd.concat([user_vec, target], axis=1).min(axis=1).sum()) * 100.0

    print(f"\n{'=' * 72}")
    print(f"  Holdings vs {source_type.upper()} {source_id}")
    print(f"{'=' * 72}")
    print(
        f"\n  Resemblance score: {resemblance_pct:.1f}% "
        f"(100% = same names & weights as benchmark; "
        f"0% = no overlap)"
    )
    print(
        f"  Benchmark weight covered by your names: {coverage_weight:.1f}% "
        f"({len(held_symbols)} of {len(target)} constituents)"
    )
    print(f"  Weight overlap (sum of min weights): {overlap_weight:.1f}%")
    print(f"  Portfolio value marked (held, in basket): PKR {total_value:,.2f}")

    detail = []
    target_pct = target * 100.0
    for r in held_rows:
        sym = r["symbol"]
        your_pct = r["value_pkr"] / total_value * 100.0
        tgt = float(target_pct[sym])
        held_only_total = float(target.loc[list(held_symbols)].sum()) * 100.0
        tgt_on_held = tgt / held_only_total * 100.0 if held_only_total > 0 else 0.0
        detail.append(
            {
                **r,
                "your_pct": your_pct,
                "benchmark_pct": tgt,
                "benchmark_pct_on_held_subset": tgt_on_held,
                "within_holdings_gap": your_pct - tgt_on_held,
            }
        )

    print("\n  Your holdings vs benchmark (in-basket only):\n")
    show = pd.DataFrame(detail)
    disp = show[
        [
            "symbol",
            "shares",
            "value_pkr",
            "your_pct",
            "benchmark_pct",
            "within_holdings_gap",
        ]
    ].copy()
    disp["value_pkr"] = disp["value_pkr"].map(lambda v: f"{v:,.0f}")
    disp["your_pct"] = disp["your_pct"].map(lambda v: f"{v:.2f}%")
    disp["benchmark_pct"] = disp["benchmark_pct"].map(lambda v: f"{v:.2f}%")
    disp["within_holdings_gap"] = disp["within_holdings_gap"].map(
        lambda v: f"{v:+.2f} pp"
    )
    disp = disp.rename(
        columns={
            "benchmark_pct": "bench_%",
            "your_pct": "your_%",
            "within_holdings_gap": "vs_renorm_bench",
        }
    )
    print(disp.sort_values("symbol").to_string(index=False))
    print(
        "\n  vs_renorm_bench: your weight minus benchmark weight re-normalized "
        "to names you hold (+ = overweight within your portfolio)."
    )

    if orphan_rows:
        print("\n  Not in current benchmark (still in your file):")
        for r in orphan_rows:
            print(f"    {r['symbol']}: {r['shares']} shares")

    print_switch_considerations(held_symbols, live_df)
    print(
        "\n  (Review only — the planner does not ask you to confirm each add/sell; "
        "use menu 1 or 2 when you want a SIP plan.)"
    )


def choose_align_benchmark_kind() -> str:
    print("\nCompare your holdings to:\n")
    print("  1. PSX Index")
    print("  2. ETF (creation-unit basket)")
    print()
    while True:
        choice = prompt("Select [1]: ") or "1"
        if choice in ("1", "index", "i"):
            return "index"
        if choice in ("2", "etf", "e"):
            return "etf"
        print("  Enter 1 or 2.")


def _scoped_frame_for_topup(
    prior_shares: dict[str, int],
    live_df: pd.DataFrame,
) -> pd.DataFrame:
    held = {s.upper() for s, q in prior_shares.items() if q > 0}
    live = live_df.copy()
    live["symbol"] = live["symbol"].astype(str).str.upper()
    return live[live["symbol"].isin(held)].reset_index(drop=True)


def run_holdings_align_flow() -> None:
    kind = choose_align_benchmark_kind()
    meta: dict | None = None
    if kind == "index":
        source_id = choose_index()
        live_df, meta = load_index_frame(source_id)
        print_overview_tables(live_df, source_id)
        source_type = "index"
    else:
        source_id = choose_etf()
        live_df, meta = load_etf_frame(source_id)
        print_etf_overview(live_df, source_id, meta)
        source_type = "etf"

    print(
        "\nLoad holdings to compare (holdings snapshot CSV, plan CSV, or manual)."
    )
    prior = choose_topup_holdings_source(
        f"psx_sip_holdings_{source_id.lower()}.csv",
        source_type,
        source_id,
    )
    report_holdings_alignment(prior, live_df, source_type, source_id)

    ans = prompt("\nPlan a top-up SIP from these same holdings? [y/N]: ").lower()
    if ans not in ("y", "yes"):
        return

    scoped = _scoped_frame_for_topup(prior, live_df)
    if scoped.empty:
        print("  No held symbols in live basket; cannot top-up.")
        return
    investment_pkr = prompt_positive_float(
        "\nMonthly SIP investment amount (PKR)"
    )
    _finalize_sip_plan(
        scoped,
        investment_pkr,
        default_csv=f"psx_sip_plan_{source_id.lower()}.csv",
        source_type=source_type,
        source_id=source_id,
        live_df=live_df,
        meta=meta,
        prior_shares=prior,
    )


def print_switch_considerations(
    held_symbols: set[str],
    live_df: pd.DataFrame,
) -> None:
    if live_df.empty:
        return
    missing = live_df[~live_df["symbol"].isin(held_symbols)].copy()
    if missing.empty:
        print("\nYou hold every priced line in the current basket.")
        return
    missing = missing.sort_values("idx_weight", ascending=False)
    covered = live_df.loc[live_df["symbol"].isin(held_symbols), "idx_weight"].sum()
    missing_total = float(missing["idx_weight"].sum())
    print(f"\n--- Consider switching? (you do not hold these basket names) ---")
    print(
        f"  Your holdings cover {covered:.2f}% of current basket equity weight; "
        f"{missing_total:.2f}% is in names you do not hold."
    )
    alerts: list[pd.Series] = []
    top_syms = set(
        live_df.sort_values("idx_weight", ascending=False)
        .head(MISSING_TOP_N_ALERT)["symbol"]
        .astype(str)
    )
    for _, row in missing.iterrows():
        w = float(row["idx_weight"])
        sym = str(row["symbol"])
        reasons: list[str] = []
        if w >= MISSING_WEIGHT_ALERT_PCT:
            reasons.append(f"weight {w:.2f}% ≥ {MISSING_WEIGHT_ALERT_PCT:g}%")
        if sym in top_syms:
            reasons.append(f"in current top {MISSING_TOP_N_ALERT}")
        if reasons:
            alerts.append(row.assign(alert="; ".join(reasons)))
    if missing_total >= MISSING_TOTAL_WEIGHT_ALERT_PCT:
        print(
            f"  Note: combined missing weight {missing_total:.2f}% ≥ "
            f"{MISSING_TOTAL_WEIGHT_ALERT_PCT:g}% — portfolio may be far from full ETF."
        )
    if not alerts:
        print("  No single name crossed the default alert thresholds (edit thresholds in code).")
        show = missing.head(8)[["symbol", "name", "idx_weight"]].copy()
        show["idx_weight"] = show["idx_weight"].map(lambda w: f"{w:.2f}%")
        print("\n  Largest names you do not hold:")
        print(show.to_string(index=False))
        return
    alert_df = pd.DataFrame(alerts)
    show = alert_df[["symbol", "name", "idx_weight", "alert"]].copy()
    show["idx_weight"] = show["idx_weight"].map(lambda w: f"{w:.2f}%")
    print("\n  Flagged for review (not auto-added to this top-up):")
    print(show.to_string(index=False))


def choose_sip_mode(source_id: str = "") -> str:
    print("\nHow should this SIP run?\n")
    print("  1. New plan (pick scope / exclusions as usual)")
    print(
        "  2. Top-up existing holdings only "
        "(snapshot CSV, old plan CSV, or manual entry)"
    )
    print("  h. Help (shortcuts)")
    default_snap = (
        f"psx_sip_holdings_{source_id.lower()}.csv" if source_id else ""
    )
    if default_snap and os.path.isfile(default_snap):
        print(f"\n  → Found ./{default_snap} — option 2 will offer to use it.")
    state = load_user_state()
    last = state.get("last_holdings_path", "")
    if last and os.path.isfile(last):
        print(f"  → Last saved: {last}")
    print()
    while True:
        choice = prompt("Select [1]: ") or "1"
        if _is_help_choice(choice):
            print_usage_tips()
            continue
        if choice in ("1", "new", "fresh", "n"):
            return "fresh"
        if choice in ("2", "topup", "top-up", "holdings", "continue", "c"):
            return "topup"
        print("  Enter 1, 2, or h.")


def resolve_user_path(raw: str) -> str:
    """Expand ~ and resolve relative paths against the current working directory."""
    return os.path.abspath(os.path.expanduser(raw.strip()))


def prompt_existing_csv_path() -> str:
    print("\n  Path can be relative, absolute, or use ~")
    print(f"  Current folder: {os.getcwd()}\n")
    while True:
        raw = prompt("CSV path: ").strip()
        if not raw:
            print("  Enter a path (required), or h for help.")
            continue
        if _is_help_choice(raw):
            print_usage_tips()
            continue
        path = resolve_user_path(raw)
        if os.path.isfile(path):
            return path
        print(f"  Not found: {path}")


def _parse_manual_holdings(raw: str) -> dict[str, int]:
    """SYMBOL:qty pairs, comma/space separated (e.g. FFC:1500, ENGRO:10)."""
    out: dict[str, int] = {}
    if not raw.strip():
        return out
    parts = re.split(r"[,;\s]+", raw.strip())
    for part in parts:
        if not part:
            continue
        if ":" in part:
            sym, _, qty_s = part.partition(":")
        elif "=" in part:
            sym, _, qty_s = part.partition("=")
        else:
            print(f"  Skipping {part!r} — use SYMBOL:qty")
            continue
        sym = sym.strip().upper()
        try:
            qty = int(float(qty_s.strip().replace(",", "")))
        except ValueError:
            print(f"  Skipping {part!r} — invalid quantity")
            continue
        if qty <= 0:
            continue
        out[sym] = out.get(sym, 0) + qty
    return out


def prompt_manual_holdings() -> dict[str, int]:
    print(
        "\nEnter each position as SYMBOL:share_count "
        "(comma-separated). Use broker statement totals.\n"
    )
    while True:
        raw = prompt("Holdings (e.g. FFC:1500, MEBL:2, OGDC:6): ")
        prior = _parse_manual_holdings(raw)
        if prior:
            return prior
        print("  Enter at least one symbol with shares > 0.")


def load_prior_shares_from_plan_csv(path: str) -> dict[str, int]:
    df = pd.read_csv(path)
    if "symbol" not in df.columns:
        print("  Plan CSV must include a 'symbol' column.")
        sys.exit(1)
    df["symbol"] = df["symbol"].astype(str).str.upper()
    if "shares_held" in df.columns:
        qty_col = "shares_held"
    elif "shares" in df.columns:
        qty_col = "shares"
        print(
            "\n  Note: using the plan's 'shares' column as your position. "
            "That is only one month's buys unless you edited the file — "
            "adjust totals if needed before saving a new snapshot."
        )
    else:
        print("  Plan CSV needs 'shares_held' or 'shares'.")
        sys.exit(1)
    df[qty_col] = pd.to_numeric(df[qty_col], errors="coerce").fillna(0).astype(int)
    df = df[df[qty_col] > 0]
    if df.empty:
        print("  No rows with share count > 0 in that plan CSV.")
        sys.exit(1)
    return df.groupby("symbol")[qty_col].sum().astype(int).to_dict()


def _is_holdings_snapshot_file(path: str) -> bool:
    with open(path, encoding="utf-8") as fh:
        for _ in range(5):
            line = fh.readline()
            if not line:
                break
            if line.strip().startswith("# psx-sip-holdings"):
                return True
            if line.strip() and not line.startswith("#"):
                break
    return False


def _load_prior_shares_from_any_csv(
    path: str,
    source_type: str,
    source_id: str,
) -> dict[str, int]:
    kind = os.environ.get("PSX_SIP_HOLDINGS_KIND", "auto").strip().lower()
    use_snapshot = kind in ("snapshot", "holdings", "h")
    use_plan = kind in ("plan", "p")
    if kind == "auto" or kind in ("", "a"):
        use_snapshot = _is_holdings_snapshot_file(path)
        use_plan = not use_snapshot
    if use_snapshot:
        header, snap = load_holdings_snapshot(path)
        _warn_if_snapshot_mismatch(header, source_type, source_id)
        held = snap[snap["shares_held"] > 0]
        if held.empty:
            print("  No shares_held > 0 in the holdings file.")
            sys.exit(1)
        return {
            str(r["symbol"]).upper(): int(r["shares_held"])
            for _, r in held.iterrows()
        }
    if use_plan:
        return load_prior_shares_from_plan_csv(path)
    print(f"  Unknown PSX_SIP_HOLDINGS_KIND={kind!r} (use auto, snapshot, or plan).")
    sys.exit(1)


def _env_holdings_csv_path() -> str | None:
    raw = os.environ.get("PSX_SIP_HOLDINGS", "").strip()
    if not raw:
        return None
    path = os.path.expanduser(raw)
    if not os.path.isfile(path):
        print(f"  PSX_SIP_HOLDINGS file not found: {path}")
        sys.exit(1)
    return path


def _discover_holdings_file_options(default_snapshot: str) -> list[tuple[str, str]]:
    seen: set[str] = set()
    options: list[tuple[str, str]] = []

    def add(label: str, path: str) -> None:
        absp = os.path.abspath(os.path.expanduser(path))
        if absp in seen or not os.path.isfile(absp):
            return
        seen.add(absp)
        options.append((label, absp))

    state = load_user_state()
    last = state.get("last_holdings_path", "")
    if last:
        sid = state.get("last_source_id", "")
        lbl = "Last saved holdings"
        if sid:
            lbl += f" ({sid})"
        add(lbl, last)
    if default_snapshot:
        add(f"File in this folder ({default_snapshot})", default_snapshot)
    return options


def choose_topup_holdings_source(
    default_snapshot: str,
    source_type: str,
    source_id: str,
) -> dict[str, int]:
    env_path = _env_holdings_csv_path()
    if env_path:
        print(f"\nUsing PSX_SIP_HOLDINGS: {env_path}")
        return _load_prior_shares_from_any_csv(env_path, source_type, source_id)

    discovered = _discover_holdings_file_options(default_snapshot)

    while True:
        print("\nLoad holdings for top-up:\n")
        print(
            "  1. Enter CSV path (any name — holdings snapshot or plan CSV)"
        )
        menu_paths: list[tuple[str, str]] = []
        next_idx = 2
        for label, path in discovered:
            print(f"  {next_idx}. {label}")
            print(f"     {path}")
            menu_paths.append((label, path))
            next_idx += 1
        manual_idx = next_idx
        print(f"  {manual_idx}. Type holdings manually (SYMBOL:qty)")
        print("  h. Help (shortcuts)")
        print()

        raw = prompt("Select [1]: ") or "1"
        if _is_help_choice(raw):
            print_usage_tips()
            continue
        if not raw.isdigit():
            print(f"  Enter 1–{manual_idx} or h.")
            continue
        pick = int(raw)

        if pick == 1:
            path = prompt_existing_csv_path()
            print(f"\nUsing: {path}")
            return _load_prior_shares_from_any_csv(path, source_type, source_id)

        if 2 <= pick < manual_idx:
            path = menu_paths[pick - 2][1]
            print(f"\nUsing: {path}")
            return _load_prior_shares_from_any_csv(path, source_type, source_id)

        if pick == manual_idx:
            return prompt_manual_holdings()

        print(f"  Enter 1–{manual_idx} or h.")


def _warn_if_snapshot_mismatch(
    header: dict[str, str] | None,
    source_type: str,
    source_id: str,
) -> None:
    if not header:
        return
    file_type = header.get("source_type", "")
    file_id = header.get("source_id", "")
    if file_type and file_type != source_type:
        print(
            f"  Warning: prior file source_type={file_type!r} "
            f"but you selected {source_type!r}."
        )
    if file_id and file_id.upper() != source_id.upper():
        ans = prompt(
            f"  Prior record is for {file_id!r}, you chose {source_id!r}. "
            "Continue anyway? [y/N]: "
        ).lower()
        if ans not in ("y", "yes"):
            sys.exit(0)


def run_topup_from_prior_shares(
    prior_shares: dict[str, int],
    live_df: pd.DataFrame,
    source_type: str,
    source_id: str,
) -> tuple[pd.DataFrame, dict[str, int]]:
    held_symbols = {s for s, q in prior_shares.items() if q > 0}
    if not held_symbols:
        print("  No holdings with share count > 0.")
        sys.exit(1)

    subset = live_df[live_df["symbol"].isin(held_symbols)].copy()
    missing_live = held_symbols - set(subset["symbol"].astype(str))
    if missing_live:
        print(
            "  Warning: no current basket price/weight for: "
            + ", ".join(sorted(missing_live))
        )
    if subset.empty:
        print("  None of your held symbols appear in the live basket.")
        sys.exit(1)

    if len(subset) < len(held_symbols):
        for sym in sorted(held_symbols - set(subset["symbol"].astype(str))):
            print(
                f"  Skipping {sym} ({prior_shares[sym]} shares) — "
                "not in live priced basket."
            )

    print(
        f"\nTop-up mode: {len(subset)} held name(s), "
        f"weights re-normalized from live {source_type} {source_id}."
    )
    show = subset[["symbol", "name", "idx_weight"]].copy()
    show["idx_weight"] = show["idx_weight"].map(lambda w: f"{w:.2f}%")
    print(show.to_string(index=False))
    print_switch_considerations(held_symbols, live_df)
    return subset.reset_index(drop=True), prior_shares


def run_topup_flow(
    live_df: pd.DataFrame,
    source_type: str,
    source_id: str,
    default_snapshot: str,
) -> tuple[pd.DataFrame, dict[str, int]]:
    prior = choose_topup_holdings_source(
        default_snapshot, source_type, source_id
    )
    return run_topup_from_prior_shares(prior, live_df, source_type, source_id)


def maybe_save_csv(plan: pd.DataFrame, default_name: str = "psx_sip_plan.csv") -> None:
    answer = prompt("\nSave full plan to CSV? [y/N]: ").lower()
    if answer not in ("y", "yes"):
        return
    path = prompt(f"File name [{default_name}]: ") or default_name
    plan.to_csv(path, index=False)
    print(f"Saved: {path}")


def _finalize_sip_plan(
    scoped: pd.DataFrame,
    investment_pkr: float,
    default_csv: str = "psx_sip_plan.csv",
    *,
    source_type: str = "index",
    source_id: str = "",
    live_df: pd.DataFrame | None = None,
    meta: dict | None = None,
    prior_shares: dict[str, int] | None = None,
) -> None:
    cost_buffer_pct = prompt_non_negative_float(
        "Extra cost buffer on buy price (%)",
        DEFAULT_COST_BUFFER_PCT,
    )

    lot_answer = prompt(
        f"\nRound down to PSX board lots ({BOARD_LOT} shares)? [y/N]: "
    ).lower()
    use_board_lot = lot_answer in ("y", "yes")

    plan = compute_sip_plan(scoped, investment_pkr, use_board_lot, cost_buffer_pct)
    print_plan(plan, investment_pkr, use_board_lot, cost_buffer_pct)
    maybe_save_csv(plan, default_name=default_csv)
    basket_df = live_df if live_df is not None else scoped
    holdings_default = (
        f"psx_sip_holdings_{source_id.lower()}.csv"
        if source_id
        else "psx_sip_holdings.csv"
    )
    maybe_save_holdings_snapshot(
        plan,
        basket_df,
        source_type=source_type,
        source_id=source_id,
        investment_pkr=investment_pkr,
        cost_buffer_pct=cost_buffer_pct,
        use_board_lot=use_board_lot,
        meta=meta,
        prior_shares=prior_shares,
        default_name=holdings_default,
    )


def _run_benchmark_sip_flow(
    df: pd.DataFrame,
    meta: dict,
    *,
    source_type: str,
    source_id: str,
    universe_label: str,
    weight_label: str,
    empty_exclusion_msg: str,
    exclusion_unit: str,
) -> None:
    """Shared Index / ETF SIP path (new plan, top-up, scope, snapshot)."""
    mode = choose_sip_mode(source_id)
    prior_shares: dict[str, int] | None = None
    scoped: pd.DataFrame
    if mode == "topup":
        default_path = f"psx_sip_holdings_{source_id.lower()}.csv"
        scoped, prior_shares = run_topup_flow(
            df, source_type, source_id, default_path
        )
    else:
        exclude_sectors, exclude_symbols = prompt_exclusions(
            df, universe_label=universe_label
        )
        filtered = apply_exclusions(df, exclude_sectors, exclude_symbols)
        if filtered.empty:
            print(empty_exclusion_msg)
            return
        if len(filtered) < len(df):
            print(
                f"\nAfter exclusions: {len(filtered)} {exclusion_unit} "
                f"(removed {len(df) - len(filtered)})."
            )
        scoped = prompt_portfolio_scope(
            filtered,
            source_id,
            universe_label=universe_label,
            weight_label=weight_label,
        )

    investment_pkr = prompt_positive_float(
        "\nMonthly SIP investment amount (PKR)"
    )

    _finalize_sip_plan(
        scoped,
        investment_pkr,
        default_csv=f"psx_sip_plan_{source_id.lower()}.csv",
        source_type=source_type,
        source_id=source_id,
        live_df=df,
        meta=meta,
        prior_shares=prior_shares,
    )


def run_index_flow() -> None:
    index_name = choose_index()
    df, meta = load_index_frame(index_name)
    print_overview_tables(df, index_name)
    _run_benchmark_sip_flow(
        df,
        meta,
        source_type="index",
        source_id=index_name,
        universe_label="index",
        weight_label="index",
        empty_exclusion_msg="All companies were excluded. Nothing to allocate.",
        exclusion_unit="companies",
    )


def run_etf_flow() -> None:
    etf_symbol = choose_etf()
    df, meta = load_etf_frame(etf_symbol)
    print_etf_overview(df, etf_symbol, meta)
    _run_benchmark_sip_flow(
        df,
        meta,
        source_type="etf",
        source_id=etf_symbol,
        universe_label="ETF basket",
        weight_label="basket",
        empty_exclusion_msg="All holdings were excluded. Nothing to allocate.",
        exclusion_unit="holdings",
    )


def main() -> None:
    print(f"PSX Index / ETF SIP Planner v{__version__} (psxdata)")
    print(f"Author: {__author__} · {__date__}")
    print("=" * 40)
    print_startup_reminders()

    while True:
        product = choose_product_type()
        if product == "etf":
            run_etf_flow()
        elif product == "insights":
            run_insights_flow()
        elif product == "align":
            run_holdings_align_flow()
        else:
            run_index_flow()

        if not prompt_return_to_menu():
            print("\nDone.")
            break


if __name__ == "__main__":
    main()
