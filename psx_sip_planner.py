#!/usr/bin/env python3
"""Interactive PSX index / ETF SIP planner and market insights using psxdata.

Author: Salim Ali Khan · Version 1.0.3 · October 2026
"""

from __future__ import annotations

__version__ = "1.0.3"
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

import logging
import math
import re
from datetime import timedelta

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


def choose_product_type() -> str:
    print("\nWhat would you like to plan?\n")
    print("  1. PSX Index (constituent stocks by index weight)")
    print("  2. ETF (underlying basket from PSX creation unit)")
    print("  3. Insights (indices, ETFs, sector leaders — no SIP plan)")
    print()
    while True:
        choice = prompt("Select [1]: ") or "1"
        if choice in ("1", "index", "i"):
            return "index"
        if choice in ("2", "etf", "e"):
            return "etf"
        if choice in ("3", "insights", "insight"):
            return "insights"
        print("  Enter 1, 2, or 3.")


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


def load_index_frame(index_name: str) -> pd.DataFrame:
    print(f"\nFetching {index_name} constituents from PSX (cached when available)...")
    df = psxdata.indices(index_name)
    if df is None or df.empty:
        print(f"No data returned for index {index_name}.")
        sys.exit(1)

    symbols_meta = psxdata.symbols()
    if "sector_name" in symbols_meta.columns:
        df = df.merge(
            symbols_meta[["symbol", "sector_name"]],
            on="symbol",
            how="left",
        )
    else:
        df["sector_name"] = ""

    df["price"] = df["current"].map(parse_price)
    df["idx_weight"] = pd.to_numeric(df["idx_weight"], errors="coerce")
    df = df.dropna(subset=["price", "idx_weight"])
    df["sector_name"] = df["sector_name"].fillna("UNKNOWN").astype(str)
    return (
        df.sort_values("idx_weight", ascending=False)
        .reset_index(drop=True)
    )


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


def prompt_exclusions(df: pd.DataFrame) -> tuple[set[str], set[str]]:
    valid_symbols = set(df["symbol"].astype(str))
    sector_map = {
        s.upper(): s for s in sorted(df["sector_name"].unique(), key=str.casefold)
    }

    print("\n--- Exclusions (optional) ---")
    print("Leave blank to include everything.\n")

    sector_list = list(sector_map.values())
    print("Sectors in this index:")
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


def prompt_portfolio_scope(df: pd.DataFrame, index_name: str) -> pd.DataFrame:
    hint = ""
    if index_name == "KMI30":
        hint = " (common for KMI30 SIP: top 5 ≈ 49% index weight, top 10 ≈ 79%)"

    print(f"\n--- Portfolio scope ---{hint}\n")
    print("  1. All stocks in the index")
    print("  2. Top 5 by index weight")
    print("  3. Top 10 by index weight")
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
            f"{raw_w:.2f}% of index weight (before re-normalizing for SIP)."
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


def run_index_flow() -> None:
    index_name = choose_index()
    df = load_index_frame(index_name)
    print_overview_tables(df, index_name)

    investment_pkr = prompt_positive_float(
        "\nMonthly SIP investment amount (PKR)"
    )

    exclude_sectors, exclude_symbols = prompt_exclusions(df)
    filtered = apply_exclusions(df, exclude_sectors, exclude_symbols)
    if filtered.empty:
        print("All companies were excluded. Nothing to allocate.")
        return
    if len(filtered) < len(df):
        print(
            f"\nAfter exclusions: {len(filtered)} companies "
            f"(removed {len(df) - len(filtered)})."
        )

    scoped = prompt_portfolio_scope(filtered, index_name)
    _finalize_sip_plan(scoped, investment_pkr)


def run_etf_flow() -> None:
    etf_symbol = choose_etf()
    df, meta = load_etf_frame(etf_symbol)
    print_etf_overview(df, etf_symbol, meta)

    investment_pkr = prompt_positive_float(
        "\nMonthly SIP investment amount (PKR)"
    )

    exclude_sectors, exclude_symbols = prompt_exclusions(df)
    filtered = apply_exclusions(df, exclude_sectors, exclude_symbols)
    if filtered.empty:
        print("All holdings were excluded. Nothing to allocate.")
        return
    if len(filtered) < len(df):
        print(
            f"\nAfter exclusions: {len(filtered)} holdings "
            f"(removed {len(df) - len(filtered)})."
        )

    _finalize_sip_plan(
        filtered,
        investment_pkr,
        default_csv=f"psx_sip_plan_{etf_symbol.lower()}.csv",
    )


def main() -> None:
    print(f"PSX Index / ETF SIP Planner v{__version__} (psxdata)")
    print(f"Author: {__author__} · {__date__}")
    print("=" * 40)

    while True:
        product = choose_product_type()
        if product == "etf":
            run_etf_flow()
        elif product == "insights":
            run_insights_flow()
        else:
            run_index_flow()

        if not prompt_return_to_menu():
            print("\nDone.")
            break


if __name__ == "__main__":
    main()
