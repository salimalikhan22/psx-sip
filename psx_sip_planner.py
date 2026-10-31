#!/usr/bin/env python3
"""Interactive PSX index / ETF SIP planner and market insights using psxdata.

Author: Salim Ali Khan · Version 1.2.2 · October 2026

Privacy: this script does not upload your holdings or plans. It only reads/writes
files you choose locally and fetches public PSX market data (see README).
"""

from __future__ import annotations

__version__ = "1.4.5"
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

import glob
import json
import logging
import math
import re
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone

import pandas as pd
import psxdata
from bs4 import BeautifulSoup
from psxdata.constants import BASE_URL, COLUMN_MAP, INDEX_NAMES, REQUEST_HEADERS
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


def _use_data_cache() -> bool:
    """psx-sip defaults to live PSX (no psxdata disk cache, no in-session ETF reuse)."""
    if os.environ.get("PSX_SIP_NO_CACHE", "").strip().lower() in (
        "1",
        "yes",
        "true",
    ):
        return False
    return os.environ.get("PSX_SIP_USE_CACHE", "").strip().lower() in (
        "1",
        "yes",
        "true",
    )


def fetch_symbols() -> pd.DataFrame:
    return psxdata.symbols(cache=_use_data_cache())


def fetch_screener() -> pd.DataFrame:
    return psxdata.screener(cache=_use_data_cache())


def fetch_indices(name: str) -> pd.DataFrame:
    return psxdata.indices(name, cache=_use_data_cache())


def fetch_sectors() -> pd.DataFrame:
    return psxdata.sectors(cache=_use_data_cache())


def fetch_stocks(
    symbol: str,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    if start is not None or end is not None:
        return psxdata.stocks(
            symbol,
            start=start,
            end=end,
            cache=_use_data_cache(),
        )
    return psxdata.stocks(symbol, cache=_use_data_cache())


BOARD_LOT = 500
DEFAULT_COST_BUFFER_PCT = 0.5
HOLDINGS_SNAPSHOT_VERSION = "2"
# Raw GitHub base for curl | python3 bootstrap (override with PSX_SIP_REPO_RAW).
DEFAULT_PSX_SIP_REPO_RAW = (
    "https://raw.githubusercontent.com/salimalikhan22/psx-sip/main"
)
# Extra folders to scan for psx_sip_*.csv (user-picked this session only).
_session_csv_search_dirs: list[str] = []


def _export_date_tag() -> str:
    """Local calendar date for export filenames (DD_MM_YYYY)."""
    return datetime.now().strftime("%d_%m_%Y")


def _export_source_slug(source_id: str, fallback: str = "plan") -> str:
    raw = (source_id or fallback).lower()
    return re.sub(r"[^\w\-]+", "_", raw).strip("_")[:50]


def export_plan_csv_name(source_id: str) -> str:
    return f"psx_sip_plan_{_export_source_slug(source_id)}_{_export_date_tag()}.csv"


def export_holdings_csv_name(source_id: str) -> str:
    return (
        f"psx_sip_holdings_{_export_source_slug(source_id, 'holdings')}"
        f"_{_export_date_tag()}.csv"
    )


def holdings_snapshot_glob_pattern(source_id: str) -> str:
    slug = _export_source_slug(source_id, "holdings")
    return f"psx_sip_holdings_{slug}_*.csv"


def _csv_search_directories() -> list[str]:
    """Folders scanned for psx_sip_*.csv: cwd, session picks, optional PSX_SIP_CSV_DIRS."""
    candidates: list[str] = [os.getcwd(), *_session_csv_search_dirs]
    extra = os.environ.get("PSX_SIP_CSV_DIRS", "").strip()
    if extra:
        for part in extra.replace(";", ",").split(","):
            part = part.strip()
            if part:
                candidates.append(part)
    seen: set[str] = set()
    roots: list[str] = []
    for raw in candidates:
        absp = os.path.abspath(os.path.expanduser(raw))
        if absp in seen or not os.path.isdir(absp):
            continue
        seen.add(absp)
        roots.append(absp)
    return roots


def _csv_location_label(path: str, roots: list[str]) -> str:
    absp = os.path.abspath(path)
    cwd = os.path.abspath(os.getcwd())
    for root in roots:
        try:
            if os.path.commonpath([absp, root]) != root:
                continue
        except ValueError:
            continue
        if os.path.abspath(root) == cwd:
            return "current folder"
        return os.path.basename(root) or root
    return "file"


def _maybe_prompt_csv_search_folder() -> bool:
    """Ask for a folder to scan when cwd has no exports. Returns True if one was added."""
    cwd = os.getcwd()
    print(f"\n  No psx_sip_*.csv in the current folder:\n    {cwd}")
    raw = prompt(
        "  Folder to search for plan/holdings CSVs "
        "(Enter = skip; you can type a file path instead): "
    ).strip()
    if not raw:
        return False
    if _is_help_choice(raw):
        print_usage_tips()
        return _maybe_prompt_csv_search_folder()
    path = os.path.abspath(os.path.expanduser(raw.strip()))
    if os.path.isdir(path):
        if path in _session_csv_search_dirs or path == cwd:
            print("  That folder is already included in the search.")
            return False
        _session_csv_search_dirs.append(path)
        print(f"  Added search folder: {path}")
        return True
    if os.path.isfile(path):
        print("  That is a file — give a folder path, or Enter then type the CSV file.")
        return False
    print(f"  Folder not found: {path}")
    return False


def _csv_export_kind(filename: str) -> str:
    low = filename.lower()
    if low.startswith("psx_sip_holdings"):
        return "holdings"
    if low.startswith("psx_sip_plan"):
        return "plan"
    if low.startswith("psx_scan"):
        return "scan"
    return "csv"


def discover_psx_sip_csv_files(
    *,
    source_id: str = "",
    include_plans: bool = True,
    include_holdings: bool = True,
    include_scans: bool = False,
    max_results: int = 30,
    prompt_extra_folder: bool = False,
) -> list[tuple[str, str]]:
    """
    (label, absolute_path) newest first; scans cwd (+ session / PSX_SIP_CSV_DIRS).
    """
    roots = _csv_search_directories()
    patterns: list[str] = []
    if include_holdings:
        patterns.append("psx_sip_holdings*.csv")
    if include_plans:
        patterns.append("psx_sip_plan*.csv")
    if include_scans:
        patterns.append("psx_scan_*.csv")

    by_path: dict[str, float] = {}
    for root in roots:
        for pat in patterns:
            for path in glob.glob(os.path.join(root, pat)):
                absp = os.path.abspath(path)
                if os.path.isfile(absp):
                    by_path[absp] = os.path.getmtime(absp)

    slug = _export_source_slug(source_id, "") if source_id else ""
    sid_low = source_id.lower() if source_id else ""
    ranked: list[tuple[int, float, str, str]] = []
    for path, mtime in by_path.items():
        name = os.path.basename(path)
        pri = 0
        name_low = name.lower()
        if slug and slug in name_low:
            pri = 3
        elif sid_low and sid_low in name_low:
            pri = 2
        elif _csv_export_kind(name) == "holdings":
            pri = 1
        loc = _csv_location_label(path, roots)
        mod = datetime.fromtimestamp(mtime).strftime("%d-%m-%Y %H:%M")
        kind = _csv_export_kind(name)
        label = f"{kind} · {loc} · {name} · {mod}"
        ranked.append((pri, mtime, label, path))

    ranked.sort(key=lambda r: (r[0], r[1]), reverse=True)
    out = [(label, path) for _pri, _mt, label, path in ranked[:max_results]]
    if out or not prompt_extra_folder:
        return out
    if _maybe_prompt_csv_search_folder():
        return discover_psx_sip_csv_files(
            source_id=source_id,
            include_plans=include_plans,
            include_holdings=include_holdings,
            include_scans=include_scans,
            max_results=max_results,
            prompt_extra_folder=False,
        )
    return []


def resolve_default_holdings_snapshot_path(source_id: str) -> str:
    """Newest holdings CSV in search dirs (cwd first), else legacy undated, else today's name."""
    slug = _export_source_slug(source_id, "holdings")
    roots = _csv_search_directories()
    dated: list[str] = []
    seen: set[str] = set()
    for root in roots:
        for pat in (f"psx_sip_holdings_{slug}_*.csv", "psx_sip_holdings*.csv"):
            for path in glob.glob(os.path.join(root, pat)):
                absp = os.path.abspath(path)
                if absp in seen or not os.path.isfile(absp):
                    continue
                seen.add(absp)
                dated.append(absp)
    if dated:
        dated.sort(key=os.path.getmtime, reverse=True)
        if source_id:
            matched = [
                p for p in dated if slug in os.path.basename(p).lower()
            ]
            if matched:
                matched.sort(key=os.path.getmtime, reverse=True)
                return matched[0]
        return dated[0]
    for root in roots:
        legacy = os.path.join(root, f"psx_sip_holdings_{source_id.lower()}.csv")
        if os.path.isfile(legacy):
            return legacy
    return export_holdings_csv_name(source_id)


MISSING_WEIGHT_ALERT_PCT = 3.0
MISSING_TOP_N_ALERT = 10
MISSING_TOTAL_WEIGHT_ALERT_PCT = 15.0
BASKET_WEIGHT_DRIFT_PP = 1.0
CUSTOM_BASKET_DEFAULT_COUNT = 12
CUSTOM_BASKET_MAX_PER_SECTOR = 3
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
    syms = fetch_symbols()
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
    symbols_meta = fetch_symbols()
    if "sector_name" not in symbols_meta.columns:
        return {}
    return (
        symbols_meta.set_index("symbol")["sector_name"].astype(str).to_dict()
    )


def _market_price_index() -> tuple[pd.DataFrame, str]:
    screener = fetch_screener()
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


def _fallback_price_from_history(symbol: str, lookback_days: int = 21) -> float | None:
    """PSX screener omits some liquid names; use recent daily close from psxdata."""
    sym = symbol.upper()
    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=lookback_days)
    try:
        hist = fetch_stocks(sym, start=start.isoformat(), end=end.isoformat())
    except Exception:
        return None
    if hist is None or hist.empty or "close" not in hist.columns:
        return None
    hist = hist.sort_values("date")
    try:
        close = float(hist.iloc[-1]["close"])
    except (ValueError, TypeError, IndexError):
        return None
    return close if close > 0 else None


def _resolve_live_price(
    symbol: str, scr: pd.DataFrame, price_col: str
) -> tuple[float | None, str]:
    sym = symbol.upper()
    price = _lookup_unit_price(sym, scr, price_col)
    if price is not None:
        return price, "screener"
    price = _fallback_price_from_history(sym)
    if price is not None:
        return price, "history"
    return None, ""


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
    history_priced: list[str] = []
    for h in parsed["holdings"]:
        sym = h["symbol"]
        price, source = _resolve_live_price(sym, scr, price_col)
        if price is None:
            missing_prices.append(sym)
            continue
        if source == "history":
            history_priced.append(sym)
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

    if history_priced and not quiet:
        print(
            "  Note: price from recent PSX daily close (not in live screener): "
            + ", ".join(history_priced)
        )
    if missing_prices and not quiet:
        print(f"  Warning: no price for: {', '.join(missing_prices)}")

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
        "history_priced_symbols": list(history_priced),
    }
    return (
        df.sort_values("idx_weight", ascending=False).reset_index(drop=True),
        meta,
    )


def _load_etf_catalog() -> dict[str, dict]:
    global _etf_catalog
    if _use_data_cache() and _etf_catalog is not None:
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
        if _use_data_cache():
            _etf_frame_cache[sym] = (df, meta)
        catalog[sym] = {
            "kind": "equity",
            "line": _composition_summary_line(df, meta),
            "price_line": price_line,
            "etf_unit_price": etf_price,
            "parsed": parsed,
            "html": html,
        }

    if _use_data_cache():
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
        df = fetch_stocks(symbol.upper())
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
    scr = fetch_screener()
    sym_meta = fetch_symbols()
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

    psx_sectors = fetch_sectors()
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
            print(
                "\n  Note: full snapshot fetches every index and ETF "
                "(slow). For a faster run use 2, 3, or 4 and pick only what you need."
            )
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
        maybe_export_insights_tables([], [], stock_stats)
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
        maybe_export_insights_tables(index_rows, etf_rows)
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
    maybe_export_insights_tables(index_rows, etf_rows, stock_stats)


def maybe_export_insights_tables(
    index_rows: list[dict],
    etf_rows: list[dict],
    stock_stats: pd.DataFrame | None = None,
) -> None:
    if not index_rows and not etf_rows and (stock_stats is None or stock_stats.empty):
        return
    answer = prompt("\nExport insights to CSV? [y/N]: ").lower()
    if answer not in ("y", "yes"):
        return
    path = prompt("File name [psx_sip_insights.csv]: ") or "psx_sip_insights.csv"
    path = resolve_user_path(path)
    parts: list[pd.DataFrame] = []
    if index_rows:
        parts.append(pd.DataFrame(index_rows).assign(table="index"))
    if etf_rows:
        parts.append(pd.DataFrame(etf_rows).assign(table="etf"))
    if stock_stats is not None and not stock_stats.empty:
        parts.append(stock_stats.assign(table="sector_pulse"))
    if not parts:
        return
    pd.concat(parts, ignore_index=True).to_csv(path, index=False)
    print(f"Saved: {path}")


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
  • Main: 1 Invest (index / ETF / dividend basket) · 2 Insights (overview, compare, scanner).
  • Index & ETF: scope, sector/symbol exclusions, new plan or top-up.
  • Dividend basket: build once (DPS + yield; EPS optional) → invest skips scope — optional skip symbol, then PKR.
  • Insights scanner: P/E, yield, EPS research — optional basket, separate from Invest dividend flow.
  • Top-up: held weights / equal / gap-fill / dividend tilt; optional sector skip this month.
  • Index/ETF top-up: optional buy filter (dividend-first or strict P/E+EPS). Dividend basket top-up skips that filter.
  • Before SIP: at basket confirm, type r to swap lines; why SYM for diagnosis.
  • Exports default to dated names: psx_sip_plan_<id>_DD_MM_YYYY.csv and psx_sip_holdings_<id>_DD_MM_YYYY.csv.
  • New vs top-up: if a holdings file exists, you see holdings vs basket before you choose.
  • CSV auto-find: current folder only; if empty, you can name another folder to search.
  • Optional: PSX_SIP_CSV_DIRS=path1,path2 always adds extra search folders.
  • curl | python3: fund JSON auto-downloads to ~/.psx-sip/ (PSX_SIP_REPO_RAW to override).
  • Snapshot load shows basket drift (new/dropped names, weight shifts).
  • Match check (menu 4): Index or ETF → CSV or SYMBOL:qty → resemblance %.
  • Paths: relative (from cwd), absolute, or ~/file.csv — all work.
  • Multi-select indices/ETFs: type all (or *) to select every item in the list.
  • Env: PSX_SIP_HOLDINGS=/path/to.csv skips the load menu in top-up.
  • Lost file? Top-up menu 2 (plan CSV) or 3 (SYMBOL:qty, e.g. FFC:1500).
  • Local reminder only: ~/.psx-sip/user_state.json (last file path). Set
    PSX_SIP_NO_LOCAL_STATE=1 to disable. Nothing is sent to the internet.
""".strip()
    )


def print_startup_reminders() -> None:
    print("  Menus: type h or ? for shortcuts.")
    if _running_from_stdin():
        print(
            "  Running from curl | python3 — fund data caches to ~/.psx-sip/ when needed."
        )
        print(
            f"  CSV exports: save in your current folder ({os.getcwd()})."
        )
    if _use_data_cache():
        print(
            "  Data cache: ON (psxdata disk ~15 min + in-session ETF). "
            "Unset PSX_SIP_USE_CACHE for live-only."
        )
    else:
        print(
            "  Data cache: OFF — live PSX each fetch (set PSX_SIP_USE_CACHE=1 to cache)."
        )
    if not _local_state_enabled():
        return
    state = load_user_state()
    last = state.get("last_holdings_path", "")
    if last and os.path.isfile(last):
        sid = state.get("last_source_id", "")
        tag = f" ({sid})" if sid else ""
        print(f"  Last holdings file{tag}: {last}")
        print(f"  Next top-up: menu 2, or PSX_SIP_HOLDINGS={last!r}")
    found = discover_psx_sip_csv_files(max_results=50, prompt_extra_folder=False)
    if found:
        print(
            f"  Found {len(found)} PSX SIP CSV(s) in {os.getcwd()} "
            "(top-up / holdings prompts will list them)."
        )


def _is_help_choice(choice: str) -> bool:
    return choice.strip().lower() in ("h", "?", "help")


def choose_top_level_mode() -> str:
    print("\nWhat would you like to do?\n")
    print("  1. Invest — SIP plan (index, ETF, or dividend basket)")
    print("  2. Insights — research & compare (no SIP plan)")
    print("  h. Help (shortcuts)")
    while True:
        choice = prompt("Select [1]: ") or "1"
        if _is_help_choice(choice):
            print_usage_tips()
            continue
        if choice in ("1", "invest", "sip", "plan"):
            return "invest"
        if choice in ("2", "insights", "research", "insight", "i"):
            return "insights"
        print("  Enter 1, 2, or h.")


def choose_invest_product() -> str:
    print(
        "\n  Invest via:\n"
        "  1. PSX Index — full index SIP (scope, exclusions; EPS/P/E matter more)\n"
        "  2. ETF — creation-unit basket (same SIP tools as index)\n"
        "  3. Dividend income — DPS + yield basket (EPS optional; basket → invest)"
    )
    while True:
        choice = prompt("Select [1]: ") or "1"
        if _is_help_choice(choice):
            print_usage_tips()
            continue
        if choice in ("1", "index", "idx"):
            return "index"
        if choice in ("2", "etf", "e"):
            return "etf"
        if choice in (
            "3",
            "dividend",
            "div",
            "income",
            "yield",
            "6",
        ):
            return "dividend"
        print("  Enter 1–3, or h.")


def choose_insights_product() -> str:
    print(
        "\n  Insights:\n"
        "  1. Market overview (indices, ETFs, sector leaders)\n"
        "  2. Compare my holdings to an index/ETF\n"
        "  3. Stock scanner (P/E, dividend yield, EPS — research)"
    )
    while True:
        choice = prompt("Select [1]: ") or "1"
        if _is_help_choice(choice):
            print_usage_tips()
            continue
        if choice in ("1", "insights", "market", "overview"):
            return "insights"
        if choice in ("2", "compare", "match", "align", "holdings", "4"):
            return "align"
        if choice in ("3", "scan", "scanner", "screener", "5"):
            return "scanner"
        print("  Enter 1–3, or h.")


def choose_product_type() -> str:
    """Top-level menu: invest (index/ETF/dividend) or insights tools."""
    mode = choose_top_level_mode()
    if mode == "insights":
        return choose_insights_product()
    return choose_invest_product()


def prompt_return_to_menu() -> bool:
    raw = prompt("\nReturn to main menu? [Y/n]: ").lower()
    return raw not in ("n", "no")


def choose_etf() -> str:
    etfs = list_etf_symbols()
    if etfs.empty:
        print("No ETFs found in PSX symbol list.")
        sys.exit(1)

    print_etf_basket_policy()
    if os.environ.get("PSX_SIP_PRELOAD_ETF_CATALOG", "").strip().lower() in (
        "1",
        "yes",
        "true",
    ):
        catalog = _load_etf_catalog()
    else:
        catalog = {}

    print(
        "\nAvailable PSX ETFs "
        "(names only — full creation unit loads after you pick one):\n"
    )
    symbols = etfs["symbol"].astype(str).tolist()
    scr, price_col = _market_price_index()
    for pos, (_, row) in enumerate(etfs.iterrows(), start=1):
        sym = str(row["symbol"])
        if sym in catalog:
            info = catalog[sym]
            price_line = info.get("price_line", _format_etf_unit_price(None))
            print(f"  {pos:2}. {sym} — {row['name']}  |  {price_line}")
            print(f"      {info.get('line', '')}")
        else:
            unit = _lookup_unit_price(sym, scr, price_col)
            print(
                f"  {pos:2}. {sym} — {row['name']}  |  "
                f"{_format_etf_unit_price(unit)}"
            )
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
    if _use_data_cache() and etf_symbol in _etf_frame_cache:
        return _etf_frame_cache[etf_symbol]

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
    if _use_data_cache():
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
    parsed = meta.get("parsed_holdings") or []
    pub_n = len(parsed)
    if pub_n:
        print(
            f"  PSX published basket: {pub_n} equity names | "
            f"in this table: {len(df)}"
        )
    hist = meta.get("history_priced_symbols") or []
    if hist:
        print(
            "  Recent daily close used (symbol not in live screener): "
            + ", ".join(hist)
        )
    missing = meta.get("missing_price_symbols") or []
    if missing:
        print(
            "  No price available (in PSX basket but omitted from SIP math): "
            + ", ".join(missing)
        )
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


def prompt_positive_float(label: str, default: float | None = None) -> float:
    hint = f" [{default:g}]" if default is not None else ""
    while True:
        raw = prompt(f"{label}{hint}: ")
        if not raw.strip() and default is not None:
            return float(default)
        try:
            value = float(raw.replace(",", ""))
        except ValueError:
            print("  Enter a valid number.")
            continue
        if value <= 0:
            print("  Amount must be greater than zero.")
            continue
        return value


def prompt_yes_no(label: str, *, default_yes: bool) -> bool:
    """y/n prompt; empty line uses default_yes."""
    tag = "Y/n" if default_yes else "y/N"
    raw = prompt(f"{label} [{tag}]: ").strip().lower()
    if not raw:
        return default_yes
    if raw in ("y", "yes"):
        return True
    if raw in ("n", "no"):
        return False
    print("  Enter y or n.")
    return prompt_yes_no(label, default_yes=default_yes)


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


def prompt_optional_float(label: str, *, hint: str = "blank = no limit") -> float | None:
    raw = prompt(f"{label} ({hint}): ").strip()
    if not raw:
        return None
    try:
        return float(raw.replace(",", ""))
    except ValueError:
        print("  Ignoring invalid number — no limit.")
        return None


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


def _is_select_all_choice(raw: str) -> bool:
    key = raw.strip().lower()
    return key in ("all", "a", "*", "everything", "full")


def _parse_index_selection(raw: str) -> list[str]:
    if _is_select_all_choice(raw):
        return list(INDEX_NAMES)
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
        "\n  Tip: comma-separate for several (e.g. 1,3,6 or KSE100,KMI30), "
        "or type all for every index."
    )
    while True:
        raw = prompt("Select indices (numbers, names, or all): ")
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
        "\n  Tip: comma-separate for several (e.g. 1,2 or MZNPETF,UBLPETF), "
        "or type all for every ETF."
    )
    while True:
        raw = prompt("Select ETFs (numbers, symbols, or all): ")
        if allow_empty and not raw.strip():
            return []
        if _is_select_all_choice(raw):
            print(f"  Selected all {len(symbols)} ETF(s): {', '.join(symbols)}")
            return [(s, names_by_symbol[s]) for s in symbols]

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
    print(f"\nFetching {index_name} constituents from PSX...")
    raw = fetch_indices(index_name)
    if raw is None or raw.empty:
        print(f"No data returned for index {index_name}.")
        sys.exit(1)

    symbols_meta = fetch_symbols()
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
        price: float | None = None
        try:
            price = parse_price(row["current"])
        except ValueError:
            price = None
        if price is None:
            price = _fallback_price_from_history(sym)
        if price is None:
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


def _rank_score(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if s.notna().sum() == 0:
        return pd.Series(0.0, index=series.index)
    lo = float(s.min())
    hi = float(s.max())
    if hi <= lo:
        return pd.Series(0.5, index=series.index)
    return (s - lo) / (hi - lo)


_company_annual_eps_cache: dict[str, list[float]] = {}
_company_financials_cache: dict[str, dict[str, list[float]]] = {}
_ANNUAL_YEAR_RE = re.compile(r"^20\d{2}$")


def _fetch_company_page_html(symbol: str) -> str:
    sym = symbol.upper()
    scraper = _get_etf_scraper()
    url = f"{BASE_URL}/company/{sym}"
    resp = scraper._request("GET", url, headers=dict(REQUEST_HEADERS))
    return resp.text


def _parse_company_annual_row(html: str, row_label: str) -> list[float]:
    """Newest annual values first (PSX /company/SYM annual financials table)."""
    want = row_label.strip().lower()
    soup = BeautifulSoup(html, "html.parser")
    for table in soup.select("table"):
        rows = table.select("tr")
        if not rows:
            continue
        header_cells = rows[0].find_all(["th", "td"])
        headers = [c.get_text(strip=True) for c in header_cells]
        if not headers or headers[0] not in ("", "Metric"):
            continue
        year_cols = headers[1:]
        if len(year_cols) < 2 or not _ANNUAL_YEAR_RE.match(str(year_cols[0])):
            continue
        for tr in rows[1:]:
            cells = [c.get_text(strip=True) for c in tr.select("td")]
            if not cells or cells[0].strip().lower() != want:
                continue
            pairs: list[tuple[int, float]] = []
            for year_s, raw in zip(year_cols, cells[1:]):
                if not _ANNUAL_YEAR_RE.match(str(year_s)):
                    continue
                val = coerce_numeric(str(raw).replace(",", ""))
                if val is None:
                    continue
                pairs.append((int(year_s), float(val)))
            if not pairs:
                continue
            pairs.sort(key=lambda p: p[0], reverse=True)
            return [v for _y, v in pairs]
    return []


def _parse_company_annual_eps(html: str) -> list[float]:
    return _parse_company_annual_row(html, "EPS")


def _company_financials(symbol: str) -> dict[str, list[float]]:
    sym = symbol.upper()
    if sym in _company_financials_cache:
        return _company_financials_cache[sym]
    try:
        html = _fetch_company_page_html(sym)
        fin = {
            "eps": _parse_company_annual_row(html, "EPS"),
            "profit_after_tax": _parse_company_annual_row(
                html, "Profit after Taxation"
            ),
        }
    except Exception:
        fin = {"eps": [], "profit_after_tax": []}
    _company_financials_cache[sym] = fin
    _company_annual_eps_cache[sym] = fin.get("eps") or []
    return fin


def _company_annual_eps_values(symbol: str) -> list[float]:
    return _company_financials(symbol).get("eps") or []


_DPS_LOOKBACK_YEARS = 5
_dividend_history_cache: dict[str, dict] = {}
_YAHOO_UA = "Mozilla/5.0 (compatible; psx-sip-planner/1.0)"


def _yahoo_dividends_enabled() -> bool:
    return os.environ.get("PSX_SIP_NO_YAHOO", "").strip().lower() not in (
        "1",
        "yes",
        "true",
    )


def _yahoo_chart_symbol(symbol: str) -> str:
    return f"{symbol.upper()}.KA"


def _fetch_yahoo_dividend_payments(
    symbol: str,
) -> list[tuple[datetime, float]]:
    """Cash dividend payments (PKR per share) from Yahoo chart events."""
    base = symbol.upper()
    tickers = [_yahoo_chart_symbol(base), base]
    seen_tickers: set[str] = set()
    for sym in tickers:
        if sym in seen_tickers:
            continue
        seen_tickers.add(sym)
        params = urllib.parse.urlencode(
            {"interval": "1d", "range": "5y", "events": "div"}
        )
        url = (
            "https://query1.finance.yahoo.com/v8/finance/chart/"
            f"{urllib.parse.quote(sym)}?{params}"
        )
        try:
            req = urllib.request.Request(url, headers={"User-Agent": _YAHOO_UA})
            with urllib.request.urlopen(req, timeout=25) as resp:
                payload = json.loads(resp.read().decode())
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError):
            continue
        block = (payload.get("chart") or {}).get("result")
        if not block:
            continue
        events = (block[0].get("events") or {}).get("dividends") or {}
        out: list[tuple[datetime, float]] = []
        for item in events.values():
            ts = item.get("date")
            amt = item.get("amount")
            if ts is None or amt is None:
                continue
            try:
                dt = datetime.fromtimestamp(int(ts), tz=timezone.utc)
                out.append((dt, float(amt)))
            except (TypeError, ValueError, OSError):
                continue
        if out:
            out.sort(key=lambda p: p[0], reverse=True)
            return out
    return []


def _psx_dividends_enabled() -> bool:
    return os.environ.get("PSX_SIP_NO_PSX_DIV", "").strip().lower() not in (
        "1",
        "yes",
        "true",
    )


def _parse_psx_announcement_date(raw: str) -> datetime | None:
    raw = raw.strip()
    for fmt in ("%b %d, %Y", "%B %d, %Y"):
        try:
            return datetime.strptime(raw, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def _parse_dps_amount_from_notice(title: str) -> float | None:
    t = title.replace(",", "")
    m = re.search(
        r"(?:rs\.?|pkr)\s*([0-9]+(?:\.[0-9]+)?)\s*(?:/-|per\s*share|/share)?",
        t,
        re.I,
    )
    if m:
        return float(m.group(1))
    return None


def _is_psx_dividend_payment_notice(title: str) -> bool:
    t = title.lower()
    if any(
        needle in t
        for needle in (
            "newspaper",
            "dispatch of",
            "dispatch ",
            "book closure",
            "notice of book",
            "publication of notice",
            "unclaimed",
            "entitlement of",
        )
    ):
        return False
    if "credit of" in t and "dividend" in t:
        return True
    if re.search(
        r"\b(final|interim|first|second|third)\b.*\b(cash\s+)?dividend\b", t
    ):
        return True
    if re.search(r"\binterim dividend 20[0-9]{2}\b", t):
        return True
    return False


def _fetch_psx_dividend_notices(
    symbol: str,
    *,
    max_offset: int = 400,
) -> list[tuple[datetime, str, float | None]]:
    """PSX company dividend notices (dps.psx.com.pk — same host as psxdata)."""
    scraper = _get_etf_scraper()
    sym = symbol.upper()
    out: list[tuple[datetime, str, float | None]] = []
    count = 100
    offset = 0
    seen: set[tuple[str, str]] = set()
    while offset <= max_offset:
        payload = {
            "type": "C",
            "symbol": sym,
            "query": "dividend",
            "count": count,
            "offset": offset,
            "date_from": "",
            "date_to": "",
            "page": "annc",
        }
        resp = scraper._request(
            "POST",
            f"{BASE_URL}/announcements",
            headers={
                **dict(REQUEST_HEADERS),
                "Content-Type": "application/x-www-form-urlencoded",
                "Referer": f"{BASE_URL}/company/{sym}",
            },
            data=payload,
        )
        soup = BeautifulSoup(resp.text, "html.parser")
        table = soup.select_one("#announcementsTable")
        if table is None:
            break
        body_rows = table.select("tbody tr")
        if not body_rows:
            break
        for tr in body_rows:
            tds = tr.select("td")
            if len(tds) < 5:
                continue
            date_s = tds[0].get_text(strip=True)
            title = tds[4].get_text(" ", strip=True)
            if not _is_psx_dividend_payment_notice(title):
                continue
            key = (date_s, title[:100].lower())
            if key in seen:
                continue
            seen.add(key)
            dt = _parse_psx_announcement_date(date_s)
            if dt is None:
                continue
            out.append((dt, title, _parse_dps_amount_from_notice(title)))
        if len(body_rows) < count:
            break
        offset += count
    out.sort(key=lambda p: p[0], reverse=True)
    return out


def _summarize_psx_dividend_notices(
    notices: list[tuple[datetime, str, float | None]],
    *,
    lookback_years: int = _DPS_LOOKBACK_YEARS,
) -> dict:
    now = datetime.now(timezone.utc)
    current_year = now.year
    start_year = current_year - lookback_years + 1
    credits_by_year: dict[int, int] = {}
    amounts_by_year: dict[int, list[float]] = {}
    for dt, _title, amt in notices:
        y = dt.year
        if y < start_year or y > current_year:
            continue
        credits_by_year[y] = credits_by_year.get(y, 0) + 1
        if amt is not None:
            amounts_by_year.setdefault(y, []).append(amt)
    years_sorted = sorted(credits_by_year.keys(), reverse=True)
    annual_totals = {
        y: float(sum(amounts_by_year[y]))
        for y in years_sorted
        if y in amounts_by_year and amounts_by_year[y]
    }
    last_calendar = annual_totals.get(current_year - 1)
    if last_calendar is None and annual_totals:
        last_calendar = annual_totals[years_sorted[0]]
    return {
        "lookback_years": lookback_years,
        "years_with_dividends": len(credits_by_year),
        "annual_totals_by_year": annual_totals,
        "annual_dps_newest_first": [annual_totals[y] for y in years_sorted if y in annual_totals],
        "credits_by_year": credits_by_year,
        "payments_count": sum(credits_by_year.values()),
        "last_fiscal_total": last_calendar,
        "source": "psx_announcements",
    }


def _merge_dps_summaries(yahoo: dict | None, psx: dict | None) -> dict:
    """Prefer Yahoo Rs totals; fill year coverage from PSX credit notices."""
    if not yahoo and not psx:
        return {
            "lookback_years": _DPS_LOOKBACK_YEARS,
            "years_with_dividends": 0,
            "annual_totals_by_year": {},
            "annual_dps_newest_first": [],
            "last_fiscal_total": None,
            "source": "none",
        }
    if not yahoo:
        return psx or {}
    if not psx:
        return yahoo
    lb = int(yahoo.get("lookback_years") or psx.get("lookback_years") or _DPS_LOOKBACK_YEARS)
    y_totals = dict(yahoo.get("annual_totals_by_year") or {})
    p_credits = dict(psx.get("credits_by_year") or {})
    p_totals = dict(psx.get("annual_totals_by_year") or {})
    all_years = set(y_totals) | set(p_credits) | set(p_totals)
    merged_totals = dict(y_totals)
    for y, val in p_totals.items():
        if y not in merged_totals:
            merged_totals[y] = val
    years_with = len(all_years)
    years_sorted = sorted(all_years, reverse=True)
    last = yahoo.get("last_fiscal_total")
    if last is None:
        last = psx.get("last_fiscal_total")
    return {
        "lookback_years": lb,
        "years_with_dividends": years_with,
        "annual_totals_by_year": merged_totals,
        "annual_dps_newest_first": [merged_totals[y] for y in years_sorted if y in merged_totals],
        "credits_by_year": p_credits,
        "payments_count": int(yahoo.get("payments_count") or 0)
        + int(psx.get("payments_count") or 0),
        "last_fiscal_total": last,
        "source": "yahoo+psx",
    }


def _summarize_dps_payments(
    payments: list[tuple[datetime, float]],
    *,
    lookback_years: int = _DPS_LOOKBACK_YEARS,
) -> dict:
    now = datetime.now(timezone.utc)
    current_year = now.year
    start_year = current_year - lookback_years + 1
    by_year: dict[int, list[float]] = {}
    for dt, amt in payments:
        y = dt.year
        if y < start_year or y > current_year:
            continue
        by_year.setdefault(y, []).append(amt)
    years_sorted = sorted(by_year.keys(), reverse=True)
    annual_totals = {y: float(sum(by_year[y])) for y in years_sorted}
    last_calendar = annual_totals.get(current_year - 1)
    if last_calendar is None and years_sorted:
        last_calendar = annual_totals.get(years_sorted[0])
    return {
        "lookback_years": lookback_years,
        "years_with_dividends": len(by_year),
        "annual_totals_by_year": annual_totals,
        "annual_dps_newest_first": [annual_totals[y] for y in years_sorted],
        "payments_count": sum(len(v) for v in by_year.values()),
        "last_fiscal_total": last_calendar,
        "source": "yahoo",
    }


def _implied_dps_from_screener(symbol: str) -> float | None:
    row = _screener_row(symbol)
    if row is None:
        return None
    price = pd.to_numeric(row.get("price"), errors="coerce")
    dy = pd.to_numeric(row.get("dividend_yield"), errors="coerce")
    if pd.isna(price) or pd.isna(dy) or float(price) <= 0:
        return None
    return float(price) * float(dy) / 100.0


def _screener_row(symbol: str) -> pd.Series | None:
    scr = _build_screener_enriched()
    if scr.empty:
        return None
    sym = symbol.upper()
    hit = scr[scr["symbol"].astype(str).str.upper() == sym]
    if hit.empty:
        return None
    return hit.iloc[0]


def _related_equity_symbol_variants(symbol: str) -> list[str]:
    """Same company may appear as HUBC (symbols table) vs HUBCXD (screener/index)."""
    sym = str(symbol or "").upper().strip()
    if not sym:
        return []
    variants: list[str] = [sym]
    if sym.endswith("XD"):
        base = sym[:-2]
        if base and base not in variants:
            variants.append(base)
    else:
        xd = sym + "XD"
        if xd not in variants:
            variants.append(xd)
    tradable = _tradable_equity_symbol(sym)
    if tradable and tradable not in variants:
        variants.append(tradable)
    return variants


_symbols_name_sector_cache: dict[str, tuple[str, str]] | None = None


def _symbols_name_sector_maps() -> tuple[dict[str, str], dict[str, str]]:
    global _symbols_name_sector_cache
    if _symbols_name_sector_cache is not None:
        names, sectors = _symbols_name_sector_cache
        return names, sectors
    names: dict[str, str] = {}
    sectors: dict[str, str] = {}
    sym_meta = fetch_symbols()
    if "symbol" in sym_meta.columns:
        for _, row in sym_meta.iterrows():
            s = str(row["symbol"]).upper()
            names[s] = str(row.get("name", s) or s)
            sectors[s] = str(row.get("sector_name", "") or "UNKNOWN")
    _symbols_name_sector_cache = (names, sectors)
    return names, sectors


def _resolve_company_name(symbol: str) -> str:
    sym = str(symbol or "").upper().strip()
    names, _sectors = _symbols_name_sector_maps()
    for variant in _related_equity_symbol_variants(sym):
        nm = names.get(variant, "").strip()
        if nm and nm.upper() != variant and nm.lower() != "nan":
            return nm
    for variant in _related_equity_symbol_variants(sym):
        row = _screener_row(variant)
        if row is None:
            continue
        nm = row.get("name")
        if nm is not None and pd.notna(nm):
            text = str(nm).strip()
            if text and text.lower() != "nan":
                return text
    return sym


def _resolve_sector_name(symbol: str, fallback: str = "UNKNOWN") -> str:
    sym = str(symbol or "").upper().strip()
    _names, sectors = _symbols_name_sector_maps()
    for variant in _related_equity_symbol_variants(sym):
        sec = sectors.get(variant, "").strip()
        if sec and sec.upper() != "UNKNOWN":
            return sec
    for variant in _related_equity_symbol_variants(sym):
        row = _screener_row(variant)
        if row is None:
            continue
        sec = str(row.get("sector_name", "") or "").strip()
        if sec and sec.upper() != "UNKNOWN":
            return sec
    fb = str(fallback or "").strip()
    return fb if fb and fb.lower() != "nan" else "UNKNOWN"


def _dps_history_for_symbol_variant(sym: str) -> dict:
    yahoo_summary: dict | None = None
    psx_summary: dict | None = None
    if _yahoo_dividends_enabled():
        try:
            payments = _fetch_yahoo_dividend_payments(sym)
            if payments:
                yahoo_summary = _summarize_dps_payments(payments)
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError, KeyError, ValueError):
            pass
    if _psx_dividends_enabled():
        try:
            notices = _fetch_psx_dividend_notices(sym)
            if notices:
                psx_summary = _summarize_psx_dividend_notices(notices)
        except Exception:
            pass
    if yahoo_summary and psx_summary:
        return _merge_dps_summaries(yahoo_summary, psx_summary)
    if yahoo_summary:
        return yahoo_summary
    if psx_summary:
        return psx_summary
    return {
        "lookback_years": _DPS_LOOKBACK_YEARS,
        "years_with_dividends": 0,
        "annual_dps_newest_first": [],
        "annual_totals_by_year": {},
        "last_fiscal_total": None,
        "source": "none",
        "implied_dps": None,
    }


def _dps_history(symbol: str) -> dict:
    sym = symbol.upper()
    if sym in _dividend_history_cache:
        return _dividend_history_cache[sym]
    best: dict | None = None
    for variant in _related_equity_symbol_variants(sym):
        summary = _dps_history_for_symbol_variant(variant)
        y = int(summary.get("years_with_dividends") or 0)
        if best is None or y > int(best.get("years_with_dividends") or 0):
            best = summary
        if y >= _DPS_LOOKBACK_YEARS - 1:
            break
    summary = best or _dps_history_for_symbol_variant(sym)
    if summary.get("years_with_dividends", 0) == 0:
        implied = _implied_dps_from_screener(sym)
        if implied is None:
            for variant in _related_equity_symbol_variants(sym):
                implied = _implied_dps_from_screener(variant)
                if implied is not None and implied > 0:
                    break
        if implied is not None and implied > 0:
            summary = dict(summary)
            summary["implied_dps"] = implied
            if summary.get("source") == "none":
                summary["source"] = "screener_implied"
    _dividend_history_cache[sym] = summary
    return summary


def _dps_income_tier(summary: dict) -> str:
    """
    Income-screen tier from 5y DPS lookback:
      regular (4-5/5), mixed (2-3/5), weak (1/5), none (0/5), yield_only (no history, screener yield).
    """
    if str(summary.get("source") or "") == "screener_implied":
        imp = summary.get("implied_dps")
        if imp is not None and float(imp) > 0:
            return "yield_only"
    lb = int(summary.get("lookback_years") or _DPS_LOOKBACK_YEARS)
    y = int(summary.get("years_with_dividends") or 0)
    if y == 0:
        return "none"
    if y >= lb - 1:
        return "regular"
    if y >= 2:
        return "mixed"
    return "weak"


def _dps_tier_excluded(tier: str, mode: str) -> bool:
    """Whether an income tier is dropped under off / lenient / balanced / strict."""
    m = str(mode or "off").strip().lower()
    if m in ("off", "n", "no", "0", "none"):
        return False
    if m in ("strict", "s"):
        return tier not in ("regular",)
    if m in ("lenient", "l"):
        return tier == "none"
    # balanced (default when filtering)
    return tier in ("none", "weak")


def _parse_dps_income_filter_mode(raw: str) -> str:
    r = str(raw or "").strip().lower()
    if r in ("n", "no", "0", "off", "none"):
        return "off"
    if r in ("3", "strict", "s"):
        return "strict"
    if r in ("1", "lenient", "l"):
        return "lenient"
    if r in ("", "2", "balanced", "moderate", "y", "yes", "b", "default"):
        return "balanced"
    return "balanced"


def _dps_tier_rank_score(tier: str) -> float:
    """Higher = better for dividend-first basket ordering."""
    return {
        "regular": 4.0,
        "mixed": 3.0,
        "yield_only": 2.0,
        "weak": 1.0,
        "none": 0.0,
    }.get(str(tier), 0.0)


def prompt_dividend_eps_filter() -> bool:
    """Dividend SIP: EPS off by default; DPS + yield drive selection."""
    print(
        "\n  Dividend-first: DPS regularity and yield matter most for income names."
    )
    return prompt_yes_no(
        "Also drop stocks with EPS down 2+ years? "
        "(strict — often removes FABL/MEBL-style payers)",
        default_yes=False,
    )


def prompt_dps_income_filter() -> str:
    print(
        "\n  DPS history filter (5-year lookback — not advice):\n"
        "    1. Off — no DPS filter\n"
        "    2. Balanced — drop 0/5 and 1/5 only; keep 2/5+ (3/5, 4/5 OK) [default]\n"
        "    3. Strict — keep 4/5+ regular payers only\n"
        "    4. Lenient — drop 0/5 only (keep 1/5+ and yield-only)"
    )
    raw = prompt("DPS filter [2]: ").strip().lower() or "2"
    mode = _parse_dps_income_filter_mode(raw)
    labels = {
        "off": "off",
        "balanced": "balanced (2/5+ kept)",
        "strict": "strict (4/5+ only)",
        "lenient": "lenient (1/5+ kept)",
    }
    print(f"  Using DPS filter: {labels.get(mode, mode)}.")
    return mode


def _describe_dps_regularity(summary: dict) -> tuple[str, bool]:
    """
    Label for display; second value True = fails balanced income screen (0/5, 1/5).
    """
    lb = int(summary.get("lookback_years") or _DPS_LOOKBACK_YEARS)
    y = int(summary.get("years_with_dividends") or 0)
    tier = _dps_income_tier(summary)
    last = summary.get("last_fiscal_total")
    credits = summary.get("credits_by_year") or {}
    src = str(summary.get("source") or "")
    if last is not None:
        last_s = f" · FY DPS Rs {float(last):.2f}"
    elif credits:
        cy = datetime.now(timezone.utc).year - 1
        c = credits.get(cy) or credits.get(max(credits.keys(), default=0), 0)
        last_s = f" · {c} PSX credit(s) in latest FY" if c else ""
    else:
        last_s = ""
    if "psx" in src and last is None and y > 0:
        last_s = (last_s or "") + " (PSX notices; Rs/sh if not in title)"
    if tier == "yield_only":
        imp = summary.get("implied_dps")
        if imp:
            return (
                f"DPS ~Rs {float(imp):.2f}/sh (yield×price; no 5y history)",
                False,
            )
        return ("DPS n/a", True)
    if tier == "none":
        return ("DPS no 5y payments", True)
    if tier == "regular":
        return (f"DPS regular {y}/{lb} yrs{last_s}", False)
    if tier == "mixed":
        return (
            f"DPS mixed {y}/{lb} yrs (OK for balanced filter){last_s}",
            False,
        )
    return (f"DPS weak {y}/{lb} yrs{last_s}", True)


def _print_dps_consistency_table(symbols: list[str]) -> dict[str, tuple[str, bool]]:
    """Fetch DPS history and print a summary table; return symbol -> (note, irregular)."""
    notes: dict[str, tuple[str, bool]] = {}
    if not symbols:
        return notes
    parts = []
    if _yahoo_dividends_enabled():
        parts.append("Yahoo cash amounts")
    if _psx_dividends_enabled():
        parts.append("PSX dividend notices (dps.psx.com.pk)")
    src = " + ".join(parts) if parts else "PSX screener implied only"
    print(f"\n  Cash dividend / DPS check ({src}):\n")
    rows: list[dict] = []
    n = len(symbols)
    for i, sym in enumerate(symbols, start=1):
        sym = sym.upper()
        if i == 1 or i == n or i % 4 == 0:
            print(f"    … DPS {i}/{n}", flush=True)
        hist = _dps_history(sym)
        note, irregular = _describe_dps_regularity(hist)
        notes[sym] = (note, irregular)
        totals = hist.get("annual_totals_by_year") or {}
        credits = hist.get("credits_by_year") or {}
        years = sorted(set(totals.keys()) | set(credits.keys()), reverse=True)

        def _fy_cell(y: int) -> str:
            if y in totals:
                return f"{totals[y]:.2f}"
            if y in credits:
                return f"{credits[y]} cr"
            return "—"

        y1 = _fy_cell(years[0]) if years else "—"
        y2 = _fy_cell(years[1]) if len(years) > 1 else "—"
        rows.append(
            {
                "symbol": sym,
                "name": _resolve_company_name(sym)[:36],
                "years_paid": f"{hist.get('years_with_dividends', 0)}/{hist.get('lookback_years', 5)}",
                "fy_latest": y1,
                "fy_prior": y2,
                "dps_status": note,
            }
        )
    show = pd.DataFrame(rows)
    print(show.to_string(index=False))
    print(
        "\n  Tiers: regular = 4-5/5 yrs paid · mixed = 2-3/5 (kept on balanced filter) · "
        "weak = 1/5 · none = 0/5 · yield-only = no history, screener yield."
    )
    print(
        "  Balanced filter (default): drops weak/none only. Strict: 4/5+ only. Lenient: 0/5 only."
    )
    if not _yahoo_dividends_enabled():
        print("  (PSX_SIP_NO_YAHOO=1 — Yahoo Rs/sh amounts skipped.)")
    if not _psx_dividends_enabled():
        print("  (PSX_SIP_NO_PSX_DIV=1 — PSX notice fallback skipped.)")
    print("  fy_* = Rs DPS total when known, else PSX credit count (cr) that year.")
    return notes


def _yoy_pct(values: list[float]) -> float | None:
    if len(values) < 2 or not values[1]:
        return None
    return (values[0] - values[1]) / abs(values[1]) * 100.0


def _consecutive_eps_decline(values: list[float], years: int) -> bool:
    if len(values) < years or years < 2:
        return False
    for i in range(years - 1):
        if values[i] >= values[i + 1]:
            return False
    return True


def _describe_eps_trend(values: list[float]) -> tuple[str, bool]:
    """Human label and whether to exclude (multi-year EPS slide)."""
    if len(values) < 2:
        return ("EPS n/a", False)
    if _consecutive_eps_decline(values, 3):
        return ("EPS ↓ 3 yrs", True)
    if _consecutive_eps_decline(values, 2):
        return ("EPS ↓ 2 yrs", True)
    if len(values) >= 3 and values[0] > values[1] > values[2]:
        return ("EPS ↑ 3 yrs", False)
    oldest = values[-1]
    if oldest and abs(oldest) > 1e-9:
        ch = (values[0] - oldest) / abs(oldest) * 100.0
        if ch >= 5:
            return (f"EPS +{ch:.0f}% vs {len(values)-1}y ago", False)
        if ch <= -5:
            return (f"EPS {ch:.0f}% vs {len(values)-1}y ago", False)
    return ("EPS mixed", False)


def _eps_growth_factor(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    oldest = values[-1]
    if not oldest or abs(oldest) < 1e-9:
        return 0.0
    return (values[0] - oldest) / abs(oldest) * 100.0


def _basket_role_label(
    blue_r: float,
    div_r: float,
    growth_r: float,
    w_blue: float,
    w_div: float,
    w_growth: float,
) -> str:
    weighted = {
        "blue-chip": w_blue * blue_r,
        "dividend": w_div * div_r,
        "growth": w_growth * growth_r,
    }
    ranked = sorted(weighted.items(), key=lambda kv: kv[1], reverse=True)
    tags = [ranked[0][0]]
    if len(ranked) > 1 and ranked[1][1] >= ranked[0][1] * 0.82:
        tags.append(ranked[1][0])
    return "+".join(tags)


def _index_constituent_weights(index_name: str) -> pd.DataFrame:
    raw = fetch_indices(index_name)
    if raw is None or raw.empty or "symbol" not in raw.columns:
        return pd.DataFrame(columns=["symbol", "idx_weight"])
    out = raw[["symbol"]].copy()
    out["symbol"] = (
        out["symbol"]
        .astype(str)
        .str.upper()
        .map(_tradable_equity_symbol)
    )
    out["idx_weight"] = pd.to_numeric(raw.get("idx_weight"), errors="coerce").fillna(0.0)
    return out.groupby("symbol", as_index=False)["idx_weight"].max()


def _etf_constituent_symbols(etf_symbols: list[str]) -> set[str]:
    found: set[str] = set()
    for sym in etf_symbols:
        parsed, _html = _fetch_etf_parsed(sym.upper())
        if not parsed or parsed.get("kind") != "equity":
            continue
        for h in parsed.get("holdings") or []:
            found.add(str(h["symbol"]).upper())
    return found


def _build_screener_enriched() -> pd.DataFrame:
    scr = fetch_screener()
    if scr.empty or "symbol" not in scr.columns:
        print("  PSX screener unavailable; cannot build custom basket.")
        sys.exit(1)
    out = scr.copy()
    out["symbol"] = out["symbol"].astype(str).str.upper()
    sym_meta = fetch_symbols()
    if "sector_name" in sym_meta.columns:
        out = out.merge(
            sym_meta[["symbol", "sector_name", "name"]],
            on="symbol",
            how="left",
            suffixes=("", "_sym"),
        )
        if "name_sym" in out.columns:
            out["name"] = out["name"].fillna(out["name_sym"])
            out = out.drop(columns=["name_sym"], errors="ignore")
    else:
        out["sector_name"] = out.get("sector", "")
        out["name"] = out["symbol"]
    out["sector_name"] = out["sector_name"].fillna("UNKNOWN").astype(str)
    out["price"] = pd.to_numeric(
        out.get("price", out.get("current")), errors="coerce"
    )
    out["dividend_yield"] = pd.to_numeric(
        out.get("dividend_yield"), errors="coerce"
    ).fillna(0.0)
    out["change_1y_pct"] = pd.to_numeric(
        out.get("change_1y_pct"), errors="coerce"
    ).fillna(0.0)
    out["market_cap"] = pd.to_numeric(out.get("market_cap"), errors="coerce").fillna(0.0)
    out["pe_ratio"] = pd.to_numeric(out.get("pe_ratio"), errors="coerce")
    return out


def print_dividend_income_strategy_tips() -> None:
    print(
        """
Dividend-income checklist (research summary — not advice):
  • Prefer consistent payers (5+ years of cash dividends), not one-off specials.
  • Watch payout vs earnings; very high yield (often 15%+) can mean a cut is priced in.
  • Diversify sectors (banking, energy, fertilizer, power, consumer) — PSX is yield-heavy
    but sector shocks (rates, oil, policy) cluster.
  • PSX publishes PSXDIV20 — top dividend names by yield + history; good research universe.
  • Reinvest dividends for compounding; review holdings at least quarterly.
  • For dividend SIPs, DPS history + yield beat EPS trends unless you opt into strict EPS.
    Regular payers (e.g. banks) may show EPS dips while still paying dividends.
  • Verify payout safety on PSX company pages before you buy.

Suggested planner knobs for income focus:
  Pool: PSXDIV20 (or KSE100+KSE30 for a wider list)
  Blend: Income tilt (20% blue chip / 50% dividend / 30% growth)
  Basket size: 6–10 names · Min yield filter: 4–6% · Keep sector cap on (default 3/name)
""".strip()
    )


def choose_custom_basket_pool(
    *, dividend_profile: bool = False
) -> tuple[list[str], list[str]]:
    if dividend_profile:
        print("\nDividend-income pool: PSXDIV20 constituents (PSX Dividend 20 index).\n")
        return (["PSXDIV20"], [])

    print("\nBuild candidate pool from:\n")
    print("  1. KSE100 constituents only")
    print("  2. KSE100 + KSE30 (union — larger blue-chip pool)")
    print("  3. KSE100 + KSE30 + selected ETF equity baskets (type all at ETF pick)")
    print("  4. PSXDIV20 only (dividend-focused index)")
    print()
    while True:
        choice = prompt("Select [2]: ") or "2"
        if choice in ("1", "kse100", "100"):
            return (["KSE100"], [])
        if choice in ("2", "both", "30"):
            return (["KSE100", "KSE30"], [])
        if choice in ("3", "etf", "mix"):
            etfs = choose_etfs_multi(allow_empty=False)
            return (["KSE100", "KSE30"], [s for s, _ in etfs])
        if choice in ("4", "div", "dividend", "psxdiv20", "div20"):
            return (["PSXDIV20"], [])
        print("  Enter 1, 2, 3, or 4.")


def choose_custom_basket_mode() -> str:
    print("\nCustom basket mode:\n")
    print("  1. Standard — you pick pool, blend, and filters")
    print(
        "  2. Dividend-income aim — PSXDIV20 pool, income tilt, "
        "strategy tips (h for help anytime)"
    )
    print()
    while True:
        choice = prompt("Select [1]: ") or "1"
        if choice in ("1", "standard", "s"):
            return "standard"
        if choice in ("2", "dividend", "div", "income", "i"):
            return "dividend"
        if _is_help_choice(choice):
            print_dividend_income_strategy_tips()
            continue
        print("  Enter 1 or 2.")


def choose_custom_basket_blend() -> tuple[float, float, float]:
    print(
        "\nBlend (must sum to 100%) — index blue-chip weight, "
        "dividend yield, 1y momentum:\n"
    )
    print("  1. Balanced — 40% blue chip / 30% dividend / 30% growth")
    print("  2. Income tilt — 20% / 50% / 30%")
    print("  3. Growth tilt — 50% / 20% / 30%")
    print("  4. Enter custom percentages")
    presets = {
        "1": (40.0, 30.0, 30.0),
        "2": (20.0, 50.0, 30.0),
        "3": (50.0, 20.0, 30.0),
    }
    while True:
        choice = prompt("Select [1]: ") or "1"
        if choice in presets:
            a, b, c = presets[choice]
            return a / 100.0, b / 100.0, c / 100.0
        if choice in ("4", "custom", "c"):
            blue = prompt_non_negative_float("Blue-chip weight %", 40.0)
            div = prompt_non_negative_float("Dividend weight %", 30.0)
            gr = prompt_non_negative_float("Growth (1y) weight %", 30.0)
            total = blue + div + gr
            if total <= 0:
                print("  Weights must be > 0.")
                continue
            if abs(total - 100.0) > 0.5:
                print(f"  Sum is {total:.1f}% — re-normalizing to 100%.")
            scale = 100.0 / total
            return blue * scale / 100.0, div * scale / 100.0, gr * scale / 100.0
        print("  Enter 1–4.")


def build_smart_custom_basket(
    index_names: list[str],
    etf_symbols: list[str],
    *,
    w_blue: float,
    w_div: float,
    w_growth: float,
    target_count: int,
    min_div_yield: float,
    max_per_sector: int,
    exclude_eps_decliners: bool = True,
) -> pd.DataFrame:
    print("\nFetching index weights, screener, and prices...")
    blue_weight: dict[str, float] = {}
    for iname in index_names:
        chunk = _index_constituent_weights(iname)
        for _, row in chunk.iterrows():
            sym = str(row["symbol"]).upper()
            w = float(row["idx_weight"])
            blue_weight[sym] = max(blue_weight.get(sym, 0.0), w)

    etf_syms = _etf_constituent_symbols(etf_symbols) if etf_symbols else set()
    pool = set(blue_weight.keys()) | etf_syms
    if not pool:
        print("  Empty candidate pool.")
        sys.exit(1)

    scr = _build_screener_enriched()
    scr = scr[scr["symbol"].isin(pool)].copy()
    if scr.empty:
        print("  No screener overlap with index/ETF pool.")
        sys.exit(1)

    scr["blue_chip_score"] = scr["symbol"].map(lambda s: blue_weight.get(s, 0.0))
    if etf_syms:
        scr["etf_member"] = scr["symbol"].isin(etf_syms).astype(float)
        scr["blue_chip_score"] = scr["blue_chip_score"] + scr["etf_member"] * 0.5

    if min_div_yield > 0:
        before = len(scr)
        scr = scr[scr["dividend_yield"] >= min_div_yield].copy()
        if scr.empty:
            print(
                f"  No names with dividend yield ≥ {min_div_yield:g}% "
                f"(had {before} in pool). Lower the filter."
            )
            sys.exit(1)
        print(f"  Dividend filter ≥ {min_div_yield:g}%: {len(scr)} names remain.")

    n_eps = len(scr)
    print(
        f"  Reading annual EPS from PSX company pages ({n_eps} names)…"
    )
    eps_notes: list[str] = []
    eps_factors: list[float] = []
    eps_exclude: list[bool] = []
    skipped_eps: list[str] = []
    for i, sym in enumerate(scr["symbol"].astype(str).str.upper(), start=1):
        if i == 1 or i == n_eps or i % 20 == 0:
            print(f"    … {i}/{n_eps}", flush=True)
        eps_vals = _company_annual_eps_values(sym)
        note, exclude = _describe_eps_trend(eps_vals)
        eps_notes.append(note)
        eps_factors.append(_eps_growth_factor(eps_vals))
        if exclude_eps_decliners and exclude:
            skipped_eps.append(f"{sym} ({note})")
            eps_exclude.append(True)
        else:
            eps_exclude.append(False)

    scr = scr.copy()
    scr["eps_trend"] = eps_notes
    scr["eps_growth_pct"] = eps_factors
    if exclude_eps_decliners and skipped_eps:
        scr = scr[~pd.Series(eps_exclude, index=scr.index)].copy()
        print(
            f"  Excluded {len(skipped_eps)} name(s) with multi-year EPS declines "
            f"(e.g. {', '.join(skipped_eps[:4])}"
            + (" …" if len(skipped_eps) > 4 else "")
            + ")."
        )
    if scr.empty:
        print("  No names left after EPS filter. Try turning the EPS filter off.")
        sys.exit(1)

    blue_r = _rank_score(scr["blue_chip_score"])
    div_r = _rank_score(scr["dividend_yield"])
    price_growth_r = _rank_score(scr["change_1y_pct"])
    eps_growth_r = _rank_score(scr["eps_growth_pct"])
    growth_r = (price_growth_r + eps_growth_r) / 2.0

    scr["score"] = (
        w_blue * blue_r + w_div * div_r + w_growth * growth_r
    )
    scr["basket_role"] = [
        _basket_role_label(b, d, g, w_blue, w_div, w_growth)
        for b, d, g in zip(blue_r, div_r, growth_r)
    ]
    scr = scr.sort_values("score", ascending=False)

    picked: list[pd.Series] = []
    sector_counts: dict[str, int] = {}
    for _, row in scr.iterrows():
        sec = str(row["sector_name"])
        if sector_counts.get(sec, 0) >= max_per_sector:
            continue
        sector_counts[sec] = sector_counts.get(sec, 0) + 1
        picked.append(row)
        if len(picked) >= target_count:
            break

    if not picked:
        print("  Could not pick any stocks (check filters).")
        sys.exit(1)
    if len(picked) < target_count:
        print(
            f"  Note: only {len(picked)} names available "
            f"(asked for {target_count}) after sector/EPS filters."
        )

    sel = pd.DataFrame(picked)
    price_ok: list[dict] = []
    for _, row in sel.iterrows():
        sym = str(row["symbol"]).upper()
        price = row.get("price")
        if price is None or (isinstance(price, float) and math.isnan(price)):
            price = _fallback_price_from_history(sym)
        if price is None:
            continue
        price_ok.append(
            {
                "symbol": sym,
                "name": str(row.get("name", sym)),
                "sector_name": str(row.get("sector_name", "UNKNOWN")),
                "price": float(price),
                "idx_weight": float(row["score"]),
                "dividend_yield": float(row.get("dividend_yield", 0.0)),
                "change_1y_pct": float(row.get("change_1y_pct", 0.0)),
                "blue_chip_score": float(row.get("blue_chip_score", 0.0)),
                "basket_role": str(row.get("basket_role", "")),
                "eps_trend": str(row.get("eps_trend", "")),
            }
        )

    if not price_ok:
        print("  Could not price any selected names.")
        sys.exit(1)

    df = pd.DataFrame(price_ok)
    total = df["idx_weight"].sum()
    if total <= 0:
        df["idx_weight"] = 1.0
    else:
        df["idx_weight"] = df["idx_weight"] / total * 100.0
    return df.sort_values("idx_weight", ascending=False).reset_index(drop=True)


def _dataframe_to_sip_basket(
    sel: pd.DataFrame,
    *,
    weight_col: str,
    basket_id: str,
    title: str,
) -> pd.DataFrame:
    """Normalize weights and ensure prices for SIP planning."""
    price_ok: list[dict] = []
    for _, row in sel.iterrows():
        sym = str(row["symbol"]).upper()
        price = row.get("price")
        if price is None or (isinstance(price, float) and math.isnan(price)):
            price = _fallback_price_from_history(sym)
        if price is None:
            continue
        w = float(row.get(weight_col, 0.0) or 0.0)
        if w <= 0:
            continue
        price_ok.append(
            {
                "symbol": sym,
                "name": str(row.get("name", sym)),
                "sector_name": str(row.get("sector_name", "UNKNOWN")),
                "price": float(price),
                "idx_weight": w,
                "dividend_yield": float(row.get("dividend_yield", 0.0) or 0.0),
            }
        )
    if not price_ok:
        print("  Could not price any selected names.")
        sys.exit(1)
    df = pd.DataFrame(price_ok)
    total = df["idx_weight"].sum()
    if total <= 0:
        df["idx_weight"] = 100.0 / len(df)
    else:
        df["idx_weight"] = df["idx_weight"] / total * 100.0
    return df.sort_values("idx_weight", ascending=False).reset_index(drop=True)


def build_yield_weighted_dividend_basket(
    pool_symbols: set[str],
    *,
    target_count: int,
    min_div_yield: float,
    max_per_sector: int,
    exclude_eps_decliners: bool,
) -> pd.DataFrame:
    scr = _build_screener_enriched()
    scr = scr[scr["symbol"].isin(pool_symbols)].copy()
    if min_div_yield > 0:
        scr = scr[scr["dividend_yield"] >= min_div_yield].copy()
    if scr.empty:
        print("  No screener names match the dividend pool/filters.")
        sys.exit(1)

    if exclude_eps_decliners:
        keep_rows: list[pd.Series] = []
        n = len(scr)
        print(f"  Checking annual EPS on PSX company pages ({n} names)…")
        for i, (_, row) in enumerate(scr.iterrows(), start=1):
            if i == 1 or i == n or i % 15 == 0:
                print(f"    … {i}/{n}", flush=True)
            sym = str(row["symbol"]).upper()
            eps_vals = _company_annual_eps_values(sym)
            _note, exclude = _describe_eps_trend(eps_vals)
            if not exclude:
                keep_rows.append(row)
        if not keep_rows:
            print("  All names failed EPS quality filter.")
            sys.exit(1)
        scr = pd.DataFrame(keep_rows)

    scr = scr.sort_values("dividend_yield", ascending=False)
    picked: list[pd.Series] = []
    sector_counts: dict[str, int] = {}
    for _, row in scr.iterrows():
        sec = str(row["sector_name"])
        if sector_counts.get(sec, 0) >= max_per_sector:
            continue
        sector_counts[sec] = sector_counts.get(sec, 0) + 1
        picked.append(row)
        if len(picked) >= target_count:
            break
    if not picked:
        print("  Could not pick any stocks.")
        sys.exit(1)
    sel = pd.DataFrame(picked)
    sel["idx_weight"] = sel["dividend_yield"].clip(lower=0.01)
    return _dataframe_to_sip_basket(
        sel, weight_col="idx_weight", basket_id="", title=""
    )


def _running_from_stdin() -> bool:
    """True when the planner was piped (curl … | python3 -), not loaded from a .py file."""
    try:
        base = os.path.basename(os.path.abspath(__file__))
    except NameError:
        return True
    return base in ("<stdin>", "-")


def _planner_root() -> str:
    if _running_from_stdin():
        return os.getcwd()
    return os.path.dirname(os.path.abspath(__file__))


def _planner_data_dir() -> str:
    return os.path.join(_planner_root(), "data")


def _psx_sip_cache_dir() -> str:
    return os.path.expanduser("~/.psx-sip")


def _remote_repo_raw_base() -> str:
    raw = os.environ.get("PSX_SIP_REPO_RAW", "").strip()
    return (raw or DEFAULT_PSX_SIP_REPO_RAW).rstrip("/")


def _fetch_url_text(url: str, *, timeout: float = 45.0) -> str | None:
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": f"psx-sip-planner/{__version__}"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8")
    except (urllib.error.URLError, OSError, TimeoutError, ValueError):
        return None


def _ensure_dividend_fund_json_cached(*, quiet: bool = False) -> str | None:
    """
    Download data/dividend_fund_composition.json for curl | python3 runs.
    Cached under ~/.psx-sip/ (refreshed when missing or PSX_SIP_REFRESH_FUNDS=1).
    """
    cache_dir = _psx_sip_cache_dir()
    dest = os.path.join(cache_dir, "dividend_fund_composition.json")
    refresh = os.environ.get("PSX_SIP_REFRESH_FUNDS", "").strip().lower() in (
        "1",
        "yes",
        "true",
    )
    if (
        not refresh
        and os.path.isfile(dest)
        and os.path.getsize(dest) > 200
    ):
        return dest
    url = f"{_remote_repo_raw_base()}/data/dividend_fund_composition.json"
    if not quiet:
        print(
            f"\n  Fetching dividend fund reference (one-time cache → {dest})…",
            flush=True,
        )
    body = _fetch_url_text(url)
    if not body:
        if not quiet:
            print("  Could not download fund reference JSON (check network / PSX_SIP_REPO_RAW).")
        return dest if os.path.isfile(dest) else None
    try:
        payload = json.loads(body)
        funds = payload.get("funds") or []
        if not funds:
            raise ValueError("empty funds")
    except (json.JSONDecodeError, ValueError):
        if not quiet:
            print("  Downloaded fund reference JSON was invalid.")
        return dest if os.path.isfile(dest) else None
    os.makedirs(cache_dir, exist_ok=True)
    with open(dest, "w", encoding="utf-8") as fh:
        fh.write(body)
    if not quiet:
        print(f"  Saved fund reference: {dest}")
    return dest


def _load_bundled_dividend_funds_module() -> list[dict]:
    """Load dividend_funds_data.py from the same folder as this script (any cwd)."""
    import importlib.util

    mod_path = os.path.join(_planner_root(), "dividend_funds_data.py")
    if not os.path.isfile(mod_path):
        return []
    try:
        spec = importlib.util.spec_from_file_location(
            "psx_sip_dividend_funds_data", mod_path
        )
        if spec is None or spec.loader is None:
            return []
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        funds = getattr(mod, "BUNDLED_DIVIDEND_FUNDS", None) or []
        return [f for f in funds if isinstance(f, dict) and f.get("holdings")]
    except Exception:
        return []


def _load_dividend_fund_compositions() -> list[dict]:
    """Top-10 equity snapshots from FMR reports (not live NAV/returns)."""
    candidates: list[str] = []
    env_path = os.environ.get("PSX_SIP_DIVIDEND_FUNDS_JSON", "").strip()
    if env_path:
        candidates.append(os.path.expanduser(env_path))
    candidates.append(os.path.join(_planner_data_dir(), "dividend_fund_composition.json"))
    candidates.append(os.path.expanduser("~/.psx-sip/dividend_fund_composition.json"))
    for path in candidates:
        if not path or not os.path.isfile(path):
            continue
        try:
            with open(path, encoding="utf-8") as fh:
                payload = json.load(fh)
            funds = payload.get("funds") or []
            out = [f for f in funds if isinstance(f, dict) and f.get("holdings")]
            if out:
                return out
        except (OSError, json.JSONDecodeError):
            continue
    cached = _ensure_dividend_fund_json_cached(
        quiet=not _running_from_stdin()
    )
    if cached and os.path.isfile(cached):
        try:
            with open(cached, encoding="utf-8") as fh:
                payload = json.load(fh)
            funds = payload.get("funds") or []
            out = [f for f in funds if isinstance(f, dict) and f.get("holdings")]
            if out:
                return out
        except (OSError, json.JSONDecodeError):
            pass
    return _load_bundled_dividend_funds_module()


def _print_psxdiv20_composition_table(*, enrich_screener: bool) -> set[str]:
    print("\nFetching PSXDIV20 constituents (live PSX index weights)…")
    raw = fetch_indices("PSXDIV20")
    if raw is None or raw.empty:
        print("  Could not load PSXDIV20.")
        return set()
    sym_meta = fetch_symbols()
    names: dict[str, str] = {}
    sectors: dict[str, str] = {}
    if "symbol" in sym_meta.columns:
        for _, row in sym_meta.iterrows():
            s = str(row["symbol"]).upper()
            names[s] = str(row.get("name", s))
            sectors[s] = str(row.get("sector_name", "") or "")
    scr_by_sym: dict[str, pd.Series] = {}
    if enrich_screener:
        scr = _build_screener_enriched()
        for _, row in scr.iterrows():
            scr_by_sym[str(row["symbol"]).upper()] = row

    rows: list[dict] = []
    symbols: set[str] = set()
    for _, row in raw.sort_values("idx_weight", ascending=False).iterrows():
        sym = str(row["symbol"]).upper()
        symbols.add(sym)
        w = float(pd.to_numeric(row.get("idx_weight"), errors="coerce") or 0.0)
        entry: dict = {
            "symbol": sym,
            "name": names.get(sym, sym),
            "sector": sectors.get(sym, ""),
            "weight_pct": w,
        }
        if enrich_screener and sym in scr_by_sym:
            sr = scr_by_sym[sym]
            entry["div_yield"] = float(sr.get("dividend_yield", 0.0) or 0.0)
            pe = sr.get("pe_ratio")
            entry["pe"] = float(pe) if pe is not None and pd.notna(pe) else None
        rows.append(entry)

    print(f"\n--- PSXDIV20 (live index) ---")
    for r in rows:
        line = f"  {r['symbol']:<6} {r['weight_pct']:5.2f}%  {r['name'][:42]}"
        if enrich_screener:
            dy = r.get("div_yield", 0.0)
            pe = r.get("pe")
            pe_s = f"{pe:.1f}" if pe is not None else "—"
            line += f"  yield {dy:.2f}%  P/E {pe_s}"
        print(line)
    print(f"\n  ({len(rows)} constituents)")
    return symbols


def _print_mutual_fund_composition(fund: dict, *, enrich_screener: bool) -> set[str]:
    symbols: set[str] = set()
    holdings = sorted(fund.get("holdings") or [], key=lambda h: h.get("rank", 99))
    scr_by_sym: dict[str, pd.Series] = {}
    if enrich_screener:
        scr = _build_screener_enriched()
        for _, row in scr.iterrows():
            scr_by_sym[str(row["symbol"]).upper()] = row

    short = str(fund.get("id") or fund.get("name", "Fund"))[:48]
    as_of = fund.get("report_as_of", "")
    hdr = f"--- {short} (FMR top-10"
    if as_of:
        hdr += f", {as_of}"
    hdr += ") ---"
    print(f"\n{hdr}")
    for h in holdings:
        sym = str(h.get("symbol", "")).upper()
        if not sym:
            continue
        symbols.add(sym)
        wt = float(h.get("weight_pct", 0.0) or 0.0)
        name = str(h.get("company_name", sym))[:40]
        rank = h.get("rank", "")
        line = f"  {rank:>2}. {sym:<6} {wt:5.2f}%  {name}"
        if enrich_screener and sym in scr_by_sym:
            sr = scr_by_sym[sym]
            dy = float(sr.get("dividend_yield", 0.0) or 0.0)
            pe = sr.get("pe_ratio")
            pe_s = f"{pe:.1f}" if pe is not None and pd.notna(pe) else "—"
            line += f"  yield {dy:.2f}%  P/E {pe_s}"
        print(line)
    return symbols


def _mf_equity_weight_map(fund: dict) -> dict[str, float]:
    """FMR top-10 weights renormalized to 100% (for a DIY stock basket)."""
    weights: dict[str, float] = {}
    for h in fund.get("holdings") or []:
        sym = _mf_fmr_symbol(
            str(h.get("symbol", "")),
            str(h.get("company_name", "")),
        )
        if not sym:
            continue
        tradable = _tradable_equity_symbol(sym)
        weights[tradable] = weights.get(tradable, 0.0) + float(
            h.get("weight_pct", 0.0) or 0.0
        )
    return _canonicalize_weight_map(weights, renormalize=True)


_screener_symbol_set_cache: set[str] | None = None


def _screener_symbol_set() -> set[str]:
    global _screener_symbol_set_cache
    if _screener_symbol_set_cache is None:
        scr = fetch_screener()
        if scr.empty or "symbol" not in scr.columns:
            _screener_symbol_set_cache = set()
        else:
            _screener_symbol_set_cache = set(
                scr["symbol"].astype(str).str.upper()
            )
    return _screener_symbol_set_cache


def _mf_fmr_symbol(raw_symbol: str, company_name: str = "") -> str:
    """Fix common FMR typos before mapping to tradable PSX tickers."""
    sym = str(raw_symbol or "").upper().strip()
    name_l = str(company_name or "").lower()
    if sym == "ABL" and "abbott" in name_l:
        return "ABOT"
    return sym


def _tradable_equity_symbol(sym: str) -> str:
    """
    Map index/FMR tickers to a PSX screener symbol.
    PSX index rows often use *XD suffixes while funds use the plain ticker.
    """
    sym = str(sym or "").upper().strip()
    if not sym:
        return sym
    scr_syms = _screener_symbol_set()
    if sym in scr_syms:
        return sym
    xd = sym + "XD"
    if xd in scr_syms:
        return xd
    if sym.endswith("XD"):
        base = sym[:-2]
        if base in scr_syms:
            return base
    return sym


def _canonicalize_weight_map(
    weights: dict[str, float], *, renormalize: bool = True
) -> dict[str, float]:
    merged: dict[str, float] = {}
    for sym, w in weights.items():
        tradable = _tradable_equity_symbol(sym)
        merged[tradable] = merged.get(tradable, 0.0) + float(w or 0.0)
    if renormalize:
        return _renormalize_weights(merged)
    return merged


def _psxdiv20_weight_map() -> dict[str, float]:
    raw = fetch_indices("PSXDIV20")
    if raw is None or raw.empty:
        return {}
    out: dict[str, float] = {}
    for _, row in raw.iterrows():
        sym = str(row["symbol"]).upper()
        w = float(pd.to_numeric(row.get("idx_weight"), errors="coerce") or 0.0)
        tradable = _tradable_equity_symbol(sym)
        out[tradable] = max(out.get(tradable, 0.0), w)
    total = sum(out.values())
    if total <= 0:
        return out
    raw_map = {s: w / total * 100.0 for s, w in out.items()}
    return _canonicalize_weight_map(raw_map, renormalize=True)


def _blue_chip_index_weight_map() -> dict[str, float]:
    """KSE100 + KSE30 index weights (tradable symbols) for blue-chip ranking."""
    combined: dict[str, float] = {}
    for index_name in ("KSE100", "KSE30"):
        raw = fetch_indices(index_name)
        if raw is None or raw.empty:
            continue
        for _, row in raw.iterrows():
            sym = _tradable_equity_symbol(str(row["symbol"]).upper())
            w = float(pd.to_numeric(row.get("idx_weight"), errors="coerce") or 0.0)
            combined[sym] = max(combined.get(sym, 0.0), w)
    total = sum(combined.values())
    if total <= 0:
        return combined
    return {s: w / total * 100.0 for s, w in combined.items()}


def _listed_in_is_blue_chip(listed_in: str) -> bool:
    li = str(listed_in or "").upper()
    return "KSE100" in li or "KSE30" in li


def _renormalize_weights(weights: dict[str, float]) -> dict[str, float]:
    total = sum(weights.values())
    if total <= 0:
        return weights
    return {s: w / total * 100.0 for s, w in weights.items()}


def _kmi30_symbol_set() -> set[str]:
    """Shariah filter: KMI30 index membership only (PSX screener listed_in)."""
    return _shariah_symbol_set("kmi30")


def _shariah_symbol_set(pool: str = "kmi30") -> set[str]:
    """Shariah-listed symbols from PSX screener (KMI30 or wider KMI All Share)."""
    scr = fetch_screener()
    if scr.empty or "listed_in" not in scr.columns:
        return set()
    wide = str(pool).strip().lower() in ("kmi_all", "all", "allshare", "2", "b")
    found: set[str] = set()
    for _, row in scr.iterrows():
        li = str(row.get("listed_in", "")).upper()
        if wide:
            if "KMI30" in li or "KMIALLSHR" in li:
                found.add(str(row["symbol"]).upper())
        elif "KMI30" in li:
            found.add(str(row["symbol"]).upper())
    return found


def _apply_shariah_filter_to_div20(
    div20: dict[str, float], *, enabled: bool
) -> dict[str, float]:
    return _apply_shariah_filter_weights(div20, enabled=enabled, context="PSXDIV20")


def _apply_shariah_filter_weights(
    weights: dict[str, float],
    *,
    enabled: bool,
    context: str = "basket",
    shariah_pool: str = "kmi30",
    quiet: bool = False,
) -> dict[str, float]:
    if not enabled or not weights:
        return weights
    kmi = _shariah_symbol_set(shariah_pool)
    pool_label = "KMI All Share" if shariah_pool != "kmi30" else "KMI30"
    if not kmi:
        if not quiet:
            print(f"  Note: could not load {pool_label} list — {context} unfiltered.")
        return weights
    filtered = {s: w for s, w in weights.items() if s in kmi}
    dropped = set(weights.keys()) - set(filtered.keys())
    if dropped and not quiet:
        print(
            f"  Shariah filter ({pool_label}): dropped {len(dropped)} name(s) from {context} "
            f"(e.g. {', '.join(sorted(dropped)[:6])}"
            + (" …" if len(dropped) > 6 else "")
            + ")."
        )
    if not filtered:
        if not quiet:
            print(f"  No names left in {context} after Shariah filter.")
        return {}
    return _renormalize_weights(filtered)


def _blend_weight_maps(
    mf: dict[str, float],
    div20: dict[str, float],
    mode: str,
) -> dict[str, float]:
    mf = _canonicalize_weight_map(mf, renormalize=False)
    div20 = _canonicalize_weight_map(div20, renormalize=False)
    if mode == "mf":
        return _renormalize_weights(mf)
    if mode == "div20":
        return _renormalize_weights(div20)
    if mode == "overlap":
        keys = set(mf.keys()) & set(div20.keys())
        if not keys:
            return {}
        blended = {k: (mf[k] + div20[k]) / 2.0 for k in keys}
        return _renormalize_weights(blended)
    if mode == "combined":
        keys = set(mf.keys()) | set(div20.keys())
        blended = {
            k: mf.get(k, 0.0) * 0.5 + div20.get(k, 0.0) * 0.5 for k in keys
        }
        return _renormalize_weights(blended)
    # union — sum weights from each source (missing side = 0)
    keys = set(mf.keys()) | set(div20.keys())
    blended = {k: mf.get(k, 0.0) + div20.get(k, 0.0) for k in keys}
    return _renormalize_weights(blended)


def _fund_is_shariah(fund: dict) -> bool:
    return str(fund.get("shariah_scope", "")).strip().lower() == "shariah"


def _dataframe_from_weight_map(
    weights: dict[str, float], *, title: str
) -> pd.DataFrame:
    if not weights:
        print("  No symbols in basket.")
        sys.exit(1)
    sym_meta = fetch_symbols()
    names: dict[str, str] = {}
    sectors: dict[str, str] = {}
    if "symbol" in sym_meta.columns:
        for _, row in sym_meta.iterrows():
            s = str(row["symbol"]).upper()
            names[s] = str(row.get("name", s))
            sectors[s] = str(row.get("sector_name", "") or "UNKNOWN")

    scr, price_col = _market_price_index()
    rows: list[dict] = []
    for sym, wt in sorted(weights.items(), key=lambda kv: kv[1], reverse=True):
        price, _src = _resolve_live_price(sym, scr, price_col)
        if price is None:
            print(f"  Skipping {sym} — no price.")
            continue
        rows.append(
            {
                "symbol": sym,
                "name": _resolve_company_name(sym),
                "sector_name": _resolve_sector_name(
                    sym, sectors.get(sym, "UNKNOWN")
                ),
                "price": float(price),
                "idx_weight": float(wt),
            }
        )
    if not rows:
        print("  Could not price any names in basket.")
        sys.exit(1)
    df = pd.DataFrame(rows)
    df["idx_weight"] = df["idx_weight"] / df["idx_weight"].sum() * 100.0
    return df.sort_values("idx_weight", ascending=False).reset_index(drop=True)


def _sorted_dividend_funds(funds: list[dict]) -> list[dict]:
    def sort_key(f: dict) -> tuple[int, str]:
        name = str(f.get("name", ""))
        if "MDYP" in name:
            return (0, name)
        if str(f.get("shariah_scope", "")).lower() == "shariah":
            return (1, name)
        return (2, name)

    return sorted(funds, key=sort_key)


def _choose_dividend_mutual_fund(funds: list[dict]) -> dict:
    funds = _sorted_dividend_funds(funds)
    mdyp = [f for f in funds if "MDYP" in str(f.get("name", ""))]
    default = mdyp[0] if mdyp else funds[0]
    if len(funds) == 1:
        return funds[0]
    print("\n  Mutual fund reference (FMR top-10 snapshot):")
    for i, f in enumerate(funds, start=1):
        label = str(f.get("id") or f.get("name", ""))[:50]
        scope = str(f.get("shariah_scope", "") or "").strip()
        tag = f" · {scope}" if scope else ""
        mark = " [default]" if f is default else ""
        print(f"    {i}. {label}{tag}{mark}")
    while True:
        raw = prompt("Fund [1]: ").strip() or "1"
        try:
            idx = int(raw) - 1
        except ValueError:
            if raw.lower() in ("mdyp", "meezan") and mdyp:
                return mdyp[0]
            print("  Enter a number from the list.")
            continue
        if 0 <= idx < len(funds):
            return funds[idx]
        print("  Out of range.")


def run_dividend_mf_psxdiv20_sip_flow() -> None:
    """Show MF + PSXDIV20, then set target stock weights and continue to SIP."""
    funds = _load_dividend_fund_compositions()
    if not funds:
        print(
            "\n  Mutual fund snapshots missing (need dividend_funds_data.py). "
            "Using PSXDIV20 only."
        )
        df, meta = load_index_frame("PSXDIV20")
        print_overview_tables(df, "PSXDIV20")
        _run_benchmark_sip_flow(
            df,
            meta,
            source_type="index",
            source_id="PSXDIV20",
            universe_label="PSXDIV20",
            weight_label="index",
            empty_exclusion_msg="All constituents were excluded.",
            exclusion_unit="constituents",
        )
        return

    fund = _choose_dividend_mutual_fund(funds)
    mf_label = str(fund.get("id") or "MF")[:40]
    shariah_fund = _fund_is_shariah(fund)
    if shariah_fund:
        print("\n  Shariah fund selected — PSXDIV20 will be filtered to KMI30 names when mixed.")
    print("\n  Reference lists (target weights — not fund performance):\n")
    _print_mutual_fund_composition(fund, enrich_screener=False)
    _print_psxdiv20_composition_table(enrich_screener=False)

    mf_w = _mf_equity_weight_map(fund)
    overlap_n = len(set(mf_w.keys()) & set(_psxdiv20_weight_map().keys()))
    print(
        f"\n  Fund has {len(mf_w)} stocks. Overlap with PSXDIV20: {overlap_n} "
        f"(overlap-only modes use just those {overlap_n}, not all 10 fund names)."
    )
    print(
        "\n  Target weights for your stock SIP:\n"
        "    1. Mutual fund top-10 only — all 10 FMR stocks (recommended)\n"
        "    2. PSXDIV20 index only (live)\n"
        "    3. Fund + PSXDIV20 — 50/50 blend on all symbols from both lists\n"
        "    4. Only symbols in BOTH lists (small basket — usually not what you want)\n"
        "    5. Fund + PSXDIV20 — sum weights from both, then renormalize"
    )
    pick = prompt("Weights [1]: ").strip() or "1"
    mode_map = {
        "1": "mf",
        "a": "mf",
        "mf": "mf",
        "fund": "mf",
        "2": "div20",
        "b": "div20",
        "div20": "div20",
        "index": "div20",
        "3": "combined",
        "combined": "combined",
        "mix": "combined",
        "c": "combined",
        "4": "overlap",
        "overlap": "overlap",
        "both": "overlap",
        "5": "union",
        "d": "union",
        "union": "union",
        "all": "union",
    }
    blend_mode = mode_map.get(pick.lower(), "mf")
    div_w = _psxdiv20_weight_map()
    if not div_w:
        print("  Could not load PSXDIV20; using fund weights only.")
        blend_mode = "mf"
    elif blend_mode != "mf" and shariah_fund:
        div_w = _apply_shariah_filter_to_div20(div_w, enabled=True)
        if not div_w and blend_mode != "mf":
            print("  Falling back to mutual fund top-10 only.")
            blend_mode = "mf"
    target_w = _blend_weight_maps(mf_w, div_w, blend_mode)
    if shariah_fund:
        target_w = _apply_shariah_filter_weights(
            target_w, enabled=True, context="merged basket"
        )
    if not target_w:
        print("  Empty basket — try option 1 (full fund top-10) or 3 (combined).")
        return

    mode_labels = {
        "mf": f"{mf_label} top-10 (all FMR stocks)",
        "div20": "PSXDIV20 index"
        + (" (KMI30-filtered)" if shariah_fund else ""),
        "overlap": f"Overlap only ({overlap_n} symbols)",
        "combined": f"{mf_label} + PSXDIV20 (50/50 blend)"
        + (" · Shariah PSXDIV20" if shariah_fund else ""),
        "union": f"{mf_label} + PSXDIV20 (merged weights)"
        + (" · Shariah PSXDIV20" if shariah_fund else ""),
    }
    title = mode_labels.get(blend_mode, "Dividend blend")
    df = _dataframe_from_weight_map(target_w, title=title)

    mf_syms = set(mf_w.keys())
    from_mf = [s for s in df["symbol"] if s in mf_syms]
    print(f"\n--- Your target basket: {title} ({len(df)} stocks) ---")
    if blend_mode != "div20":
        print(f"  Includes {len(from_mf)} of {len(mf_syms)} mutual fund top-10 names.")
    show = df[["symbol", "name", "idx_weight"]].copy()
    show["idx_weight"] = show["idx_weight"].map(lambda w: f"{w:.2f}%")
    show = show.rename(columns={"idx_weight": "target_weight"})
    print(show.to_string(index=False))

    shariah_fund = _fund_is_shariah(fund)
    df = prompt_dividend_basket_use_or_replace(
        df,
        title=title,
        replace_ctx={
            "shariah_only": shariah_fund,
            "shariah_pool": "kmi_all" if shariah_fund else "kmi30",
            "min_div_yield": 0.0,
            "max_pe": None,
            "exclude_eps": False,
            "dps_income_filter": "off",
            "weight_mode": "index",
            "prefer_blue_chip": True,
        },
    )
    if df is None:
        return

    basket_id = re.sub(r"[^\w\-]+", "_", f"div_{blend_mode}_{mf_label}")[:36]
    meta = {
        "index_name": basket_id,
        "as_of": _utc_now_iso(),
        "title": title,
        "parsed_holdings": [],
        "missing_price_symbols": [],
        "dividend_blend": blend_mode,
        "sip_profile": "dividend_basket",
    }
    _run_benchmark_sip_flow(
        df,
        meta,
        source_type="custom",
        source_id=basket_id,
        universe_label="dividend basket",
        weight_label="target",
        empty_exclusion_msg="All names were excluded. Nothing to allocate.",
        exclusion_unit="names",
    )


def run_dividend_reference_composition_flow() -> None:
    """Show FMR top-10 stock lists + PSXDIV20 — no SIP, no performance."""
    funds = _load_dividend_fund_compositions()
    if not funds:
        root = _planner_root()
        print(
            "\n  Could not load mutual fund holdings.\n"
            f"  Run the planner from your psx-sip folder (script at {root})\n"
            "  and ensure dividend_funds_data.py is beside psx_sip_planner.py.\n"
            "  Or: git pull in psx-sip, then re-run."
        )
        _print_psxdiv20_composition_table(enrich_screener=False)
        return

    print(
        "\n  Holdings only (% of fund assets, from saved FMR snapshots).\n"
        "  Weights are not live — update data/ when new FMRs are published."
    )
    choice = prompt(
        "Show [1] MDYP + PSXDIV20  [2] All funds + PSXDIV20  [3] PSXDIV20 only [1]: "
    ) or "1"

    enrich_screener = False
    if choice not in ("3", "div20", "index"):
        extra = prompt("Add screener yield & P/E on each line? [y/N]: ").lower()
        enrich_screener = extra in ("y", "yes")

    if choice in ("3", "div20", "index"):
        _print_psxdiv20_composition_table(enrich_screener=enrich_screener)
        return

    if choice in ("2", "all", "a"):
        to_show = funds
    else:
        mdyp = [f for f in funds if "MDYP" in str(f.get("name", ""))]
        to_show = mdyp if mdyp else funds[:1]

    mdyp_syms: set[str] = set()
    for f in to_show:
        mdyp_syms |= _print_mutual_fund_composition(f, enrich_screener=enrich_screener)
    div20_syms = _print_psxdiv20_composition_table(enrich_screener=enrich_screener)
    if mdyp_syms and div20_syms:
        both = sorted(mdyp_syms & div20_syms)
        print(
            f"\n  Symbols in both fund list and PSXDIV20 ({len(both)}): "
            + ", ".join(both)
        )


def _load_dividend_pool_from_holdings() -> set[str]:
    env_path = os.environ.get("PSX_SIP_HOLDINGS", "").strip()
    if env_path:
        path = os.path.expanduser(env_path)
        print(f"\n  Using PSX_SIP_HOLDINGS: {path}")
    else:
        print("\n  Load holdings to screen (snapshot or plan CSV).")
        path = prompt_existing_csv_path("holdings")
    if not os.path.isfile(path):
        print(f"  File not found: {path}")
        return set()
    prior, _, _ = _load_prior_shares_from_any_csv(path, "custom", "holdings")
    return {
        _tradable_equity_symbol(s) for s in prior.keys() if prior.get(s, 0) > 0
    }


def _dividend_screen_candidate_weights(
    *,
    universe_mode: str,
    shariah_only: bool,
    shariah_pool: str,
    holdings_symbols: set[str] | None,
) -> tuple[dict[str, float], dict[str, float]]:
    """
    Candidate symbols -> index reference weight (0 if not in PSXDIV20).
    Second dict is full PSXDIV20 (tradable, optionally KMI30-filtered) for weight lookup.
    """
    div_w = _psxdiv20_weight_map()
    if not div_w:
        print("  Could not load PSXDIV20.")
        sys.exit(1)
    div_ref = div_w
    if shariah_only:
        div_ref = _apply_shariah_filter_weights(
            div_w, enabled=True, context="PSXDIV20", shariah_pool=shariah_pool
        ) or {}
    kmi = _shariah_symbol_set(shariah_pool) if shariah_only else None

    if holdings_symbols:
        tradable = {_tradable_equity_symbol(s) for s in holdings_symbols}
        weights: dict[str, float] = {}
        for sym in sorted(tradable):
            if shariah_only and kmi is not None and sym not in kmi:
                continue
            weights[sym] = float(div_ref.get(sym, 0.0))
        if not weights:
            print("  No holdings left after Shariah (KMI30) filter.")
            sys.exit(1)
        return weights, div_ref

    if universe_mode == "div20_only":
        if not div_ref:
            sys.exit(1)
        return dict(div_ref), div_ref

    scr = _build_screener_enriched()
    if universe_mode == "kmi30_screen":
        weights = dict(div_ref)
        for _, row in scr.iterrows():
            sym = str(row["symbol"]).upper()
            if sym in weights:
                continue
            if shariah_only and kmi is not None and sym not in kmi:
                continue
            dy = float(pd.to_numeric(row.get("dividend_yield"), errors="coerce") or 0.0)
            if dy <= 0:
                continue
            weights[sym] = 0.0
        return weights, div_ref

    # div20_expand — PSXDIV20 first; backfill from screener in basket builder
    if not div_ref:
        sys.exit(1)
    return dict(div_ref), div_ref


def _build_psxdiv20_screened_basket(
    *,
    target_count: int,
    shariah_only: bool,
    shariah_pool: str,
    pool_symbols: set[str] | None,
    universe_mode: str,
    min_div_yield: float,
    max_pe: float | None,
    exclude_eps_decliners: bool,
    dps_income_filter: str,
    weight_mode: str,
    prefer_blue_chip: bool = True,
    max_per_sector: int = CUSTOM_BASKET_MAX_PER_SECTOR,
) -> pd.DataFrame:
    candidates, div_ref = _dividend_screen_candidate_weights(
        universe_mode=universe_mode,
        shariah_only=shariah_only,
        shariah_pool=shariah_pool,
        holdings_symbols=pool_symbols,
    )
    expand_universe = universe_mode in ("div20_expand", "kmi30_screen") and not pool_symbols

    scr = _build_screener_enriched()
    scr_by = {str(r["symbol"]).upper(): r for _, r in scr.iterrows()}
    kmi = _shariah_symbol_set(shariah_pool) if shariah_only else None
    blue_chip_w = _blue_chip_index_weight_map()

    def _screener_yield(sym: str) -> float:
        row = scr_by.get(sym)
        if row is None:
            return 0.0
        return float(pd.to_numeric(row.get("dividend_yield"), errors="coerce") or 0.0)

    def _blue_chip_rank(sym: str) -> float:
        row = scr_by.get(sym)
        li = str(row.get("listed_in", "")) if row is not None else ""
        bc = float(blue_chip_w.get(sym, 0.0))
        if _listed_in_is_blue_chip(li):
            bc = max(bc, 0.01)
        return bc

    def _candidate_sort_key(sym: str, idx_w: float) -> tuple[float, ...]:
        dy = _screener_yield(sym)
        bc = _blue_chip_rank(sym)
        tier = _dps_income_tier(_dps_history(sym))
        dps_score = _dps_tier_rank_score(tier)
        if weight_mode == "yield":
            return (dy, dps_score, bc, idx_w)
        return (idx_w, dps_score, dy, bc)

    ranked = sorted(
        candidates.items(),
        key=lambda kv: _candidate_sort_key(kv[0], kv[1]),
        reverse=True,
    )
    picked: list[tuple[str, float, pd.Series]] = []
    skipped: list[str] = []
    sector_counts: dict[str, int] = {}

    def _passes_fundamentals(sym: str) -> tuple[bool, pd.Series | None]:
        row = scr_by.get(sym)
        if row is None:
            skipped.append(f"{sym} (not on screener)")
            return False, None
        dy = float(pd.to_numeric(row.get("dividend_yield"), errors="coerce") or 0.0)
        if min_div_yield > 0 and dy < min_div_yield:
            skipped.append(f"{sym} (yield {dy:.2f}% < {min_div_yield:g}%)")
            return False, row
        if max_pe is not None:
            pe = pd.to_numeric(row.get("pe_ratio"), errors="coerce")
            if pd.notna(pe) and float(pe) > max_pe:
                skipped.append(f"{sym} (P/E {float(pe):.1f} > {max_pe:g})")
                return False, row
        if dps_income_filter != "off":
            hist = _dps_history(sym)
            tier = _dps_income_tier(hist)
            if _dps_tier_excluded(tier, dps_income_filter):
                skipped.append(f"{sym} (DPS {tier}, filter={dps_income_filter})")
                return False, row
        if exclude_eps_decliners:
            eps_vals = _company_annual_eps_values(sym)
            _note, exclude = _describe_eps_trend(eps_vals)
            if exclude:
                skipped.append(f"{sym} (EPS trend — optional strict filter)")
                return False, row
        return True, row

    def _try_add(sym: str, idx_w: float) -> bool:
        if len(picked) >= target_count:
            return False
        if any(p[0] == sym for p in picked):
            return False
        ok, row = _passes_fundamentals(sym)
        if not ok or row is None:
            return False
        sec = _resolve_sector_name(sym, str(row.get("sector_name", "UNKNOWN")))
        if sector_counts.get(sec, 0) >= max_per_sector:
            skipped.append(f"{sym} (sector cap {sec})")
            return False
        sector_counts[sec] = sector_counts.get(sec, 0) + 1
        picked.append((sym, idx_w, row))
        return True

    for sym, wt in ranked:
        if len(picked) >= target_count:
            break
        _try_add(sym, wt)

    if expand_universe and len(picked) < target_count:
        backfill: list[tuple[str, float]] = []
        for _, row in scr.iterrows():
            sym = str(row["symbol"]).upper()
            if sym in candidates and candidates.get(sym, 0) > 0:
                continue
            if shariah_only and kmi is not None and sym not in kmi:
                continue
            dy = float(pd.to_numeric(row.get("dividend_yield"), errors="coerce") or 0.0)
            if min_div_yield > 0 and dy < min_div_yield:
                continue
            backfill.append((sym, dy))
        def _backfill_sort_key(item: tuple[str, float]) -> tuple[int, float, float]:
            sym, dy = item
            row = scr_by.get(sym)
            li = str(row.get("listed_in", "")) if row is not None else ""
            bc = _blue_chip_rank(sym)
            is_blue = bc > 0 or _listed_in_is_blue_chip(li)
            tier = 0 if (not prefer_blue_chip or is_blue) else 1
            return (tier, -bc, -dy)

        backfill.sort(key=_backfill_sort_key)
        added = 0
        added_non_blue = 0

        def _is_blue_chip(sym: str) -> bool:
            row = scr_by.get(sym)
            li = str(row.get("listed_in", "")) if row is not None else ""
            return _blue_chip_rank(sym) > 0 or _listed_in_is_blue_chip(li)

        def _backfill_one(sym: str) -> None:
            nonlocal added, added_non_blue
            is_blue = _is_blue_chip(sym)
            if _try_add(sym, float(div_ref.get(sym, 0.0))):
                added += 1
                if not is_blue:
                    added_non_blue += 1

        if prefer_blue_chip:
            for sym, _dy in backfill:
                if len(picked) >= target_count:
                    break
                if not _is_blue_chip(sym):
                    continue
                _backfill_one(sym)
        for sym, _dy in backfill:
            if len(picked) >= target_count:
                break
            if prefer_blue_chip and _is_blue_chip(sym):
                continue
            _backfill_one(sym)
        if added:
            msg = (
                f"  Added {added} name(s) outside PSXDIV20 to reach your target "
                f"(dividend backfill; KSE100/KSE30 preferred when enabled)."
            )
            if prefer_blue_chip and added_non_blue:
                msg += f" {added_non_blue} not in KSE100/KSE30."
            print(msg)

    if not picked:
        print("  No names passed your filters.")
        if skipped:
            print("  Skipped examples: " + ", ".join(skipped[:10]))
        sys.exit(1)

    if len(picked) < target_count:
        print(
            f"  Note: only {len(picked)} name(s) passed filters "
            f"(asked for {target_count}). "
            "Try lower min yield, universe option 3, or turn off EPS/DPS skips."
        )

    rows: list[dict] = []
    for sym, wt, row in picked:
        dy = float(row.get("dividend_yield", 0.0) or 0.0)
        pe = pd.to_numeric(row.get("pe_ratio"), errors="coerce")
        ref_w = float(div_ref.get(sym, wt))
        if weight_mode == "index":
            alloc_w = ref_w if ref_w > 0 else max(dy, 0.01)
        else:
            alloc_w = max(dy, 0.01)
        rows.append(
            {
                "symbol": sym,
                "name": _resolve_company_name(sym),
                "sector_name": _resolve_sector_name(
                    sym, str(row.get("sector_name", "UNKNOWN"))
                ),
                "price": float(row.get("price") or 0.0),
                "idx_weight": alloc_w,
                "dividend_yield": dy,
                "pe_ratio": float(pe) if pd.notna(pe) else None,
                "in_psxdiv20": sym in div_ref,
            }
        )
    df = pd.DataFrame(rows)
    total = df["idx_weight"].sum()
    if total <= 0:
        df["idx_weight"] = 100.0 / len(df)
    else:
        df["idx_weight"] = df["idx_weight"] / total * 100.0
    return df.sort_values("idx_weight", ascending=False).reset_index(drop=True)


def _dividend_basket_div_ref(
    *, shariah_only: bool, shariah_pool: str, quiet: bool = False
) -> dict[str, float]:
    div_w = _psxdiv20_weight_map()
    if shariah_only:
        filtered = _apply_shariah_filter_weights(
            div_w,
            enabled=True,
            context="PSXDIV20",
            shariah_pool=shariah_pool,
            quiet=quiet,
        )
        return filtered or {}
    return div_w


def _renormalize_dividend_basket_df(
    df: pd.DataFrame,
    *,
    weight_mode: str,
    div_ref: dict[str, float],
) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if weight_mode == "yield":
        out["idx_weight"] = pd.to_numeric(
            out["dividend_yield"], errors="coerce"
        ).fillna(0.01).clip(lower=0.01)
    else:
        weights: list[float] = []
        for sym in out["symbol"].astype(str).str.upper():
            ref = float(div_ref.get(sym, 0.0) or 0.0)
            if ref <= 0:
                row = _screener_row(sym)
                ref = 0.01
                if row is not None:
                    ref = max(
                        float(
                            pd.to_numeric(row.get("dividend_yield"), errors="coerce")
                            or 0.01
                        ),
                        0.01,
                    )
            weights.append(ref)
        out["idx_weight"] = weights
    total = float(out["idx_weight"].sum())
    if total <= 0:
        out["idx_weight"] = 100.0 / len(out)
    else:
        out["idx_weight"] = out["idx_weight"] / total * 100.0
    return out.sort_values("idx_weight", ascending=False).reset_index(drop=True)


def _dividend_screener_row_to_basket_row(
    sym: str,
    *,
    div_ref: dict[str, float],
    weight_mode: str,
) -> dict | None:
    sym = _tradable_equity_symbol(sym.upper())
    row = _screener_row(sym)
    if row is None:
        return None
    price = float(pd.to_numeric(row.get("price"), errors="coerce") or 0.0)
    if price <= 0:
        return None
    dy = float(pd.to_numeric(row.get("dividend_yield"), errors="coerce") or 0.0)
    pe = pd.to_numeric(row.get("pe_ratio"), errors="coerce")
    ref_w = float(div_ref.get(sym, 0.0) or 0.0)
    if weight_mode == "index":
        alloc_w = ref_w if ref_w > 0 else max(dy, 0.01)
    else:
        alloc_w = max(dy, 0.01)
    return {
        "symbol": sym,
        "name": _resolve_company_name(sym),
        "sector_name": _resolve_sector_name(sym, str(row.get("sector_name", "UNKNOWN"))),
        "price": price,
        "idx_weight": alloc_w,
        "dividend_yield": dy,
        "pe_ratio": float(pe) if pd.notna(pe) else None,
        "in_psxdiv20": sym in div_ref,
    }


def _dividend_replacement_candidates(
    held_symbols: set[str],
    *,
    shariah_only: bool,
    shariah_pool: str,
    min_div_yield: float,
    max_pe: float | None,
    exclude_eps: bool,
    dps_income_filter: str,
    prefer_blue_chip: bool,
    limit: int = 12,
    div_ref: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Fast shortlist: screener pre-filter, then DPS/EPS only on top ~40 by yield."""
    if div_ref is None:
        div_ref = _dividend_basket_div_ref(
            shariah_only=shariah_only, shariah_pool=shariah_pool, quiet=True
        )
    scr = _build_screener_enriched()
    kmi = _shariah_symbol_set(shariah_pool) if shariah_only else None
    blue_chip_w = _blue_chip_index_weight_map()
    prelim: list[dict] = []
    for _, row in scr.iterrows():
        sym = str(row["symbol"]).upper()
        if sym in held_symbols:
            continue
        if shariah_only and kmi is not None and sym not in kmi:
            continue
        dy = float(pd.to_numeric(row.get("dividend_yield"), errors="coerce") or 0.0)
        if min_div_yield > 0 and dy < min_div_yield:
            continue
        pe = pd.to_numeric(row.get("pe_ratio"), errors="coerce")
        if max_pe is not None and pd.notna(pe) and float(pe) > max_pe:
            continue
        li = str(row.get("listed_in", ""))
        bc = float(blue_chip_w.get(sym, 0.0))
        if _listed_in_is_blue_chip(li):
            bc = max(bc, 0.01)
        prelim.append(
            {
                "symbol": sym,
                "name": _resolve_company_name(sym),
                "dividend_yield": dy,
                "pe_ratio": float(pe) if pd.notna(pe) else None,
                "blue_chip_rank": bc,
                "in_psxdiv20": sym in div_ref,
            }
        )
    if not prelim:
        return pd.DataFrame()
    pre_df = pd.DataFrame(prelim)
    if prefer_blue_chip:
        pre_df = pre_df.sort_values(
            ["blue_chip_rank", "dividend_yield"], ascending=[False, False]
        )
    else:
        pre_df = pre_df.sort_values("dividend_yield", ascending=False)
    shortlist = pre_df.head(40)

    picked: list[dict] = []
    need_eps = exclude_eps
    need_dps = dps_income_filter != "off"
    n = len(shortlist)
    if need_eps or need_dps:
        print(
            f"  Checking DPS/EPS on top {n} candidates (not whole market)…",
            flush=True,
        )
    for i, (_, row) in enumerate(shortlist.iterrows(), start=1):
        sym = str(row["symbol"]).upper()
        if need_eps or need_dps:
            if i == 1 or i == n or i % 5 == 0:
                print(f"    … {i}/{n}", flush=True)
        if exclude_eps:
            _note, exclude = _describe_eps_trend(_company_annual_eps_values(sym))
            if exclude:
                continue
        if dps_income_filter != "off":
            tier = _dps_income_tier(_dps_history(sym))
            if _dps_tier_excluded(tier, dps_income_filter):
                continue
        picked.append(row.to_dict())
        if len(picked) >= limit:
            break
    if not picked:
        return pd.DataFrame()
    return pd.DataFrame(picked).reset_index(drop=True)


def _parse_basket_drop_symbols(raw: str, basket_symbols: list[str]) -> list[str]:
    held = set(basket_symbols)
    out: list[str] = []
    for part in raw.replace(";", ",").split(","):
        token = part.strip().upper()
        if not token:
            continue
        if token.isdigit():
            idx = int(token) - 1
            if 0 <= idx < len(basket_symbols):
                out.append(basket_symbols[idx])
            continue
        sym = _tradable_equity_symbol(token)
        if sym in held:
            out.append(sym)
    # preserve order, unique
    seen: set[str] = set()
    unique: list[str] = []
    for s in out:
        if s not in seen:
            seen.add(s)
            unique.append(s)
    return unique


def _parse_replacement_picks(
    raw: str, cands: pd.DataFrame, count: int
) -> list[str]:
    """Resolve comma-sep symbols or 1-based indices into replacement tickers."""
    text = raw.strip()
    if not text:
        return []
    parts = [p.strip() for p in text.replace(";", ",").split(",") if p.strip()]
    out: list[str] = []
    cand_syms = [
        str(cands.iloc[j]["symbol"]).upper() for j in range(len(cands))
    ]
    for part in parts:
        if part.isdigit():
            j = int(part) - 1
            if 0 <= j < len(cand_syms):
                out.append(cand_syms[j])
            continue
        out.append(_tradable_equity_symbol(part.upper()))
    return out[:count]


def _print_dividend_target_basket_table(df: pd.DataFrame, *, title: str) -> None:
    print(f"\n--- Target basket: {title} ({len(df)} stocks) ---")
    if df.empty:
        print("  (empty)")
        return
    for line_no, (_, row) in enumerate(df.iterrows(), start=1):
        print(
            f"  {line_no:>2}. {str(row['symbol']):<8} "
            f"{str(row.get('name', ''))[:34]:<34} "
            f"{float(row.get('idx_weight', 0)):.2f}%"
        )
    show_cols = ["symbol", "name", "sector_name", "idx_weight", "dividend_yield"]
    if "pe_ratio" in df.columns:
        show_cols.append("pe_ratio")
    if "in_psxdiv20" in df.columns:
        show_cols.append("in_psxdiv20")
    show = df[show_cols].copy()
    show["idx_weight"] = show["idx_weight"].map(lambda w: f"{w:.2f}%")
    if "dividend_yield" in show.columns:
        show["dividend_yield"] = show["dividend_yield"].map(
            lambda v: f"{float(v):.2f}%"
        )
    if "pe_ratio" in show.columns:
        show["pe_ratio"] = show["pe_ratio"].map(
            lambda v: f"{v:.1f}" if v is not None and pd.notna(v) else "—"
        )
    if "in_psxdiv20" in show.columns:
        show["in_psxdiv20"] = show["in_psxdiv20"].map(
            lambda v: "index" if v else "backfill"
        )
    show = show.rename(
        columns={
            "idx_weight": "target_weight",
            "pe_ratio": "P/E",
            "sector_name": "sector",
            "in_psxdiv20": "psxdiv20",
        }
    )
    print()
    print(show.to_string(index=False))


def _print_dividend_screen_symbol_diagnosis(
    symbols: list[str],
    *,
    picked_symbols: set[str],
    shariah_only: bool,
    shariah_pool: str,
    min_div_yield: float,
    max_pe: float | None,
    exclude_eps: bool,
    dps_income_filter: str,
    target_count: int,
    universe_mode: str,
) -> None:
    """Explain which filter removed each symbol (same rules as screened basket build)."""
    if not symbols:
        return
    div_w = _psxdiv20_weight_map()
    div_ref = _dividend_basket_div_ref(
        shariah_only=shariah_only, shariah_pool=shariah_pool, quiet=True
    )
    kmi = _shariah_symbol_set(shariah_pool) if shariah_only else None
    ranked = sorted(div_ref.items(), key=lambda kv: kv[1], reverse=True)
    rank_map = {s: i + 1 for i, (s, _) in enumerate(ranked)}

    print("\n  --- Why name(s) missed your basket ---")
    for raw in symbols:
        sym = _tradable_equity_symbol(raw.upper())
        name = _resolve_company_name(sym)
        print(f"\n  {raw} → {sym}  ({name})")
        if sym in picked_symbols:
            print("    Already in your basket.")
            continue
        row = _screener_row(sym)
        if row is None:
            print("    Blocked: not on PSX screener (no price/yield).")
            continue
        reasons: list[str] = []
        in_raw = sym in div_w
        in_ref = sym in div_ref
        if in_ref:
            print(
                f"    PSXDIV20: yes (#{rank_map.get(sym, '?')} of {len(ranked)} "
                f"by index weight, {div_ref.get(sym, 0):.2f}%)"
            )
        elif in_raw and shariah_only:
            print("    PSXDIV20: in index but removed by Shariah filter (not KMI list).")
            reasons.append("Shariah index filter")
        else:
            print("    PSXDIV20: not an index member — needs a backfill slot if filters pass.")

        if shariah_only and kmi is not None and sym not in kmi:
            reasons.append("not on selected Shariah list (KMI30 / KMI All Share)")

        dy = float(pd.to_numeric(row.get("dividend_yield"), errors="coerce") or 0.0)
        if min_div_yield > 0 and dy < min_div_yield:
            reasons.append(f"yield {dy:.2f}% < min {min_div_yield:g}%")
        else:
            print(f"    Yield: {dy:.2f}% (passes min {min_div_yield:g}%)")

        pe = pd.to_numeric(row.get("pe_ratio"), errors="coerce")
        if max_pe is not None and pd.notna(pe) and float(pe) > max_pe:
            reasons.append(f"P/E {float(pe):.1f} > cap {max_pe:g}")

        if exclude_eps:
            _eps_note, eps_ex = _describe_eps_trend(_company_annual_eps_values(sym))
            if eps_ex:
                reasons.append(f"EPS filter ({_eps_note})")
            else:
                print(f"    EPS: {_eps_note}")

        if dps_income_filter != "off":
            tier = _dps_income_tier(_dps_history(sym))
            dps_note, _ = _describe_dps_regularity(_dps_history(sym))
            if _dps_tier_excluded(tier, dps_income_filter):
                reasons.append(f"DPS filter ({dps_note})")
            else:
                print(f"    DPS: {dps_note}")

        if universe_mode == "div20_only" and not in_ref:
            reasons.append("universe = PSXDIV20 only (not in index)")

        if not reasons and not in_ref:
            # backfill competition
            held_yield = []
            for s in picked_symbols:
                r = _screener_row(s)
                if r is not None:
                    held_yield.append(
                        (
                            s,
                            float(
                                pd.to_numeric(r.get("dividend_yield"), errors="coerce")
                                or 0
                            ),
                        )
                    )
            held_yield.sort(key=lambda t: t[1])
            print(
                "    Filters pass, but not chosen: only "
                f"{target_count} slots; PSXDIV20 names fill first, then "
                "backfill by blue-chip + yield."
            )
            if held_yield:
                lo = held_yield[0]
                print(
                    f"    Lowest-yield pick in basket: {lo[0]} at {lo[1]:.2f}% "
                    f"(you: {dy:.2f}%)."
                )
        elif reasons:
            print("    Blocked by: " + "; ".join(reasons))
        elif in_ref and rank_map.get(sym, 99) > target_count:
            print(
                f"    Index rank #{rank_map[sym]} — builder fills by index weight "
                f"order until {target_count} names pass filters (others can bump this)."
            )

    print(
        "\n  Tip: with default dividend-first settings, EPS is not filtered — "
        "FABL/MEBL drop only if you enabled strict EPS. "
        "SAZEW needs a backfill slot vs higher-yield names (e.g. MARI, OGDC)."
    )


def prompt_dividend_basket_use_or_replace(
    df: pd.DataFrame,
    *,
    title: str,
    replace_ctx: dict | None,
) -> pd.DataFrame | None:
    """
    Show basket; [Y] proceed, [r] swap symbol(s) using same filters, [n] cancel.
    replace_ctx keys: shariah_only, shariah_pool, min_div_yield, max_pe,
    exclude_eps, dps_income_filter, weight_mode, prefer_blue_chip.
    """
    basket = df.copy()
    if "dividend_yield" not in basket.columns:
        basket["dividend_yield"] = 0.0
    if "sector_name" not in basket.columns:
        basket["sector_name"] = "UNKNOWN"
    if "name" not in basket.columns:
        basket["name"] = basket["symbol"].astype(str).map(_resolve_company_name)

    div_ref: dict[str, float] = {}
    weight_mode = "yield"
    if replace_ctx:
        cached = replace_ctx.get("div_ref")
        if isinstance(cached, dict) and cached:
            div_ref = cached
        else:
            div_ref = _dividend_basket_div_ref(
                shariah_only=bool(replace_ctx.get("shariah_only")),
                shariah_pool=str(replace_ctx.get("shariah_pool", "kmi30")),
                quiet=True,
            )
        weight_mode = str(replace_ctx.get("weight_mode", "yield"))

    while True:
        _print_dividend_target_basket_table(basket, title=title)
        ans = prompt(
            "\nProceed to SIP? [Y] yes  [r] replace stock(s)  [n] cancel  "
            "[why MEBL,FABL]: "
        ).strip()
        ans_l = ans.lower()
        if ans_l.startswith("why") or ans_l.startswith("?"):
            tail = ans_l.replace("why", "", 1).replace("?", "", 1).strip()
            syms = [
                t.strip().upper()
                for t in tail.replace(";", ",").split(",")
                if t.strip()
            ]
            if syms and replace_ctx:
                _print_dividend_screen_symbol_diagnosis(
                    syms,
                    picked_symbols=set(basket["symbol"].astype(str).str.upper()),
                    shariah_only=bool(replace_ctx.get("shariah_only")),
                    shariah_pool=str(replace_ctx.get("shariah_pool", "kmi30")),
                    min_div_yield=float(replace_ctx.get("min_div_yield", 0.0)),
                    max_pe=replace_ctx.get("max_pe"),
                    exclude_eps=bool(replace_ctx.get("exclude_eps")),
                    dps_income_filter=str(
                        replace_ctx.get("dps_income_filter", "off")
                    ),
                    target_count=len(basket),
                    universe_mode=str(
                        replace_ctx.get("universe_mode", "div20_expand")
                    ),
                )
            else:
                print("  Example: why MEBL,FABL,SAZEW")
            continue
        if ans_l in ("n", "no"):
            print("  Cancelled.")
            return None
        if ans_l in ("", "y", "yes"):
            return basket
        if ans_l not in ("r", "replace", "swap", "edit"):
            print("  Enter Y, r, n, or why SYMBOL.")
            continue

        held = set(basket["symbol"].astype(str).str.upper())
        syms = basket["symbol"].astype(str).str.upper().tolist()
        print(
            "\n  Replace — line number(s) or symbol(s), comma-sep "
            "(e.g. 5,6 or ISL,CSAPXD):"
        )
        drop_raw = prompt("Drop: ").strip()
        if not drop_raw:
            continue
        drop_list = _parse_basket_drop_symbols(drop_raw, syms)
        if not drop_list:
            print("  Could not match any line or symbol in this basket.")
            continue
        print(f"  Removing: {', '.join(drop_list)}")

        held_after = held - set(drop_list)
        new_syms: list[str] = []
        cands = pd.DataFrame()
        if replace_ctx:
            cands = _dividend_replacement_candidates(
                held_after,
                shariah_only=bool(replace_ctx.get("shariah_only")),
                shariah_pool=str(replace_ctx.get("shariah_pool", "kmi30")),
                min_div_yield=float(replace_ctx.get("min_div_yield", 0.0)),
                max_pe=replace_ctx.get("max_pe"),
                exclude_eps=bool(replace_ctx.get("exclude_eps")),
                dps_income_filter=str(
                    replace_ctx.get("dps_income_filter", "off")
                ),
                prefer_blue_chip=bool(replace_ctx.get("prefer_blue_chip", True)),
                div_ref=div_ref,
            )
            if not cands.empty:
                print("\n  Suggested replacements (same filters as your basket):")
                for j, (_, row) in enumerate(cands.iterrows(), start=1):
                    pe = row.get("pe_ratio")
                    pe_s = (
                        f"{float(pe):.1f}" if pe is not None and pd.notna(pe) else "—"
                    )
                    tag = "index" if row.get("in_psxdiv20") else "screener"
                    print(
                        f"    {j}. {row['symbol']:<8} "
                        f"{str(row['name'])[:30]:<30} "
                        f"yield {float(row['dividend_yield']):.2f}%  P/E {pe_s}  ({tag})"
                    )
            else:
                print("  No screener names match your filters (try looser filters).")

        n_drop = len(drop_list)
        pick = prompt(
            f"\nAdd {n_drop} replacement(s) — comma-sep symbols or list numbers "
            f"(e.g. FABL,MEBL or 1,3): "
        ).strip()
        if pick and not cands.empty:
            new_syms = _parse_replacement_picks(pick, cands, n_drop)
        elif pick:
            new_syms = [
                _tradable_equity_symbol(p.strip().upper())
                for p in pick.replace(";", ",").split(",")
                if p.strip()
            ][:n_drop]
        if len(new_syms) < n_drop:
            manual = prompt(
                "Type symbol(s) manually, comma-sep (or blank to cancel): "
            ).strip()
            if not manual:
                continue
            extra = [
                _tradable_equity_symbol(p.strip().upper())
                for p in manual.replace(";", ",").split(",")
                if p.strip()
            ]
            new_syms = (new_syms + extra)[:n_drop]

        if len(new_syms) != n_drop:
            print(f"  Need exactly {n_drop} replacement(s); got {len(new_syms)}.")
            continue

        ok = True
        for drop_sym, new_sym in zip(drop_list, new_syms):
            if new_sym in held_after:
                print(f"  {new_sym} is already in the basket.")
                ok = False
                break
            new_row = _dividend_screener_row_to_basket_row(
                new_sym, div_ref=div_ref, weight_mode=weight_mode
            )
            if new_row is None:
                print(f"  Could not price or find {new_sym} on the PSX screener.")
                ok = False
                break
            basket = basket[
                basket["symbol"].astype(str).str.upper() != drop_sym
            ]
            basket = pd.concat(
                [basket, pd.DataFrame([new_row])], ignore_index=True
            )
            held_after = held_after - {drop_sym} | {new_sym}
            print(f"  Swapped {drop_sym} → {new_sym} ({new_row['name']}).")
            dps_note, _ = _describe_dps_regularity(_dps_history(new_sym))
            print(f"    DPS: {dps_note}")
        if not ok:
            continue
        basket = _renormalize_dividend_basket_df(
            basket, weight_mode=weight_mode, div_ref=div_ref
        )


def run_dividend_psxdiv20_screened_sip_flow() -> None:
    """Top-N dividend basket with yield / P/E / optional DPS — not mutual-fund snapshot."""
    print(
        "\n  Dividend screened basket (PSX screener + optional PSXDIV20 weights — not advice)."
    )
    shariah_only = prompt_yes_no("Shariah-only (Islamic-index listed)", default_yes=True)
    shariah_pool = "kmi30"
    if shariah_only:
        print(
            "\n  Shariah list: 1=KMI30 (large cap)  "
            "2=KMI All Share (wider — FATIMA, MTLXD, etc.)"
        )
        sp = prompt("Shariah list [2]: ").strip().lower() or "2"
        shariah_pool = "kmi_all" if sp in ("2", "b", "all", "wide", "kmi_all") else "kmi30"

    print("\n  Universe:")
    print("    1. PSXDIV20 first, then KMI30 dividend backfill (recommended)")
    print("    2. PSXDIV20 index members only")
    print("    3. KMI30 dividend screener (all sectors — FATIMA, MTL, etc.)")
    print("    4. My holdings only — screen what I own (not limited to PSXDIV20)")
    uni_choice = prompt("Universe [1]: ").strip().lower() or "1"
    universe_map = {
        "1": "div20_expand",
        "expand": "div20_expand",
        "a": "div20_expand",
        "2": "div20_only",
        "div20": "div20_only",
        "index": "div20_only",
        "3": "kmi30_screen",
        "kmi": "kmi30_screen",
        "screener": "kmi30_screen",
        "b": "kmi30_screen",
        "4": "holdings",
        "holdings": "holdings",
        "mine": "holdings",
        "csv": "holdings",
    }
    universe_mode = universe_map.get(uni_choice, "div20_expand")
    pool_symbols: set[str] | None = None
    if universe_mode == "holdings":
        pool_symbols = _load_dividend_pool_from_holdings()
        if not pool_symbols:
            print("  Falling back to universe 1 (PSXDIV20 + backfill).")
            universe_mode = "div20_expand"
        else:
            print(
                f"  Holdings pool: {len(pool_symbols)} symbol(s) "
                "(PSXDIV20 weight when indexed; else yield-ranked)."
            )

    target_count = int(prompt_positive_float("\nHow many stocks (typical 8)", 8.0))
    target_count = max(3, min(20, target_count))
    min_div = prompt_non_negative_float("Minimum dividend yield %", 2.0)
    max_pe_raw = prompt("Maximum P/E (blank = no cap): ").strip()
    max_pe = float(max_pe_raw) if max_pe_raw else None
    exclude_eps = prompt_dividend_eps_filter()
    dps_income_filter = prompt_dps_income_filter()
    wt_choice = prompt(
        "Weights [1] PSXDIV20 index  [2] dividend yield [2]: "
    ).strip().lower() or "2"
    weight_mode = "yield" if wt_choice in ("2", "yield", "y") else "index"
    prefer_blue_chip = prompt_yes_no(
        "Prefer KSE100/KSE30 blue chips when filling the basket",
        default_yes=True,
    )

    print("\nBuilding screened dividend basket…")
    df = _build_psxdiv20_screened_basket(
        target_count=target_count,
        shariah_only=shariah_only,
        shariah_pool=shariah_pool,
        pool_symbols=pool_symbols,
        universe_mode=universe_mode,
        min_div_yield=min_div,
        max_pe=max_pe,
        exclude_eps_decliners=exclude_eps,
        dps_income_filter=dps_income_filter,
        weight_mode=weight_mode,
        prefer_blue_chip=prefer_blue_chip,
    )

    sym_list = df["symbol"].astype(str).tolist()
    _print_dps_consistency_table(sym_list)
    print(
        "\n  Note: PSX index/screener often uses *XD tickers (e.g. HUBCXD); "
        "company names and DPS history are resolved from the main line (e.g. HUBC) "
        "when PSX has no separate profile for the XD symbol."
    )
    print(
        "  PSXDIV20 members are index rules (yield + history); backfill names are "
        "high yield from your Shariah screener — not all are in the index."
    )

    title = "PSXDIV20 screened"
    if shariah_only:
        title += (
            " · KMI All Share"
            if shariah_pool != "kmi30"
            else " · KMI30"
        )
    if pool_symbols:
        title += " · holdings pool"

    replace_ctx = {
        "shariah_only": shariah_only,
        "shariah_pool": shariah_pool,
        "min_div_yield": min_div,
        "max_pe": max_pe,
        "exclude_eps": exclude_eps,
        "dps_income_filter": dps_income_filter,
        "weight_mode": weight_mode,
        "prefer_blue_chip": prefer_blue_chip,
        "universe_mode": universe_mode,
        "div_ref": _dividend_basket_div_ref(
            shariah_only=shariah_only, shariah_pool=shariah_pool, quiet=True
        ),
    }
    df = prompt_dividend_basket_use_or_replace(
        df, title=title, replace_ctx=replace_ctx
    )
    if df is None:
        return

    basket_id = "div_psxdiv20_screened"
    meta = {
        "index_name": basket_id,
        "as_of": _utc_now_iso(),
        "title": title,
        "parsed_holdings": [],
        "missing_price_symbols": [],
        "sip_profile": "dividend_basket",
    }
    _run_benchmark_sip_flow(
        df,
        meta,
        source_type="custom",
        source_id=basket_id,
        universe_label="dividend basket",
        weight_label="target",
        empty_exclusion_msg="All names were excluded. Nothing to allocate.",
        exclusion_unit="names",
    )


def choose_dividend_portfolio_style() -> str:
    print(
        "\n  1. Screened basket SIP (PSXDIV20 + backfill — DPS & yield) [recommended]"
    )
    print("  2. PSXDIV20 index SIP (all constituents — scope like KSE100)")
    print("  3. High-yield pick (count + pool: PSXDIV20 or screener)")
    print("  4. Mutual fund + PSXDIV20 blend (FMR snapshot — may drift)")
    print("  5. Fund/index holdings only — no SIP")
    print("  h. Dividend income tips")
    while True:
        choice = prompt("Select [1]: ") or "1"
        if _is_help_choice(choice):
            print_dividend_income_strategy_tips()
            continue
        if choice in ("1", "screen", "screened", "top", "psxdiv20"):
            return "screened"
        if choice in ("2", "index", "div20"):
            return "index"
        if choice in ("3", "sip", "yield", "scan", "stocks"):
            return "yield_sip"
        if choice in ("4", "mf", "blend", "both", "fund"):
            return "mf_div_sip"
        if choice in ("5", "ref", "reference", "holdings", "see", "view"):
            return "reference"
        print("  Enter 1–5, or h.")


def _choose_high_yield_sip_pool() -> tuple[set[str], str, str]:
    print("\n  Stock pool for high-yield SIP:")
    print("  a. PSXDIV20 members only (recommended)")
    print("  b. Whole PSX screener")
    sub = prompt("Pool [a]: ").lower() or "a"
    if sub in ("b", "all", "market", "scan"):
        scr = _build_screener_enriched()
        return (
            set(scr["symbol"].astype(str)),
            "Screener yield basket",
            "dividend_screener",
        )
    return (
        set(_index_constituent_weights("PSXDIV20")["symbol"].astype(str)),
        "PSXDIV20 yield basket",
        "dividend_psxdiv20",
    )


def run_dividend_portfolio_flow() -> None:
    print("\nDividend menu — compare fund holdings or plan a stock SIP (not advice).")
    style = choose_dividend_portfolio_style()

    if style == "screened":
        run_dividend_psxdiv20_screened_sip_flow()
        return

    if style == "mf_div_sip":
        run_dividend_mf_psxdiv20_sip_flow()
        return

    if style == "reference":
        run_dividend_reference_composition_flow()
        return

    if style == "index":
        df, meta = load_index_frame("PSXDIV20")
        print_overview_tables(df, "PSXDIV20")
        _run_benchmark_sip_flow(
            df,
            meta,
            source_type="index",
            source_id="PSXDIV20",
            universe_label="PSXDIV20",
            weight_label="index",
            empty_exclusion_msg="All constituents were excluded. Nothing to allocate.",
            exclusion_unit="constituents",
        )
        return

    pool, label, basket_id = _choose_high_yield_sip_pool()
    target_count = int(
        prompt_positive_float("\nHow many stocks (typical 8)", 8.0)
    )
    target_count = max(3, min(40, target_count))
    min_div = prompt_non_negative_float("Minimum dividend yield %", 4.0)
    eps_filter = prompt("Skip stocks with EPS down 2+ years? [Y/n]: ").lower()
    exclude_eps = eps_filter not in ("n", "no")

    print(f"\nBuilding {label}…")
    df = build_yield_weighted_dividend_basket(
        pool,
        target_count=target_count,
        min_div_yield=min_div,
        max_per_sector=CUSTOM_BASKET_MAX_PER_SECTOR,
        exclude_eps_decliners=exclude_eps,
    )

    print(f"\n{'=' * 72}")
    print(f"  {label} ({len(df)} names, weights ∝ dividend yield)")
    print(f"{'=' * 72}\n")
    show = df[["symbol", "name", "sector_name", "price", "idx_weight", "dividend_yield"]].copy()
    show["price"] = show["price"].map(lambda p: f"{p:,.2f}")
    show["idx_weight"] = show["idx_weight"].map(lambda w: f"{w:.2f}%")
    show["dividend_yield"] = show["dividend_yield"].map(lambda v: f"{v:.2f}%")
    show = show.rename(columns={"idx_weight": "target_weight"})
    print(show.to_string(index=False))

    df = prompt_dividend_basket_use_or_replace(
        df,
        title=label,
        replace_ctx={
            "shariah_only": False,
            "shariah_pool": "kmi30",
            "min_div_yield": min_div,
            "max_pe": None,
            "exclude_eps": exclude_eps,
            "dps_income_filter": "off",
            "weight_mode": "yield",
            "prefer_blue_chip": True,
        },
    )
    if df is None:
        return

    meta = {
        "index_name": basket_id,
        "as_of": _utc_now_iso(),
        "title": label,
        "parsed_holdings": [],
        "missing_price_symbols": [],
        "sip_profile": "dividend_basket",
    }
    _run_benchmark_sip_flow(
        df,
        meta,
        source_type="custom",
        source_id=basket_id,
        universe_label="dividend basket",
        weight_label="target",
        empty_exclusion_msg="All names were excluded. Nothing to allocate.",
        exclusion_unit="names",
    )


def _filter_screener_scanner(scr: pd.DataFrame) -> pd.DataFrame:
    print(
        "\n  Filters use the live PSX screener. "
        "Leave blank at any prompt for no limit on that field.\n"
        "  Note: PSX company pages show annual EPS and profit after tax (000's); "
        "operating cash flow is not published on that page.\n"
    )
    min_pe = prompt_optional_float("Minimum P/E ratio")
    max_pe = prompt_optional_float("Maximum P/E ratio")
    min_div = prompt_optional_float("Minimum dividend yield %")
    max_div = prompt_optional_float("Maximum dividend yield %")
    min_cap = prompt_optional_float("Minimum market cap (PKR, as on screener)")
    min_1y = prompt_optional_float("Minimum 1-year price change %")
    sector_raw = prompt("Sector name contains (optional): ").strip()
    out = scr.copy()
    out["pe_ratio"] = pd.to_numeric(out["pe_ratio"], errors="coerce")

    if min_pe is not None:
        out = out[out["pe_ratio"].fillna(-1) >= min_pe]
    if max_pe is not None:
        out = out[(out["pe_ratio"].notna()) & (out["pe_ratio"] <= max_pe)]
    if min_div is not None:
        out = out[out["dividend_yield"] >= min_div]
    if max_div is not None:
        out = out[out["dividend_yield"] <= max_div]
    if min_cap is not None:
        out = out[out["market_cap"] >= min_cap]
    if min_1y is not None:
        out = out[out["change_1y_pct"] >= min_1y]
    if sector_raw:
        needle = sector_raw.lower()
        out = out[
            out["sector_name"].astype(str).str.lower().str.contains(needle, na=False)
        ]
    return out.reset_index(drop=True)


def _enrich_scanner_with_company_pages(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    cap = 80
    if len(df) > cap:
        print(
            f"  Enriching first {cap} rows (of {len(df)}) with PSX company pages…"
        )
        work = df.head(cap).copy()
    else:
        work = df.copy()
        print(f"  Enriching {len(work)} row(s) with PSX company pages…")

    dps_ans = prompt(
        "  Include DPS / 5y dividend payment history (Yahoo .KA)? [Y/n]: "
    ).lower()
    with_dps = dps_ans not in ("n", "no") and _yahoo_dividends_enabled()
    if dps_ans not in ("n", "no") and not _yahoo_dividends_enabled():
        print("  PSX_SIP_NO_YAHOO=1 — skipping DPS history.")

    eps_latest: list[float | None] = []
    eps_notes: list[str] = []
    profit_yoy: list[float | None] = []
    dps_notes: list[str] = []
    fy_dps: list[float | None] = []
    for i, sym in enumerate(work["symbol"].astype(str).str.upper(), start=1):
        if i == 1 or i == len(work) or i % 10 == 0:
            print(f"    … {i}/{len(work)}", flush=True)
        fin = _company_financials(sym)
        eps = fin.get("eps") or []
        pat = fin.get("profit_after_tax") or []
        eps_latest.append(eps[0] if eps else None)
        note, _ex = _describe_eps_trend(eps)
        eps_notes.append(note)
        profit_yoy.append(_yoy_pct(pat))
        if with_dps:
            hist = _dps_history(sym)
            dnote, _irr = _describe_dps_regularity(hist)
            dps_notes.append(dnote)
            fy_dps.append(hist.get("last_fiscal_total"))
        else:
            dps_notes.append("")
            fy_dps.append(None)

    work["eps_latest"] = eps_latest
    work["eps_trend"] = eps_notes
    work["profit_yoy_pct"] = profit_yoy
    if with_dps:
        work["dps_status"] = dps_notes
        work["fy_dps_total"] = fy_dps
    return work


def run_stock_scanner_flow() -> None:
    print(
        "\nStock scanner — filter PSX screener, optionally pull EPS / DPS / profit "
        "from PSX + Yahoo.\n"
        "  Research tool only; not investment advice."
    )
    print("\nFetching screener…")
    scr = _build_screener_enriched()
    filtered = _filter_screener_scanner(scr)
    print(f"\n  {len(filtered)} symbol(s) after screener filters (from {len(scr)}).")
    if filtered.empty:
        print("  No matches. Widen filters and try again.")
        return

    sort_choice = prompt(
        "Sort by [1] dividend yield  [2] P/E (low first)  [3] 1y change  [4] market cap [1]: "
    ) or "1"
    sort_map = {
        "1": ("dividend_yield", False),
        "2": ("pe_ratio", True),
        "3": ("change_1y_pct", False),
        "4": ("market_cap", False),
        "yield": ("dividend_yield", False),
        "pe": ("pe_ratio", True),
        "1y": ("change_1y_pct", False),
        "cap": ("market_cap", False),
    }
    col, asc = sort_map.get(sort_choice.strip().lower(), ("dividend_yield", False))
    filtered = filtered.sort_values(col, ascending=asc, na_position="last")

    enrich = prompt(
        "Fetch EPS & profit-after-tax from PSX company pages for matches? [y/N]: "
    ).lower()
    if enrich in ("y", "yes"):
        min_eps = prompt_optional_float("Minimum latest annual EPS (PKR)")
        exclude_eps = prompt(
            "Drop names with EPS down 2+ years in a row? [Y/n]: "
        ).lower()
        drop_eps_slide = exclude_eps not in ("n", "no")
        min_pat_yoy = prompt_optional_float(
            "Minimum profit-after-tax YoY % (latest vs prior year)"
        )
        filtered = _enrich_scanner_with_company_pages(filtered)
        if min_eps is not None:
            filtered = filtered[
                filtered["eps_latest"].fillna(-1e9) >= min_eps
            ]
        if drop_eps_slide:
            filtered = filtered[
                ~filtered["eps_trend"].astype(str).str.contains("↓", regex=False)
            ]
        if min_pat_yoy is not None:
            filtered = filtered[
                filtered["profit_yoy_pct"].fillna(-1e9) >= min_pat_yoy
            ]
        print(f"  {len(filtered)} symbol(s) after company-page filters.")

    max_show = int(prompt_positive_float("How many rows to display", 40.0))
    max_show = max(5, min(200, max_show))
    view = filtered.head(max_show)

    cols = [
        "symbol",
        "name",
        "sector_name",
        "price",
        "pe_ratio",
        "dividend_yield",
        "change_1y_pct",
        "market_cap",
    ]
    if "eps_latest" in view.columns:
        cols.extend(["eps_latest", "eps_trend", "profit_yoy_pct"])
    if "fy_dps_total" in view.columns:
        cols.extend(["fy_dps_total", "dps_status"])
    show = view[[c for c in cols if c in view.columns]].copy()
    if "price" in show.columns:
        show["price"] = show["price"].map(lambda p: f"{p:,.2f}" if pd.notna(p) else "")
    if "pe_ratio" in show.columns:
        show["pe_ratio"] = show["pe_ratio"].map(
            lambda v: f"{v:.2f}" if pd.notna(v) else "—"
        )
    if "dividend_yield" in show.columns:
        show["dividend_yield"] = show["dividend_yield"].map(
            lambda v: f"{v:.2f}%" if pd.notna(v) else ""
        )
    if "change_1y_pct" in show.columns:
        show["change_1y_pct"] = show["change_1y_pct"].map(
            lambda v: f"{v:+.1f}%" if pd.notna(v) else ""
        )
    if "profit_yoy_pct" in show.columns:
        show["profit_yoy_pct"] = show["profit_yoy_pct"].map(
            lambda v: f"{v:+.1f}%" if pd.notna(v) else "—"
        )

    print(f"\n{'=' * 72}")
    print(f"  Scanner results (showing {len(show)} of {len(filtered)})")
    print(f"{'=' * 72}\n")
    print(show.to_string(index=False))

    export = prompt("\nSave full filtered list to CSV? [y/N]: ").lower()
    if export in ("y", "yes"):
        default_name = f"psx_scan_{_export_date_tag()}.csv"
        path = prompt(f"File path [{default_name}]: ").strip() or default_name
        path = os.path.expanduser(path)
        filtered.to_csv(path, index=False)
        print(f"  Wrote {len(filtered)} row(s) to {path}")

    build = prompt(
        "\nBuild a dividend-weighted SIP basket from these results? [y/N]: "
    ).lower()
    if build not in ("y", "yes"):
        return
    take = int(
        prompt_positive_float(
            f"How many top names to include (max {len(filtered)})",
            float(min(12, len(filtered))),
        )
    )
    take = max(3, min(len(filtered), take))
    basket = filtered.head(take).copy()
    basket["idx_weight"] = basket["dividend_yield"].clip(lower=0.01)
    df = _dataframe_to_sip_basket(basket, weight_col="idx_weight", basket_id="", title="")
    basket_id = prompt("Basket label [scanner_dividend]: ").strip() or "scanner_dividend"
    basket_id = re.sub(r"[^\w\-]+", "_", basket_id)[:40]
    meta = {
        "index_name": basket_id,
        "as_of": _utc_now_iso(),
        "title": f"Scanner basket {basket_id}",
        "parsed_holdings": [],
        "missing_price_symbols": [],
        "sip_profile": "dividend_basket",
    }
    _run_benchmark_sip_flow(
        df,
        meta,
        source_type="custom",
        source_id=basket_id,
        universe_label="scanner basket",
        weight_label="target",
        empty_exclusion_msg="All names were excluded. Nothing to allocate.",
        exclusion_unit="names",
    )


def run_custom_basket_flow() -> None:
    print(
        "\nCustom basket SIP — blend blue-chip index/ETF names with "
        "dividend yield and 1-year momentum (PSX screener).\n"
        "  Mid-term / swing-friendly: you choose the mix, not a single index or ETF.\n"
        "  Not investment advice — review names before you buy."
    )
    mode = choose_custom_basket_mode()
    dividend_profile = mode == "dividend"
    if dividend_profile:
        print_dividend_income_strategy_tips()
        print()

    index_names, etf_symbols = choose_custom_basket_pool(
        dividend_profile=dividend_profile
    )
    if dividend_profile:
        w_blue, w_div, w_growth = 0.20, 0.50, 0.30
        print(
            "\n  Using income tilt blend: 20% blue chip / 50% dividend / 30% 1y growth."
        )
        target_count = 8
        min_div = 4.0
        print(
            f"  Defaults: {target_count} stocks, minimum dividend yield {min_div:g}%."
        )
        override = prompt("Change defaults? [y/N]: ").lower()
        if override in ("y", "yes"):
            w_blue, w_div, w_growth = choose_custom_basket_blend()
            target_count = int(
                prompt_positive_float(
                    f"\nHow many stocks in the basket (typical 8)",
                    8.0,
                )
            )
            min_div = prompt_non_negative_float(
                "Minimum dividend yield % (0 = no filter)", min_div
            )
    else:
        w_blue, w_div, w_growth = choose_custom_basket_blend()
        target_count = int(
            prompt_positive_float(
                f"\nHow many stocks in the basket (typical {CUSTOM_BASKET_DEFAULT_COUNT})",
                float(CUSTOM_BASKET_DEFAULT_COUNT),
            )
        )
        min_div = prompt_non_negative_float(
            "Minimum dividend yield % (0 = no filter)", 0.0
        )
    target_count = max(3, min(40, target_count))
    eps_filter = prompt(
        "Drop names with annual EPS down 2+ years in a row (PSX data)? [Y/n]: "
    ).lower()
    exclude_eps = eps_filter not in ("n", "no")
    basket_id = prompt("Basket label for files [smart_basket]: ").strip() or "smart_basket"
    basket_id = re.sub(r"[^\w\-]+", "_", basket_id)[:40]

    df = build_smart_custom_basket(
        index_names,
        etf_symbols,
        w_blue=w_blue,
        w_div=w_div,
        w_growth=w_growth,
        target_count=target_count,
        min_div_yield=min_div,
        max_per_sector=CUSTOM_BASKET_MAX_PER_SECTOR,
        exclude_eps_decliners=exclude_eps,
    )

    print(f"\n{'=' * 72}")
    print(f"  Proposed custom basket ({len(df)} names)")
    print(
        f"  Blend: {w_blue*100:.0f}% blue chip / {w_div*100:.0f}% dividend / "
        f"{w_growth*100:.0f}% 1y"
    )
    print(f"{'=' * 72}\n")
    print(
        "  basket_role: why the name scored well (blue-chip / dividend / growth). "
        "eps_trend: annual EPS path from PSX company page."
    )
    show = df[
        [
            "symbol",
            "name",
            "basket_role",
            "eps_trend",
            "sector_name",
            "price",
            "idx_weight",
            "dividend_yield",
            "change_1y_pct",
        ]
    ].copy()
    show["price"] = show["price"].map(lambda p: f"{p:,.2f}")
    show["idx_weight"] = show["idx_weight"].map(lambda w: f"{w:.2f}%")
    show["dividend_yield"] = show["dividend_yield"].map(lambda v: f"{v:.2f}%")
    show["change_1y_pct"] = show["change_1y_pct"].map(lambda v: f"{v:+.1f}%")
    show = show.rename(
        columns={
            "idx_weight": "target_weight",
            "change_1y_pct": "1y_price",
            "basket_role": "role",
        }
    )
    print(show.to_string(index=False))

    df = prompt_dividend_basket_use_or_replace(
        df,
        title=f"Custom basket {basket_id}",
        replace_ctx={
            "shariah_only": False,
            "shariah_pool": "kmi30",
            "min_div_yield": min_div,
            "max_pe": None,
            "exclude_eps": exclude_eps,
            "dps_income_filter": "off",
            "weight_mode": "yield",
            "prefer_blue_chip": True,
        },
    )
    if df is None:
        return

    meta = {
        "index_name": basket_id,
        "as_of": _utc_now_iso(),
        "title": f"Custom basket {basket_id}",
        "parsed_holdings": [],
        "missing_price_symbols": [],
        "custom_blend": f"{w_blue:.2f},{w_div:.2f},{w_growth:.2f}",
        "sip_profile": "dividend_basket",
    }
    _run_benchmark_sip_flow(
        df,
        meta,
        source_type="custom",
        source_id=basket_id,
        universe_label="custom basket",
        weight_label="target",
        empty_exclusion_msg="All names were excluded. Nothing to allocate.",
        exclusion_unit="names",
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


def _dividend_yield_by_symbol() -> dict[str, float]:
    scr = _build_screener_enriched()
    if scr.empty or "dividend_yield" not in scr.columns:
        return {}
    out: dict[str, float] = {}
    for _, row in scr.iterrows():
        sym = str(row["symbol"]).upper()
        val = pd.to_numeric(row.get("dividend_yield"), errors="coerce")
        out[sym] = float(val) if pd.notna(val) else 0.0
    return out


def apply_dividend_yield_weights(df: pd.DataFrame) -> pd.DataFrame:
    """Renormalize idx_weight by live screener dividend yield (100% yield — use blend for top-up)."""
    yields = _dividend_yield_by_symbol()
    if not yields:
        print("  Could not load screener yields — keeping basket weights.")
        return df
    out = df.copy()
    sym = out["symbol"].astype(str).str.upper()
    dy = sym.map(lambda s: max(float(yields.get(s, 0.0) or 0.0), 0.01))
    total = float(dy.sum())
    if total <= 0:
        return df
    out["idx_weight"] = dy / total * 100.0
    return out.sort_values("idx_weight", ascending=False).reset_index(drop=True)


def apply_dividend_tilt_blend(
    df: pd.DataFrame,
    *,
    basket_share: float = 0.70,
) -> pd.DataFrame:
    """Keep every name in the plan: blend basket weights with dividend-yield weights."""
    if df.empty:
        return df
    basket_share = max(0.0, min(1.0, basket_share))
    yields = _dividend_yield_by_symbol()
    out = df.copy()
    w_b = out["idx_weight"].astype(float)
    b_sum = float(w_b.sum())
    if b_sum <= 0:
        w_b = pd.Series([1.0] * len(out), index=out.index)
        b_sum = float(len(out))
    w_b = w_b / b_sum * 100.0

    if not yields or basket_share >= 1.0 - 1e-9:
        out["idx_weight"] = w_b
        return out.sort_values("idx_weight", ascending=False).reset_index(drop=True)

    sym = out["symbol"].astype(str).str.upper()
    dy = sym.map(lambda s: max(float(yields.get(s, 0.0) or 0.0), 0.01))
    d_sum = float(dy.sum())
    w_d = dy / d_sum * 100.0 if d_sum > 0 else w_b

    out["idx_weight"] = basket_share * w_b + (1.0 - basket_share) * w_d
    return out.sort_values("idx_weight", ascending=False).reset_index(drop=True)


def maybe_rotate_dividend_additions(
    scoped: pd.DataFrame,
    live_df: pd.DataFrame,
) -> pd.DataFrame:
    """Optionally add 2–3 high-yield basket names not yet in this top-up (monthly rotation)."""
    ans = prompt(
        "\nDividend tilt: rotate in 2–3 high-yield basket names you do not hold "
        "yet (held names stay in the plan)? [Y/n]: "
    ).lower()
    if ans in ("n", "no"):
        return scoped

    count_raw = prompt("How many dividend names this month [3]: ").strip() or "3"
    try:
        count = int(count_raw)
    except ValueError:
        count = 3
    count = max(1, min(5, count))

    yields = _dividend_yield_by_symbol()
    live = live_df.copy()
    live["symbol"] = live["symbol"].astype(str).str.upper()
    live["dividend_yield"] = live["symbol"].map(
        lambda s: float(yields.get(s, 0.0) or 0.0)
    )
    ranked = live.sort_values(
        ["dividend_yield", "idx_weight"], ascending=[False, False]
    )
    order = ranked["symbol"].astype(str).tolist()
    if not order:
        return scoped

    in_scope = set(scoped["symbol"].astype(str).str.upper())
    state = load_user_state()
    start = int(state.get("dividend_rotation_index", 0) or 0) % len(order)
    picks: list[str] = []
    for i in range(len(order)):
        sym = order[(start + i) % len(order)]
        if sym not in in_scope:
            picks.append(sym)
        if len(picks) >= count:
            break

    if not picks:
        print("  Top-yield basket names are already in this top-up.")
        return scoped

    next_idx = (start + max(1, len(picks))) % len(order)
    save_user_state(dividend_rotation_index=str(next_idx))

    extra = ranked[ranked["symbol"].isin(picks)].drop(
        columns=["dividend_yield"], errors="ignore"
    )
    print(
        "\n  Dividend rotation — adding to this top-up (does not remove held names): "
        + ", ".join(picks)
    )
    return (
        pd.concat([scoped, extra], ignore_index=True)
        .drop_duplicates(subset=["symbol"], keep="first")
        .sort_values("idx_weight", ascending=False)
        .reset_index(drop=True)
    )


def _pe_ratio_by_symbol() -> dict[str, float]:
    scr = _build_screener_enriched()
    if scr.empty or "pe_ratio" not in scr.columns:
        return {}
    out: dict[str, float] = {}
    for _, row in scr.iterrows():
        sym = str(row["symbol"]).upper()
        pe = pd.to_numeric(row.get("pe_ratio"), errors="coerce")
        if pd.notna(pe):
            out[sym] = float(pe)
    return out


def _is_regular_dps_payer(sym: str) -> tuple[bool, str]:
    """True when name passes balanced DPS income screen (keeps 2/5+ and yield-only)."""
    hist = _dps_history(sym)
    note, _irregular = _describe_dps_regularity(hist)
    tier = _dps_income_tier(hist)
    if _dps_tier_excluded(tier, "balanced"):
        return False, note
    return True, note


def _symbol_quality_check(
    sym: str,
    *,
    max_pe: float | None,
    exclude_eps_decline: bool,
    pe_map: dict[str, float],
    eps_cache: dict[str, tuple[str, bool]],
    protect_regular_dps: bool = False,
) -> tuple[bool, str]:
    if protect_regular_dps:
        ok_dps, dnote = _is_regular_dps_payer(sym)
        if ok_dps:
            return True, f"regular DPS ({dnote})"
        return False, dnote
    sym = sym.upper()
    if max_pe is not None:
        pe = pe_map.get(sym)
        if pe is not None and pe > max_pe + 1e-9:
            return False, f"P/E {pe:.1f} > {max_pe:g}"
    if exclude_eps_decline:
        if sym not in eps_cache:
            eps_cache[sym] = _describe_eps_trend(_company_annual_eps_values(sym))
        note, exclude = eps_cache[sym]
        if exclude:
            return False, note
    return True, "ok"


def _quality_filter_dataframe(
    df: pd.DataFrame,
    *,
    max_pe: float | None,
    exclude_eps_decline: bool,
    protect_regular_dps: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (kept rows, dropped rows with quality_reason)."""
    if df.empty:
        return df, df.iloc[0:0].copy()
    pe_map = _pe_ratio_by_symbol()
    eps_cache: dict[str, tuple[str, bool]] = {}
    kept_rows: list[pd.Series] = []
    dropped: list[dict] = []
    n = len(df)
    if exclude_eps_decline and n > 0 and not protect_regular_dps:
        print(f"  Checking annual EPS on PSX company pages ({n} name(s)…)…")
    elif protect_regular_dps and n > 0:
        print(f"  Dividend-first filter on {n} name(s) (regular DPS payers kept)…")
    for i, (_, row) in enumerate(df.iterrows(), start=1):
        sym = str(row["symbol"]).upper()
        if exclude_eps_decline and not protect_regular_dps and (i == 1 or i == n or i % 5 == 0):
            print(f"    … {i}/{n}", flush=True)
        ok, reason = _symbol_quality_check(
            sym,
            max_pe=max_pe,
            exclude_eps_decline=exclude_eps_decline,
            pe_map=pe_map,
            eps_cache=eps_cache,
            protect_regular_dps=protect_regular_dps,
        )
        if ok:
            kept_rows.append(row)
        else:
            pe = pe_map.get(sym)
            pe_s = f"{pe:.1f}" if pe is not None else "—"
            dropped.append(
                {
                    "symbol": sym,
                    "name": row.get("name", sym),
                    "pe_ratio": pe_s,
                    "quality_reason": reason,
                }
            )
    kept = pd.DataFrame(kept_rows) if kept_rows else df.iloc[0:0].copy()
    drop_df = pd.DataFrame(dropped)
    return kept.reset_index(drop=True), drop_df


def prompt_topup_quality_and_replacements(
    scoped: pd.DataFrame,
    live_df: pd.DataFrame,
) -> pd.DataFrame:
    """Drop names failing P/E or EPS rules; optionally add other benchmark lines."""
    ans = prompt(
        "\nApply a buy filter this month? [y/N] "
        "(tip: after DPS table, use 1 dividend-first or N — strict P/E removes FFC/MEBL/LUCK): "
    ).lower()
    if ans not in ("y", "yes"):
        return scoped

    style = prompt(
        "  1. Dividend-first — drop irregular/no DPS only; keep regular payers (recommended)\n"
        "  2. Strict — P/E cap + optional EPS (can remove good dividend names)\n"
        "Select [1]: "
    ).strip() or "1"
    dividend_first = style not in ("2", "strict", "pe", "eps")

    if dividend_first:
        max_pe = None
        exclude_eps = False
        print(
            "  Dividend-first: skipping only irregular / no-DPS names; "
            "regular payers stay even if P/E or EPS look weak."
        )
    else:
        max_pe_raw = prompt(
            "Maximum P/E (Enter=12, type none for no cap) [12]: "
        ).strip()
        if max_pe_raw == "":
            max_pe = 12.0
        elif max_pe_raw.lower() in ("none", "no", "skip", "-", "*"):
            max_pe = None
        else:
            try:
                max_pe = float(max_pe_raw.replace(",", ""))
            except ValueError:
                print("  Invalid P/E — using 12.")
                max_pe = 12.0
        eps_ans = prompt("Skip EPS down 2+ years in a row? [Y/n]: ").lower()
        exclude_eps = eps_ans not in ("n", "no")

    kept, dropped = _quality_filter_dataframe(
        scoped,
        max_pe=max_pe,
        exclude_eps_decline=exclude_eps,
        protect_regular_dps=dividend_first,
    )
    if dropped.empty:
        print("  All names in this top-up pass the quality rules.")
        return scoped

    print("\n  Removed from this month's buys (you can still hold them):")
    print(dropped.to_string(index=False))

    if len(scoped) >= 3 and len(kept) < max(3, (len(scoped) + 1) // 2):
        print(
            f"\n  Warning: only {len(kept)} name(s) left — "
            "this month's SIP would concentrate in very few lines."
        )
        relax = prompt(
            "Keep the full top-up list and skip the quality filter this month? [Y/n]: "
        ).lower()
        if relax not in ("n", "no"):
            print("  Keeping all names in this top-up (no quality filter).")
            return scoped

    add_prompt = (
        "\nAdd replacement(s) from the basket (dividend-first picks by DPS)? [Y/n]: "
        if dividend_first
        else "\nAdd replacement(s) from the live basket that pass the same rules? [Y/n]: "
    )
    add_ans = prompt(add_prompt).lower()
    if add_ans and add_ans not in ("y", "yes", "n", "no"):
        print("  (Unrecognized answer — treating as Yes for replacements.)")
        add_ans = "y"
    if add_ans in ("n", "no"):
        if kept.empty:
            print("  Nothing left to allocate — keeping original list.")
            return scoped
        return kept

    live = live_df.copy()
    live["symbol"] = live["symbol"].astype(str).str.upper()
    in_scope = set(scoped["symbol"].astype(str).str.upper())
    pool = live[~live["symbol"].isin(in_scope)].copy()
    if pool.empty:
        print("  No other basket names to add.")
        return kept if not kept.empty else scoped

    pe_map = _pe_ratio_by_symbol()
    eps_cache: dict[str, tuple[str, bool]] = {}
    good_rows: list[pd.Series] = []
    n_pool = len(pool)
    if exclude_eps:
        print(f"  Screening {n_pool} other basket name(s)…")
    for i, (_, row) in enumerate(pool.iterrows(), start=1):
        sym = str(row["symbol"]).upper()
        if exclude_eps and (i == 1 or i == n_pool or i % 5 == 0):
            print(f"    … {i}/{n_pool}", flush=True)
        ok, _reason = _symbol_quality_check(
            sym,
            max_pe=max_pe,
            exclude_eps_decline=exclude_eps,
            pe_map=pe_map,
            eps_cache=eps_cache,
            protect_regular_dps=dividend_first,
        )
        if ok:
            good_rows.append(row)
    candidates = pd.DataFrame(good_rows) if good_rows else pool.iloc[0:0].copy()
    if candidates.empty:
        print("  No basket names pass the quality rules.")
        return kept if not kept.empty else scoped

    yields = _dividend_yield_by_symbol()
    if dividend_first:
        pref = "1"
    else:
        pref = prompt(
            "Pick replacements by [1] dividend yield  [2] basket weight [1]: "
        ).strip() or "1"
    if pref in ("2", "weight", "basket", "w"):
        candidates = candidates.sort_values("idx_weight", ascending=False)
    else:
        candidates["dividend_yield"] = candidates["symbol"].map(
            lambda s: float(yields.get(str(s).upper(), 0.0) or 0.0)
        )
        candidates = candidates.sort_values(
            ["dividend_yield", "idx_weight"], ascending=[False, False]
        )

    default_n = max(1, len(dropped))
    n_add = int(
        prompt_positive_float(
            f"How many names to add (≤ {len(candidates)} available)",
            float(default_n),
        )
    )
    n_add = max(1, min(n_add, len(candidates)))

    manual = prompt(
        "Or force symbol(s) from the list above (comma-separated, optional): "
    ).strip()
    picks: list[pd.Series] = []
    if manual:
        valid = set(candidates["symbol"].astype(str).str.upper())
        for tok in manual.replace(";", ",").split(","):
            sym = tok.strip().upper()
            if sym and sym in valid:
                row = candidates[
                    candidates["symbol"].astype(str).str.upper() == sym
                ].iloc[0]
                picks.append(row)
            elif sym:
                print(f"  Skipping {sym} — not in quality-filtered basket pool.")
    if not picks:
        picks = [candidates.iloc[i] for i in range(n_add)]
    else:
        picks = picks[:n_add]

    add_df = pd.DataFrame(picks)
    print("\n  Adding to this top-up:")
    show = add_df[["symbol", "name", "idx_weight"]].copy()
    if "dividend_yield" in add_df.columns:
        show["dividend_yield"] = add_df["dividend_yield"].map(
            lambda v: f"{float(v):.2f}%"
        )
    show["idx_weight"] = show["idx_weight"].map(lambda w: f"{w:.2f}%")
    print(show.to_string(index=False))

    if kept.empty:
        return add_df.reset_index(drop=True)
    return (
        pd.concat([kept, add_df], ignore_index=True)
        .drop_duplicates(subset=["symbol"], keep="first")
        .sort_values("idx_weight", ascending=False)
        .reset_index(drop=True)
    )


def prompt_optional_topup_exclusions(
    df: pd.DataFrame,
    *,
    universe_label: str,
) -> pd.DataFrame:
    ans = prompt(
        "\nExclude sectors or symbols from this top-up only? [y/N]: "
    ).lower()
    if ans not in ("y", "yes"):
        return df
    exclude_sectors, exclude_symbols = prompt_exclusions(
        df, universe_label=universe_label
    )
    filtered = apply_exclusions(df, exclude_sectors, exclude_symbols)
    if filtered.empty:
        print("  All names excluded — keeping the prior selection.")
        return df
    if len(filtered) < len(df):
        print(
            f"\n  Top-up universe: {len(filtered)} name(s) "
            f"(excluded {len(df) - len(filtered)})."
        )
    return filtered


def apply_topup_allocation_tweaks(
    scoped: pd.DataFrame,
    *,
    live_df: pd.DataFrame,
    universe_label: str,
    topup_strategy: str,
    sip_profile: str = "standard",
) -> pd.DataFrame:
    if topup_strategy == "dividend_tilt":
        scoped = maybe_rotate_dividend_additions(scoped, live_df)
    scoped = prompt_optional_topup_exclusions(scoped, universe_label=universe_label)
    if sip_profile != "dividend_basket":
        scoped = prompt_topup_quality_and_replacements(scoped, live_df)
    if scoped.empty:
        return scoped
    if topup_strategy == "dividend_tilt":
        share_raw = prompt(
            "\nDividend tilt: share of weights from basket vs yield "
            "(0.7 = 70% basket / 30% yield) [0.7]: "
        ).strip()
        try:
            basket_share = float(share_raw) if share_raw else 0.70
        except ValueError:
            basket_share = 0.70
        basket_share = max(0.0, min(1.0, basket_share))
        print(
            f"\n  Dividend tilt: {basket_share * 100:.0f}% basket weights + "
            f"{(1 - basket_share) * 100:.0f}% screener dividend yield "
            f"— all {len(scoped)} name(s) stay in the plan."
        )
        scoped = apply_dividend_tilt_blend(scoped, basket_share=basket_share)
        show = scoped[["symbol", "name", "idx_weight"]].copy()
        show["idx_weight"] = show["idx_weight"].map(lambda w: f"{w:.2f}%")
        print(show.to_string(index=False))
    return scoped


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


def _finalize_plan_metrics(plan: pd.DataFrame, investment_pkr: float) -> pd.DataFrame:
    plan["invested_pkr"] = plan["shares"] * plan["effective_price"]
    total_invested = plan["invested_pkr"].sum()
    if total_invested > 0:
        plan["actual_weight_pct"] = plan["invested_pkr"] / total_invested * 100.0
    else:
        plan["actual_weight_pct"] = 0.0
    if "target_pkr" in plan.columns:
        plan["leftover_pkr"] = plan["target_pkr"] - plan["invested_pkr"]
    return plan.sort_values("weight_pct", ascending=False).reset_index(drop=True)


def compute_equal_held_topup_plan(
    df: pd.DataFrame,
    investment_pkr: float,
    use_board_lot: bool,
    cost_buffer_pct: float,
) -> pd.DataFrame:
    if df.empty:
        return df
    plan = df.copy()
    plan["effective_price"] = plan["price"] * (1.0 + cost_buffer_pct / 100.0)
    n = len(plan)
    plan["weight_pct"] = 100.0 / n
    plan["target_pkr"] = investment_pkr / n
    plan["shares"] = _initial_share_counts(
        plan["target_pkr"], plan["effective_price"], use_board_lot
    )
    plan = _redeploy_cash_to_weights(plan, investment_pkr, use_board_lot)
    return _finalize_plan_metrics(plan, investment_pkr)


def compute_benchmark_gap_topup_plan(
    live_df: pd.DataFrame,
    prior_shares: dict[str, int],
    investment_pkr: float,
    use_board_lot: bool,
    cost_buffer_pct: float,
) -> pd.DataFrame:
    if live_df.empty:
        return live_df
    live = live_df.copy()
    live["symbol"] = live["symbol"].astype(str).str.upper()
    total_w = float(live["idx_weight"].sum())
    if total_w <= 0:
        print("Benchmark weights are zero; cannot top-up.")
        sys.exit(1)

    live["effective_price"] = live["price"] * (1.0 + cost_buffer_pct / 100.0)
    shares_held = live["symbol"].map(lambda s: int(prior_shares.get(s, 0)))
    live["current_value"] = shares_held * live["price"]
    current_value = float(live["current_value"].sum())
    post_total = current_value + investment_pkr

    live["weight_pct"] = live["idx_weight"] / total_w * 100.0
    live["target_value"] = post_total * (live["idx_weight"] / total_w)
    live["gap_pkr"] = (live["target_value"] - live["current_value"]).clip(lower=0.0)
    gap_sum = float(live["gap_pkr"].sum())
    if gap_sum > 1e-6:
        live["target_pkr"] = investment_pkr * (live["gap_pkr"] / gap_sum)
    else:
        live["target_pkr"] = investment_pkr * (live["idx_weight"] / total_w)

    live["shares"] = _initial_share_counts(
        live["target_pkr"], live["effective_price"], use_board_lot
    )
    live = _redeploy_cash_to_weights(live, investment_pkr, use_board_lot)
    return _finalize_plan_metrics(live, investment_pkr)


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
        shares_add = int(plan_by_sym.loc[sym, "shares"]) if sym in plan_by_sym.index else 0
        invested_this = (
            float(plan_by_sym.loc[sym, "invested_pkr"]) if sym in plan_by_sym.index else 0.0
        )
        shares_before = int(prior.get(sym, 0))
        shares_total = shares_before + shares_add
        return {
            "symbol": sym,
            "name": name,
            "sector_name": sector_name or "",
            "basket_shares": basket_shares,
            "price": price,
            "idx_weight_pct": idx_weight,
            "shares_before_run": shares_before,
            "shares_to_add_this_run": shares_add,
            "shares_held": shares_total,
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


def prompt_holdings_snapshot_path(default_name: str) -> str:
    print("  Path must be a CSV file (not a folder). Relative, absolute, or ~ OK.")
    while True:
        path_raw = prompt(f"Holdings file path [{default_name}]: ") or default_name
        path = resolve_user_path(path_raw)
        if os.path.isdir(path):
            suggested = os.path.join(path.rstrip(os.sep), default_name)
            print(f"  {path!r} is a directory, not a file.")
            fix = prompt(f"  Save as {suggested}? [Y/n]: ").lower()
            if fix in ("", "y", "yes"):
                path = suggested
            else:
                continue
        parent = os.path.dirname(path)
        if parent and not os.path.isdir(parent):
            mk = prompt(f"  Create folder {parent}? [y/N]: ").lower()
            if mk in ("y", "yes"):
                os.makedirs(parent, exist_ok=True)
            else:
                print("  Pick another path or create the folder.")
                continue
        if os.path.isdir(path):
            continue
        return path


def write_holdings_snapshot(
    path: str,
    table: pd.DataFrame,
    header: dict[str, str],
) -> None:
    if os.path.isdir(path):
        raise IsADirectoryError(f"Holdings path is a directory: {path}")
    lines = [
        f"# psx-sip-holdings snapshot_version={HOLDINGS_SNAPSHOT_VERSION}",
    ]
    for key in sorted(header.keys()):
        val = str(header[key]).replace("\n", " ")
        lines.append(f"# {key}={val}")
    lines.append(
        "# shares_before_run + shares_to_add_this_run = shares_held (use shares_held next top-up)."
    )
    lines.append(
        "# Edit shares_held only if you traded outside this planner."
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


def _coerce_share_counts(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0).astype(int)


def normalize_holdings_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Accept legacy snapshots/plan CSVs with varying column names."""
    out = df.copy()
    if "symbol" not in out.columns:
        print("  File must include a 'symbol' column.")
        sys.exit(1)
    out["symbol"] = out["symbol"].astype(str).str.upper().str.strip()

    legacy_note: list[str] = []

    if "shares_to_add_this_run" not in out.columns and "shares_this_run" in out.columns:
        out["shares_to_add_this_run"] = _coerce_share_counts(out["shares_this_run"])
        legacy_note.append("shares_this_run → shares_to_add_this_run (display only)")

    if "shares_before_run" in out.columns:
        out["shares_before_run"] = _coerce_share_counts(out["shares_before_run"])
    else:
        out["shares_before_run"] = 0

    if "shares_to_add_this_run" in out.columns:
        out["shares_to_add_this_run"] = _coerce_share_counts(
            out["shares_to_add_this_run"]
        )

    if "shares_held" in out.columns:
        out["shares_held"] = _coerce_share_counts(out["shares_held"])
    elif "shares_held_total" in out.columns:
        out["shares_held"] = _coerce_share_counts(out["shares_held_total"])
        legacy_note.append("shares_held_total → shares_held")
    elif (
        "shares_before_run" in out.columns
        and "shares_to_add_this_run" in out.columns
    ):
        out["shares_held"] = out["shares_before_run"] + out["shares_to_add_this_run"]
        legacy_note.append("computed shares_held from before + add columns")
    elif "shares" in out.columns:
        out["shares_held"] = _coerce_share_counts(out["shares"])
        legacy_note.append("plan-style 'shares' used as shares_held (position totals)")
    else:
        print(
            "  Could not find share counts. Need one of: shares_held, "
            "shares_held_total, shares, or shares_before_run + shares_to_add_this_run."
        )
        sys.exit(1)

    if int(out["shares_held"].sum()) == 0:
        if "shares_to_add_this_run" in out.columns and int(
            out["shares_to_add_this_run"].sum()
        ) > 0:
            out["shares_held"] = out["shares_to_add_this_run"]
            legacy_note.append(
                "no shares_held totals — using shares_to_add_this_run as position"
            )
        elif "shares" in out.columns and int(_coerce_share_counts(out["shares"]).sum()) > 0:
            out["shares_held"] = _coerce_share_counts(out["shares"])
            legacy_note.append("using 'shares' column as position")

    if legacy_note:
        print("  Legacy / simplified file: " + "; ".join(legacy_note))

    return out


def prior_shares_from_holdings_df(df: pd.DataFrame) -> dict[str, int]:
    norm = normalize_holdings_dataframe(df)
    held = norm[norm["shares_held"] > 0]
    if held.empty:
        return {}
    return (
        held.groupby("symbol", as_index=False)["shares_held"]
        .max()
        .set_index("symbol")["shares_held"]
        .astype(int)
        .to_dict()
    )


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
    snap_ver = header.get("snapshot_version")
    if snap_ver not in (None, "1", HOLDINGS_SNAPSHOT_VERSION):
        print(
            f"  Warning: snapshot version {snap_ver} "
            f"(planner expects {HOLDINGS_SNAPSHOT_VERSION})."
        )
    from io import StringIO

    if not data_lines or not data_lines[0].strip():
        print("  Holdings file has no data rows.")
        sys.exit(1)
    df = normalize_holdings_dataframe(pd.read_csv(StringIO("".join(data_lines))))
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
    path = prompt_holdings_snapshot_path(default_name)
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
    adds = table["shares_to_add_this_run"].sum() if "shares_to_add_this_run" in table.columns else 0
    inv = table["invested_pkr_this_run"].sum() if "invested_pkr_this_run" in table.columns else 0
    snap_header = {
        "planner_version": __version__,
        "generated_at": _utc_now_iso(),
        "source_type": source_type,
        "source_id": source_id,
        "investment_pkr_this_run": f"{investment_pkr:.2f}",
        "total_shares_added_this_run": str(int(adds)),
        "total_invested_pkr_this_run": f"{float(inv):.2f}",
        "cost_buffer_pct": f"{cost_buffer_pct:.2f}",
        "board_lot": "yes" if use_board_lot else "no",
        "basket_as_of": str((meta or {}).get("as_of", "")),
    }
    action = table[table["shares_to_add_this_run"] > 0].copy()
    if not action.empty:
        show = action[
            [
                "symbol",
                "shares_before_run",
                "shares_to_add_this_run",
                "shares_held",
                "invested_pkr_this_run",
            ]
        ]
        print("\n  This top-up / plan — buy this run (also stored in the CSV):")
        print(show.to_string(index=False))
        print(
            f"\n  Run totals: +{int(action['shares_to_add_this_run'].sum())} shares, "
            f"PKR {action['invested_pkr_this_run'].sum():,.2f} invested "
            f"(→ shares_held column is your cumulative total for the next run)."
        )

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


TOPUP_OVERWEIGHT_PP = 2.0
TOPUP_UNDERWEIGHT_PP = 2.0
TOPUP_LIGHTEN_MAX_PE = 12.0
TOPUP_FAVOR_MIN_BENCH_PCT = 3.0


def _benchmark_holdings_detail(
    prior_shares: dict[str, int],
    live_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, float, set[str]] | None:
    """Value in-basket holdings; compare weights to benchmark."""
    target = _benchmark_target_pct(live_df)
    if target.empty:
        return None

    live = live_df.copy()
    live["symbol"] = live["symbol"].astype(str).str.upper()
    by_sym = live.set_index("symbol")
    pe_map = _pe_ratio_by_symbol()
    div_map = _dividend_yield_by_symbol()

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
        row = by_sym.loc[sym]
        price = float(row["price"])
        value = sh * price
        total_value += value
        held_rows.append(
            {
                "symbol": sym,
                "name": str(row.get("name", sym)),
                "sector_name": str(row.get("sector_name", "UNKNOWN")),
                "shares": sh,
                "price": price,
                "value_pkr": value,
            }
        )

    if total_value <= 0:
        return None

    held_symbols = {r["symbol"] for r in held_rows}
    target_pct = target * 100.0
    held_only_total = float(target.loc[list(held_symbols)].sum()) * 100.0

    detail: list[dict] = []
    for r in held_rows:
        sym = r["symbol"]
        your_pct = r["value_pkr"] / total_value * 100.0
        bench_pct = float(target_pct[sym])
        bench_on_held = (
            bench_pct / held_only_total * 100.0 if held_only_total > 0 else 0.0
        )
        pe = pe_map.get(sym)
        detail.append(
            {
                **r,
                "your_pct": your_pct,
                "benchmark_pct": bench_pct,
                "bench_on_held_pct": bench_on_held,
                "weight_gap_pp": your_pct - bench_on_held,
                "pe_ratio": pe,
                "dividend_yield": float(div_map.get(sym, 0.0) or 0.0),
            }
        )

    held_df = pd.DataFrame(detail).sort_values("value_pkr", ascending=False)
    orphan_df = pd.DataFrame(orphan_rows)
    return held_df, orphan_df, total_value, held_symbols


def report_topup_new_money_guide(
    prior_shares: dict[str, int],
    live_df: pd.DataFrame,
    source_type: str,
    source_id: str,
) -> tuple[set[str], set[str], bool, bool]:
    """
    Show diversification + dividend profile; suggest new SIP only (no sales).
    Returns symbols to skip/add on this top-up and whether user opted in.
    """
    empty: tuple[set[str], set[str], bool, bool] = (set(), set(), False, False)
    parsed = _benchmark_holdings_detail(prior_shares, live_df)
    if parsed is None:
        print("\n  Could not value holdings for mix report (check symbols/shares).")
        return empty
    held_df, orphan_df, total_value, held_symbols = parsed

    print(f"\n{'=' * 72}")
    print(f"  Your portfolio mix vs {source_type.upper()} {source_id}")
    print(f"  (Review only — nothing is sold or removed from your holdings file.)")
    print(f"{'=' * 72}")
    print(
        f"\n  Marked value (in-basket): PKR {total_value:,.2f} · "
        f"{len(held_symbols)} / {len(live_df)} basket names held"
    )

    user_sec = (
        held_df.groupby("sector_name", as_index=False)["value_pkr"]
        .sum()
        .assign(sector_pct=lambda d: d["value_pkr"] / total_value * 100.0)
    )
    bench_sec = sector_weight_table(live_df).rename(
        columns={"sector_weight": "bench_sector_pct"}
    )
    mix = user_sec.merge(
        bench_sec[["sector_name", "bench_sector_pct", "stocks"]],
        on="sector_name",
        how="outer",
    ).fillna(0.0)
    mix["sector_pct"] = mix["sector_pct"].fillna(0.0)
    mix["gap_pp"] = mix["sector_pct"] - mix["bench_sector_pct"]
    mix = mix.sort_values("sector_pct", ascending=False)

    print("\n  Sector diversification (by current market value):\n")
    sec_show = mix[
        ["sector_name", "sector_pct", "bench_sector_pct", "gap_pp", "stocks"]
    ].copy()
    sec_show["sector_pct"] = sec_show["sector_pct"].map(
        lambda v: f"{v:.1f}%" if v > 0 else "—"
    )
    sec_show["bench_sector_pct"] = sec_show["bench_sector_pct"].map(
        lambda v: f"{v:.1f}%"
    )
    sec_show["gap_pp"] = sec_show["gap_pp"].map(lambda v: f"{v:+.1f} pp")
    sec_show = sec_show.rename(
        columns={
            "sector_name": "sector",
            "sector_pct": "your_%",
            "bench_sector_pct": "bench_%",
            "gap_pp": "vs_bench",
            "stocks": "names_in_basket",
        }
    )
    print(sec_show.to_string(index=False))

    top3 = held_df.head(3)["your_pct"].sum()
    top1 = float(held_df.iloc[0]["your_pct"]) if not held_df.empty else 0.0
    n_sectors = len(user_sec)
    print(
        f"\n  Concentration: largest line {top1:.1f}% · top 3 lines {top3:.1f}% · "
        f"{n_sectors} sector(s) represented"
    )

    port_div = float(
        (held_df["value_pkr"] * held_df["dividend_yield"]).sum() / total_value
    )
    live = live_df.copy()
    live["symbol"] = live["symbol"].astype(str).str.upper()
    w = live["idx_weight"] / float(live["idx_weight"].sum())
    live["dividend_yield"] = live["symbol"].map(
        lambda s: float(_dividend_yield_by_symbol().get(s, 0.0) or 0.0)
    )
    bench_div = float((w * live["dividend_yield"]).sum())
    print(
        f"\n  Dividend yield (screener, value-weighted): "
        f"your book ≈ {port_div:.2f}% · full basket ≈ {bench_div:.2f}%"
    )
    if port_div + 0.25 < bench_div:
        print(
            "  → New cash tilted toward higher-yield basket names would "
            "move you toward the basket’s income profile."
        )

    lighten_rows: list[dict] = []
    for _, row in held_df.iterrows():
        reasons: list[str] = []
        gap = float(row["weight_gap_pp"])
        if gap >= TOPUP_OVERWEIGHT_PP:
            reasons.append(f"overweight +{gap:.1f}pp vs held-bench")
        pe = row.get("pe_ratio")
        if pe is not None and pd.notna(pe) and float(pe) > TOPUP_LIGHTEN_MAX_PE:
            reasons.append(f"P/E {float(pe):.1f} > {TOPUP_LIGHTEN_MAX_PE:g}")
        if reasons:
            lighten_rows.append(
                {
                    "symbol": row["symbol"],
                    "name": row["name"],
                    "your_pct": row["your_pct"],
                    "bench_on_held_pct": row["bench_on_held_pct"],
                    "dividend_yield": row["dividend_yield"],
                    "pe_ratio": pe,
                    "suggestion": "skip new buys this month",
                    "reasons": "; ".join(reasons),
                }
            )

    fund_check = prompt(
        "\nCheck EPS (PSX) + DPS dividend history on held names? [y/N]: "
    ).lower()
    dps_notes: dict[str, tuple[str, bool]] = {}
    if fund_check in ("y", "yes"):
        n = len(held_df)
        print(f"  Reading annual EPS ({n} held names)…")
        for i, (_, row) in enumerate(held_df.iterrows(), start=1):
            sym = str(row["symbol"]).upper()
            if i == 1 or i == n or i % 4 == 0:
                print(f"    … EPS {i}/{n}", flush=True)
            note, exclude = _describe_eps_trend(_company_annual_eps_values(sym))
            if exclude:
                found = next(
                    (r for r in lighten_rows if r["symbol"] == sym), None
                )
                if found:
                    found["reasons"] += f"; {note}"
                else:
                    lighten_rows.append(
                        {
                            "symbol": sym,
                            "name": row["name"],
                            "your_pct": row["your_pct"],
                            "bench_on_held_pct": row["bench_on_held_pct"],
                            "dividend_yield": row["dividend_yield"],
                            "pe_ratio": row.get("pe_ratio"),
                            "suggestion": "skip new buys this month",
                            "reasons": note,
                        }
                    )
        held_syms = [str(s).upper() for s in held_df["symbol"].astype(str)]
        dps_notes = _print_dps_consistency_table(held_syms)
        skip_irregular = prompt(
            "\nSkip new buys this month for irregular / no DPS history names? [y/N]: "
        ).lower() in ("y", "yes")
        if skip_irregular:
            for sym, (_note, irregular) in dps_notes.items():
                if not irregular:
                    continue
                row = held_df[held_df["symbol"].astype(str).str.upper() == sym]
                if row.empty:
                    continue
                r0 = row.iloc[0]
                dps_note = _describe_dps_regularity(_dps_history(sym))[0]
                found = next(
                    (r for r in lighten_rows if r["symbol"] == sym), None
                )
                if found:
                    found["reasons"] += f"; {dps_note}"
                else:
                    lighten_rows.append(
                        {
                            "symbol": sym,
                            "name": r0["name"],
                            "your_pct": r0["your_pct"],
                            "bench_on_held_pct": r0["bench_on_held_pct"],
                            "dividend_yield": r0["dividend_yield"],
                            "pe_ratio": r0.get("pe_ratio"),
                            "suggestion": "skip new buys this month",
                            "reasons": dps_note,
                        }
                    )

    skip_symbols = {str(r["symbol"]).upper() for r in lighten_rows}

    print("\n  --- New SIP: deprioritize (you still hold; skip only this month) ---")
    if lighten_rows:
        ld = pd.DataFrame(lighten_rows).sort_values("your_pct", ascending=False)
        show = ld[
            [
                "symbol",
                "your_pct",
                "bench_on_held_pct",
                "dividend_yield",
                "pe_ratio",
                "reasons",
            ]
        ].copy()
        show["your_pct"] = show["your_pct"].map(lambda v: f"{v:.1f}%")
        show["bench_on_held_pct"] = show["bench_on_held_pct"].map(
            lambda v: f"{v:.1f}%"
        )
        show["dividend_yield"] = show["dividend_yield"].map(
            lambda v: f"{v:.2f}%"
        )
        show["pe_ratio"] = show["pe_ratio"].map(
            lambda v: f"{v:.1f}" if v is not None and pd.notna(v) else "—"
        )
        print(show.to_string(index=False))
    else:
        print("  None flagged (overweight / P/E / EPS rules).")

    favor_rows: list[dict] = []
    live_syms = set(live["symbol"].astype(str).str.upper())
    missing = live[~live["symbol"].isin(held_symbols)].copy()
    for _, row in missing.iterrows():
        sym = str(row["symbol"]).upper()
        bw = float(row["idx_weight"])
        dy = float(_dividend_yield_by_symbol().get(sym, 0.0) or 0.0)
        if bw < TOPUP_FAVOR_MIN_BENCH_PCT and dy < max(bench_div, 3.0):
            continue
        dps_note = ""
        if fund_check in ("y", "yes"):
            dps_note = _describe_dps_regularity(_dps_history(sym))[0]
        favor_rows.append(
            {
                "symbol": sym,
                "name": row.get("name", sym),
                "benchmark_pct": bw,
                "dividend_yield": dy,
                "suggestion": "add with new cash",
                "reasons": f"not held; basket {bw:.1f}%"
                + (f"; {dps_note}" if dps_note else ""),
            }
        )
    favor_rows.sort(
        key=lambda r: (r["dividend_yield"], r["benchmark_pct"]), reverse=True
    )

    for _, row in held_df.iterrows():
        gap = float(row["weight_gap_pp"])
        if gap <= -TOPUP_UNDERWEIGHT_PP:
            sym = str(row["symbol"]).upper()
            favor_rows.append(
                {
                    "symbol": sym,
                    "name": row["name"],
                    "benchmark_pct": float(row["benchmark_pct"]),
                    "dividend_yield": float(row["dividend_yield"]),
                    "suggestion": "add more (underweight)",
                    "reasons": f"underweight {gap:.1f}pp vs held-bench",
                }
            )

    print("\n  --- New SIP: favor (same rules — buys only) ---")
    if favor_rows:
        fd = pd.DataFrame(favor_rows).drop_duplicates(subset=["symbol"], keep="first")
        fd = fd.sort_values(
            ["dividend_yield", "benchmark_pct"], ascending=[False, False]
        )
        show = fd.head(12)[
            [
                "symbol",
                "name",
                "benchmark_pct",
                "dividend_yield",
                "suggestion",
                "reasons",
            ]
        ].copy()
        show["benchmark_pct"] = show["benchmark_pct"].map(lambda v: f"{v:.1f}%")
        show["dividend_yield"] = show["dividend_yield"].map(
            lambda v: f"{v:.2f}%"
        )
        print(show.to_string(index=False))
    else:
        print("  No basket names stand out — try benchmark gap-fill (strategy 3).")

    add_symbols = set()
    if favor_rows:
        fd = pd.DataFrame(favor_rows).drop_duplicates(subset=["symbol"], keep="first")
        fd = fd.sort_values(
            ["dividend_yield", "benchmark_pct"], ascending=[False, False]
        )
        for sym in fd.head(5)["symbol"].astype(str).str.upper():
            if sym not in held_symbols:
                add_symbols.add(sym)

    apply_skip = False
    apply_add = False
    if skip_symbols:
        apply_skip = prompt(
            "\nApply deprioritize list to THIS top-up (skip new buys in those symbols)? [y/N]: "
        ).lower() in ("y", "yes")
    if add_symbols:
        apply_add = prompt(
            "Add top suggested basket name(s) you do not hold yet to THIS top-up? [y/N]: "
        ).lower() in ("y", "yes")

    if orphan_df is not None and not orphan_df.empty:
        print("\n  Symbols in your file but not in the live basket:")
        for _, r in orphan_df.iterrows():
            print(f"    {r['symbol']}: {int(r['shares'])} shares")

    print(
        "\n  Next: pick top-up strategy (4 = dividend tilt), then optional "
        "sector exclusions / quality filter."
    )
    return skip_symbols, add_symbols, apply_skip, apply_add


def apply_new_money_hint_selection(
    scoped: pd.DataFrame,
    live_df: pd.DataFrame,
    skip_symbols: set[str],
    add_symbols: set[str],
    apply_skip: bool,
    apply_add: bool,
) -> pd.DataFrame:
    if scoped.empty:
        return scoped
    out = scoped.copy()
    out["symbol"] = out["symbol"].astype(str).str.upper()
    if apply_skip and skip_symbols:
        before = len(out)
        out = out[~out["symbol"].isin(skip_symbols)].copy()
        removed = before - len(out)
        if removed:
            print(
                f"\n  Applied mix guide: skipped new buys in "
                f"{', '.join(sorted(skip_symbols & set(scoped['symbol'].astype(str).str.upper())))}."
            )
    if apply_add and add_symbols:
        live = live_df.copy()
        live["symbol"] = live["symbol"].astype(str).str.upper()
        have = set(out["symbol"].astype(str))
        extra = live[live["symbol"].isin(add_symbols - have)]
        if not extra.empty:
            out = (
                pd.concat([out, extra], ignore_index=True)
                .drop_duplicates(subset=["symbol"], keep="first")
                .sort_values("idx_weight", ascending=False)
                .reset_index(drop=True)
            )
            print(
                "  Applied mix guide: added to this top-up: "
                + ", ".join(extra["symbol"].astype(str).tolist())
            )
    return out.reset_index(drop=True)


def report_holdings_alignment(
    prior_shares: dict[str, int],
    live_df: pd.DataFrame,
    source_type: str,
    source_id: str,
    *,
    compact: bool = False,
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

    show = pd.DataFrame(detail)
    if compact:
        print("\n  Held lines (largest by value):\n")
        disp = show.sort_values("value_pkr", ascending=False).head(8)
    else:
        print("\n  Your holdings vs benchmark (in-basket only):\n")
        disp = show.sort_values("symbol")
    disp = disp[
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
    print(disp.to_string(index=False))
    if not compact:
        print(
            "\n  vs_renorm_bench: your weight minus benchmark weight re-normalized "
            "to names you hold (+ = overweight within your portfolio)."
        )

    if orphan_rows:
        print("\n  Not in current benchmark (still in your file):")
        for r in orphan_rows:
            print(f"    {r['symbol']}: {r['shares']} shares")

    if not compact:
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
    prior, snap_header, snap_df = choose_topup_holdings_source(
        resolve_default_holdings_snapshot_path(source_id),
        source_type,
        source_id,
    )
    report_holdings_alignment(prior, live_df, source_type, source_id)

    ans = prompt("\nPlan a top-up SIP from these same holdings? [y/N]: ").lower()
    if ans not in ("y", "yes"):
        return

    if snap_df is not None and not snap_df.empty:
        report_basket_drift(snap_df, live_df, snap_header)
    skip_syms, add_syms, apply_skip, apply_add = report_topup_new_money_guide(
        prior, live_df, source_type, source_id
    )
    topup_strategy = choose_topup_strategy()
    scoped, prior = run_topup_from_prior_shares(
        prior,
        live_df,
        source_type,
        source_id,
        topup_strategy=topup_strategy,
        snapshot_df=snap_df,
        snapshot_header=snap_header,
        show_basket_drift=False,
        show_switch_hints=False,
    )
    if scoped.empty:
        print("  No symbols to allocate; cannot top-up.")
        return
    scoped = apply_new_money_hint_selection(
        scoped,
        live_df,
        skip_syms,
        add_syms,
        apply_skip,
        apply_add,
    )
    if scoped.empty:
        print("  No symbols left after mix guide.")
        return
    scoped = apply_topup_allocation_tweaks(
        scoped,
        live_df=live_df,
        universe_label=universe_label_from_source(source_type, source_id),
        topup_strategy=topup_strategy,
    )
    if scoped.empty:
        print("  No symbols left after adjustments.")
        return
    investment_pkr = prompt_positive_float(
        "\nMonthly SIP investment amount (PKR)"
    )
    _finalize_sip_plan(
        scoped,
        investment_pkr,
        default_csv=export_plan_csv_name(source_id),
        source_type=source_type,
        source_id=source_id,
        live_df=live_df,
        meta=meta,
        prior_shares=prior,
        topup_strategy=topup_strategy,
    )


def report_basket_drift(
    snapshot_df: pd.DataFrame,
    live_df: pd.DataFrame,
    header: dict[str, str] | None,
) -> None:
    """Compare saved snapshot basket lines to live index/ETF/custom weights."""
    if snapshot_df.empty or live_df.empty:
        return
    snap = snapshot_df.copy()
    snap["symbol"] = snap["symbol"].astype(str).str.upper()
    if "idx_weight_pct" in snap.columns:
        snap["saved_weight"] = pd.to_numeric(
            snap["idx_weight_pct"], errors="coerce"
        )
    else:
        snap["saved_weight"] = float("nan")

    live = live_df.copy()
    live["symbol"] = live["symbol"].astype(str).str.upper()
    live_w = live.set_index("symbol")["idx_weight"]
    live_total = float(live_w.sum()) or 1.0
    live_pct = live_w / live_total * 100.0

    basket_rows = snap
    if "data_status" in snap.columns:
        st = snap["data_status"]
        basket_rows = snap[
            st.isna() | st.astype(str).str.lower().isin(("ok", "", "nan"))
        ]

    saved_syms = set(basket_rows["symbol"].astype(str))
    live_syms = set(live["symbol"].astype(str))
    added = sorted(live_syms - saved_syms)
    dropped = sorted(saved_syms - live_syms)

    print(f"\n--- Basket change vs your last snapshot ---")
    if header:
        gen = header.get("generated_at", "")
        if gen:
            print(f"  Snapshot from: {gen}")
        as_of = header.get("basket_as_of", "")
        if as_of:
            print(f"  Saved basket_as_of: {as_of}")

    if added:
        show = live[live["symbol"].isin(added)].sort_values("idx_weight", ascending=False)
        print(f"\n  New in live benchmark ({len(added)}):")
        for _, row in show.head(12).iterrows():
            w = float(row["idx_weight"]) / live_total * 100.0
            print(f"    {row['symbol']}: {w:.2f}%")
        if len(added) > 12:
            print(f"    … and {len(added) - 12} more")

    if dropped:
        print(f"\n  No longer in live benchmark ({len(dropped)}):")
        for sym in dropped[:12]:
            sw = basket_rows.loc[basket_rows["symbol"] == sym, "saved_weight"]
            txt = f" (was {float(sw.iloc[0]):.2f}%)" if len(sw) else ""
            print(f"    {sym}{txt}")
        if len(dropped) > 12:
            print(f"    … and {len(dropped) - 12} more")

    drift_rows: list[dict] = []
    for sym in sorted(saved_syms & live_syms):
        saved = basket_rows.loc[basket_rows["symbol"] == sym, "saved_weight"]
        if saved.empty or pd.isna(saved.iloc[0]):
            continue
        old_w = float(saved.iloc[0])
        new_w = float(live_pct.get(sym, 0.0))
        delta = new_w - old_w
        if abs(delta) >= BASKET_WEIGHT_DRIFT_PP:
            drift_rows.append(
                {
                    "symbol": sym,
                    "saved_pct": old_w,
                    "live_pct": new_w,
                    "change_pp": delta,
                }
            )

    if drift_rows:
        drift_rows.sort(key=lambda r: abs(r["change_pp"]), reverse=True)
        print(
            f"\n  Weight shifts ≥ {BASKET_WEIGHT_DRIFT_PP:g} pp "
            f"since snapshot ({len(drift_rows)} names):"
        )
        df = pd.DataFrame(drift_rows[:10])
        df["saved_pct"] = df["saved_pct"].map(lambda v: f"{v:.2f}%")
        df["live_pct"] = df["live_pct"].map(lambda v: f"{v:.2f}%")
        df["change_pp"] = df["change_pp"].map(lambda v: f"{v:+.2f}")
        print(df.to_string(index=False))
    elif not added and not dropped:
        print("  No material basket membership or weight drift detected.")


def choose_topup_strategy() -> str:
    print("\nHow should this top-up allocate new cash?\n")
    print(
        "  1. Held names only — live weights among symbols you already own "
        "(default)"
    )
    print("  2. Equal split — same PKR budget to each held name")
    print(
        "  3. Benchmark gap-fill — move toward full benchmark weights "
        "(may buy names you do not hold yet)"
    )
    print(
        "  4. Dividend tilt — keep all names; blend basket + yield weights; "
        "optional rotate 2–3 high-yield basket lines"
    )
    print()
    while True:
        choice = prompt("Select [1]: ") or "1"
        if choice in ("1", "held", "weights", "default"):
            return "held_weights"
        if choice in ("2", "equal", "split"):
            return "equal_held"
        if choice in ("3", "benchmark", "gap", "rebalance", "full"):
            return "benchmark_gap"
        if choice in ("4", "dividend", "div", "yield", "income"):
            return "dividend_tilt"
        print("  Enter 1, 2, 3, or 4.")


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
    alerts: list[dict] = []
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
            entry = row.to_dict()
            entry["alert"] = "; ".join(reasons)
            alerts.append(entry)
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


def _peek_holdings_file(
    path: str,
) -> tuple[dict[str, int], dict[str, str] | None, pd.DataFrame | None] | None:
    """Load shares from a CSV without exiting the program on failure."""
    try:
        if _is_holdings_snapshot_file(path):
            header, snap = load_holdings_snapshot(path)
            prior = prior_shares_from_holdings_df(snap)
            if not prior:
                return None
            return prior, header, snap
        prior = load_prior_shares_from_plan_csv(path)
        if not prior:
            return None
        return prior, None, None
    except (OSError, ValueError, pd.errors.EmptyDataError):
        return None


def preview_saved_holdings_vs_basket(
    live_df: pd.DataFrame,
    source_type: str,
    source_id: str,
) -> None:
    """If a prior holdings file exists, show alignment before new vs top-up."""
    if live_df.empty or not source_id:
        return
    path = resolve_default_holdings_snapshot_path(source_id)
    if not os.path.isfile(path):
        legacy = f"psx_sip_holdings_{source_id.lower()}.csv"
        if os.path.isfile(legacy):
            path = legacy
        else:
            state = load_user_state()
            last = state.get("last_holdings_path", "")
            if last and os.path.isfile(last):
                path = last
            else:
                return
    peek = _peek_holdings_file(path)
    if peek is None:
        return
    prior, header, snap = peek
    print(
        f"\n{'=' * 72}\n"
        f"  Saved holdings vs current basket — {os.path.basename(path)}\n"
        f"  (Pick 1 = fresh monthly plan · 2 = top-up from these holdings)\n"
        f"{'=' * 72}"
    )
    report_holdings_alignment(
        prior, live_df, source_type, source_id, compact=True
    )
    if snap is not None and not snap.empty:
        report_basket_drift(snap, live_df, header)
    print_switch_considerations({s for s, q in prior.items() if q > 0}, live_df)


def choose_sip_mode(
    source_id: str = "",
    *,
    sip_profile: str = "standard",
    live_df: pd.DataFrame | None = None,
    source_type: str = "",
) -> str:
    if live_df is not None and not live_df.empty and source_id:
        preview_saved_holdings_vs_basket(live_df, source_type, source_id)
    if sip_profile == "dividend_basket":
        print("\n  Dividend basket is set — next: PKR amount → plan CSV.\n")
        print("  1. Invest in this basket (all names unless you skip below)")
        print(
            "  2. Top-up existing holdings "
            "(holdings CSV; PSX_SIP_HOLDINGS skips this menu)"
        )
    else:
        print("\nHow should this SIP run?\n")
        print("  1. New plan (scope & exclusions)")
        print(
            "  2. Top-up existing holdings "
            "(snapshot CSV, old plan CSV, or manual entry)"
        )
    print("  h. Help (shortcuts)")
    default_snap = (
        resolve_default_holdings_snapshot_path(source_id) if source_id else ""
    )
    if default_snap and os.path.isfile(default_snap):
        print(
            f"\n  → Holdings on disk: {default_snap} — option 2 uses this (or pick another)."
        )
    state = load_user_state()
    last = state.get("last_holdings_path", "")
    if last and os.path.isfile(last) and os.path.abspath(last) != os.path.abspath(
        default_snap or ""
    ):
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


def prompt_existing_csv_path(source_id: str = "") -> str:
    discovered = discover_psx_sip_csv_files(
        source_id=source_id,
        include_plans=True,
        include_holdings=True,
        max_results=25,
        prompt_extra_folder=True,
    )
    if discovered:
        roots = _csv_search_directories()
        where = roots[0] if len(roots) == 1 else ", ".join(roots)
        print(f"\n  Found PSX SIP CSV file(s) in: {where}")
        while True:
            print("\n  1. Type a different CSV path")
            for i, (label, path) in enumerate(discovered, start=2):
                print(f"  {i}. {label}")
                print(f"     {path}")
            print("  h. Help (shortcuts)\n")
            default_pick = "2"
            raw = prompt(f"Select [{default_pick}]: ").strip() or default_pick
            if _is_help_choice(raw):
                print_usage_tips()
                continue
            if raw == "1":
                break
            if raw.isdigit():
                pick = int(raw)
                if 2 <= pick < 2 + len(discovered):
                    return discovered[pick - 2][1]
            print(f"  Enter 1–{1 + len(discovered)} or h.")
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
    prior = prior_shares_from_holdings_df(df)
    if not prior:
        print("  No rows with share count > 0 in that CSV.")
        sys.exit(1)
    if "shares" in df.columns and "shares_held" not in df.columns:
        print(
            "\n  Note: using 'shares' as your position — often one month's buys only; "
            "edit or save a v2 holdings snapshot for cumulative totals."
        )
    return prior


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
) -> tuple[dict[str, int], dict[str, str] | None, pd.DataFrame | None]:
    kind = os.environ.get("PSX_SIP_HOLDINGS_KIND", "auto").strip().lower()
    use_snapshot = kind in ("snapshot", "holdings", "h")
    use_plan = kind in ("plan", "p")
    if kind == "auto" or kind in ("", "a"):
        use_snapshot = _is_holdings_snapshot_file(path)
        use_plan = not use_snapshot
    if use_snapshot:
        header, snap = load_holdings_snapshot(path)
        _warn_if_snapshot_mismatch(header, source_type, source_id)
        prior = prior_shares_from_holdings_df(snap)
        if not prior:
            print("  No shares_held > 0 in the holdings file.")
            sys.exit(1)
        return prior, header, snap
    if use_plan:
        return load_prior_shares_from_plan_csv(path), None, None
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


def _discover_holdings_file_options(
    default_snapshot: str,
    source_id: str = "",
) -> list[tuple[str, str]]:
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
    if default_snapshot and os.path.isfile(default_snapshot):
        add(
            f"Suggested for this basket ({os.path.basename(default_snapshot)})",
            default_snapshot,
        )
    for label, path in discover_psx_sip_csv_files(
        source_id=source_id,
        include_plans=True,
        include_holdings=True,
        max_results=30,
        prompt_extra_folder=True,
    ):
        add(label, path)
    return options


def choose_topup_holdings_source(
    default_snapshot: str,
    source_type: str,
    source_id: str,
) -> tuple[dict[str, int], dict[str, str] | None, pd.DataFrame | None]:
    env_path = _env_holdings_csv_path()
    if env_path:
        print(f"\nUsing PSX_SIP_HOLDINGS: {env_path}")
        return _load_prior_shares_from_any_csv(env_path, source_type, source_id)

    discovered = _discover_holdings_file_options(default_snapshot, source_id)

    while True:
        print("\nLoad holdings for top-up:\n")
        if discovered:
            roots = _csv_search_directories()
            where = roots[0] if len(roots) == 1 else ", ".join(roots)
            print(f"  Found {len(discovered)} CSV(s) — search folder(s): {where}\n")
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

        default_pick = "2" if discovered else "1"
        raw = prompt(f"Select [{default_pick}]: ") or default_pick
        if _is_help_choice(raw):
            print_usage_tips()
            continue
        if not raw.isdigit():
            print(f"  Enter 1–{manual_idx} or h.")
            continue
        pick = int(raw)

        if pick == 1:
            path = prompt_existing_csv_path(source_id)
            print(f"\nUsing: {path}")
            return _load_prior_shares_from_any_csv(path, source_type, source_id)

        if 2 <= pick < manual_idx:
            path = menu_paths[pick - 2][1]
            print(f"\nUsing: {path}")
            return _load_prior_shares_from_any_csv(path, source_type, source_id)

        if pick == manual_idx:
            return prompt_manual_holdings(), None, None

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
    *,
    topup_strategy: str = "held_weights",
    snapshot_df: pd.DataFrame | None = None,
    snapshot_header: dict[str, str] | None = None,
    show_basket_drift: bool = True,
    show_switch_hints: bool = True,
) -> tuple[pd.DataFrame, dict[str, int]]:
    held_symbols = {s for s, q in prior_shares.items() if q > 0}
    if not held_symbols:
        print("  No holdings with share count > 0.")
        sys.exit(1)

    if show_basket_drift and snapshot_df is not None:
        report_basket_drift(snapshot_df, live_df, snapshot_header)

    if topup_strategy == "benchmark_gap":
        print(
            f"\nTop-up: benchmark gap-fill on full {source_type} {source_id} "
            f"({len(live_df)} names)."
        )
        if show_switch_hints:
            print_switch_considerations(held_symbols, live_df)
        return live_df.reset_index(drop=True), prior_shares

    live_syms = live_df["symbol"].astype(str).str.upper()
    subset = live_df[live_syms.isin(held_symbols)].copy()
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

    if topup_strategy == "equal_held":
        mode_label = "equal PKR per held name"
    elif topup_strategy == "dividend_tilt":
        mode_label = (
            "held names — basket + dividend blend after optional exclusions"
        )
    else:
        mode_label = "live weights among held names only"
    print(
        f"\nTop-up mode: {len(subset)} held name(s) — {mode_label} "
        f"({source_type} {source_id})."
    )
    show = subset[["symbol", "name", "idx_weight"]].copy()
    show["idx_weight"] = show["idx_weight"].map(lambda w: f"{w:.2f}%")
    print(show.to_string(index=False))
    if show_switch_hints:
        print_switch_considerations(held_symbols, live_df)
    return subset.reset_index(drop=True), prior_shares


def run_topup_flow(
    live_df: pd.DataFrame,
    source_type: str,
    source_id: str,
    default_snapshot: str,
    *,
    sip_profile: str = "standard",
) -> tuple[pd.DataFrame, dict[str, int], str]:
    prior, snap_header, snap_df = choose_topup_holdings_source(
        default_snapshot, source_type, source_id
    )
    print(f"\n{'=' * 72}")
    print("  Top-up: your holdings vs current basket")
    print(f"{'=' * 72}")
    report_holdings_alignment(prior, live_df, source_type, source_id)
    if snap_df is not None and not snap_df.empty:
        report_basket_drift(snap_df, live_df, snap_header)
    print_switch_considerations(
        {s for s, q in prior.items() if q > 0}, live_df
    )
    skip_syms, add_syms, apply_skip, apply_add = report_topup_new_money_guide(
        prior, live_df, source_type, source_id
    )
    topup_strategy = choose_topup_strategy()
    scoped, prior = run_topup_from_prior_shares(
        prior,
        live_df,
        source_type,
        source_id,
        topup_strategy=topup_strategy,
        snapshot_df=snap_df,
        snapshot_header=snap_header,
        show_basket_drift=False,
        show_switch_hints=False,
    )
    scoped = apply_new_money_hint_selection(
        scoped,
        live_df,
        skip_syms,
        add_syms,
        apply_skip,
        apply_add,
    )
    label = "ETF basket" if source_type == "etf" else universe_label_from_source(source_type, source_id)
    scoped = apply_topup_allocation_tweaks(
        scoped,
        live_df=live_df,
        universe_label=label,
        topup_strategy=topup_strategy,
        sip_profile=sip_profile,
    )
    return scoped, prior, topup_strategy


def universe_label_from_source(source_type: str, source_id: str) -> str:
    if source_type == "index":
        return "index"
    if source_type == "etf":
        return "ETF basket"
    if source_type == "custom":
        return "basket"
    return source_id or "basket"


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
    topup_strategy: str = "held_weights",
) -> None:
    cost_buffer_pct = prompt_non_negative_float(
        "Extra cost buffer on buy price (%)",
        DEFAULT_COST_BUFFER_PCT,
    )

    lot_answer = prompt(
        f"\nRound down to PSX board lots ({BOARD_LOT} shares)? [y/N]: "
    ).lower()
    use_board_lot = lot_answer in ("y", "yes")

    basket_df = live_df if live_df is not None else scoped
    if (
        prior_shares
        and topup_strategy == "benchmark_gap"
        and live_df is not None
    ):
        plan = compute_benchmark_gap_topup_plan(
            live_df,
            prior_shares,
            investment_pkr,
            use_board_lot,
            cost_buffer_pct,
        )
        print(
            "\n  Top-up math: sized vs full benchmark after adding this deposit "
            "(may include new names)."
        )
    elif prior_shares and topup_strategy == "equal_held":
        plan = compute_equal_held_topup_plan(
            scoped, investment_pkr, use_board_lot, cost_buffer_pct
        )
    else:
        plan = compute_sip_plan(scoped, investment_pkr, use_board_lot, cost_buffer_pct)
    print_plan(plan, investment_pkr, use_board_lot, cost_buffer_pct)
    maybe_save_csv(plan, default_name=default_csv)
    holdings_default = (
        export_holdings_csv_name(source_id) if source_id else export_holdings_csv_name("holdings")
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
    """Shared Index / ETF / dividend SIP path (new plan, top-up, snapshot)."""
    sip_profile = str(meta.get("sip_profile") or "standard")
    mode = choose_sip_mode(
        source_id,
        sip_profile=sip_profile,
        live_df=df,
        source_type=source_type,
    )
    prior_shares: dict[str, int] | None = None
    topup_strategy = "held_weights"
    scoped: pd.DataFrame
    if mode == "topup":
        default_path = resolve_default_holdings_snapshot_path(source_id)
        scoped, prior_shares, topup_strategy = run_topup_flow(
            df, source_type, source_id, default_path, sip_profile=sip_profile
        )
        if scoped.empty:
            print("  No symbols left for this top-up after adjustments.")
            return
    elif sip_profile == "dividend_basket":
        print(
            f"\n  Using your confirmed basket ({len(df)} names, target weights)."
        )
        valid = set(df["symbol"].astype(str))
        symbol_raw = prompt(
            "Skip any symbol this month? (comma-sep, blank = invest in all): "
        )
        exclude_symbols = parse_symbol_list(symbol_raw, valid)
        scoped = apply_exclusions(df, set(), exclude_symbols)
        if scoped.empty:
            print(empty_exclusion_msg)
            return
        if exclude_symbols:
            print(f"  Investing in {len(scoped)} of {len(df)} names this run.")
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
        default_csv=export_plan_csv_name(source_id),
        source_type=source_type,
        source_id=source_id,
        live_df=df,
        meta=meta,
        prior_shares=prior_shares,
        topup_strategy=topup_strategy,
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
    print(f"PSX SIP Planner v{__version__} (index, ETF, scanner — psxdata)")
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
        elif product == "scanner":
            run_stock_scanner_flow()
        elif product == "dividend":
            run_dividend_portfolio_flow()
        else:
            run_index_flow()

        if not prompt_return_to_menu():
            print("\nDone.")
            break


if __name__ == "__main__":
    main()
