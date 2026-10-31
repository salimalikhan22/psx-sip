# PSX SIP Planner

Interactive **Pakistan Stock Exchange (PSX)** tool for index/ETF/dividend SIP planning and market insights.

**Main menu:** **Invest** (index, ETF, dividend basket SIP) · **Insights** (overview, compare holdings, stock scanner).

Data comes from [psxdata](https://pypi.org/project/psxdata/) (PSX Data Portal).

**Author:** Salim Ali Khan · **Version:** 1.4.5 · **Date:** October 2026

## Requirements

- Python **3.10+**
- Network access to `dps.psx.com.pk`

## Run with curl (any machine, no clone)

From the folder where you keep CSV exports (`cd` there first):

```bash
curl -fsSL "https://raw.githubusercontent.com/salimalikhan22/psx-sip/main/psx_sip_planner.py" | python3 -
```

Shorthand (same as above; `-s` silent, `-f` fail on HTTP errors):

```bash
curl -sSf "https://raw.githubusercontent.com/salimalikhan22/psx-sip/main/psx_sip_planner.py" | python3 -
```

- Prompts use your **terminal** (`/dev/tty`) even though the script comes from a pipe.
- **Index / ETF / dividend screened SIP / scanner** work without a git checkout.
- **Mutual-fund reference** (menu 4 / holdings view): on first use, downloads  
  `dividend_fund_composition.json` from GitHub into **`~/.psx-sip/`** (needs network once).
- Plan/holdings CSVs are read/written in your **current working directory** only.

Override the download base URL (fork or branch):

```bash
export PSX_SIP_REPO_RAW="https://raw.githubusercontent.com/YOUR_USER/psx-sip/main"
```

Force re-download fund JSON: `PSX_SIP_REFRESH_FUNDS=1`

## Run from git clone (offline-friendly fund data)

Clone this repository, create a venv, install deps, run from the folder where you keep your CSV exports:

```bash
git clone https://github.com/salimalikhan22/psx-sip.git
cd psx-sip
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cd ~/my-sip-exports   # optional: folder for plan/holdings CSVs
../psx-sip/.venv/bin/python ../psx-sip/psx_sip_planner.py
```

Or run inside the clone (CSVs saved in `psx-sip/`):

```bash
cd psx-sip
.venv/bin/python psx_sip_planner.py
```

Without a venv (auto `pip install` on first run if deps missing):

```bash
python3 psx_sip_planner.py
```

Skip auto-install:

```bash
PSX_SIP_NO_PIP=1 python3 psx_sip_planner.py
```

## What is in this repo (runtime)

These files are **required** for the full planner (including dividend fund reference and PSXDIV20 menus):

| Path | Purpose |
|------|---------|
| `psx_sip_planner.py` | Main program |
| `dividend_funds_data.py` | Mutual fund reference loader |
| `data/dividend_fund_composition.json` | Saved FMR top-10 snapshots (not live NAV) |
| `requirements.txt` | `psxdata`, `pandas` |
| `.gitignore` | Excludes local `*.csv` / `*.xlsx` exports |

**Not in git:** your plan/holdings CSVs, Excel workbooks, `.venv`, and optional `~/.psx-sip/user_state.json` (last file path only).

Local helper scripts such as `build_dividend_*.py` and generated spreadsheets are **optional** and intentionally not published.

## Privacy

- Source on GitHub contains **no portfolio data**.
- **No telemetry** — network use is public PSX data via `psxdata`.
- Plans/holdings are written only when you confirm save, to paths you choose.
- Default is **live PSX** each fetch; optional `PSX_SIP_USE_CACHE=1` uses `~/.psxdata/cache/`.
- Type **`h`** at any menu for shortcuts.

## Top-up and CSV discovery

- **Invest →** product → **Top-up** loads a holdings snapshot or plan CSV.
- The planner lists `psx_sip_*.csv` in your **current working directory**; if none, it asks for a **folder to search**.
- Optional: `PSX_SIP_HOLDINGS=/path/to/file.csv` skips the load menu.
- Optional: `PSX_SIP_CSV_DIRS=dir1,dir2` always searches extra folders.

| `PSX_SIP_HOLDINGS_KIND` | Effect |
|-------------------------|--------|
| `auto` (default) | Detect snapshot vs plan CSV |
| `snapshot` | Force holdings snapshot (`# psx-sip-holdings` header) |
| `plan` | Force SIP plan CSV |

## Environment variables

| Variable | Effect |
|----------|--------|
| `PSX_SIP_HOLDINGS` | Path to holdings/plan CSV for top-up |
| `PSX_SIP_HOLDINGS_KIND` | `auto`, `snapshot`, or `plan` |
| `PSX_SIP_CSV_DIRS` | Comma-separated folders to search for exports |
| `PSX_SIP_NO_LOCAL_STATE` | Do not use `~/.psx-sip/user_state.json` |
| `PSX_SIP_USE_CACHE` | Enable psxdata disk cache |
| `PSX_SIP_NO_CACHE` | Force live PSX |
| `PSX_SIP_PRELOAD_ETF_CATALOG` | Pre-fetch all ETF baskets before pick |
| `PSX_SIP_SKIP_BOOTSTRAP` / `PSX_SIP_NO_PIP` | Control dependency install |
| `PSX_SIP_VERBOSE` | Show psxdata scraper warnings |
| `PSX_SIP_REPO_RAW` | GitHub raw base URL for curl-mode fund JSON bootstrap |
| `PSX_SIP_REFRESH_FUNDS` | Re-download fund reference JSON into `~/.psx-sip/` |

## psxdata warnings

OHLC / duplicate-date messages come from **psxdata** while cleaning PSX index history. They are hidden by default; set `PSX_SIP_VERBOSE=1` to show them.
