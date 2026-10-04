# PSX SIP Planner

Interactive **Pakistan Stock Exchange (PSX)** tool:

1. **Index SIP** — allocate a monthly amount across index constituents (weights, exclusions, board lots).
2. **ETF SIP** — same for an ETF’s published creation-unit basket (with sector preview and ETF unit price).
3. **Insights** — index/ETF levels, week/month/year highs, and sector leaders (no SIP plan).

Data comes from [psxdata](https://pypi.org/project/psxdata/) (PSX Data Portal). Cache lives under `~/.psxdata/cache/` by default.

**Author:** Salim Ali Khan · **Version:** 1.0.0 · **Date:** October 2026

## Requirements

- Python **3.10+**
- Network access to `dps.psx.com.pk`

## Run from a zip (any machine)

```bash
unzip psx-sip-portable.zip -d psx-sip && cd psx-sip
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python psx_sip_planner.py
```

Or without a venv (uses your user site-packages after first-time bootstrap):

```bash
python3 psx_sip_planner.py
```

Skip auto-install (fail if deps missing):

```bash
PSX_SIP_NO_PIP=1 python3 psx_sip_planner.py
```

## Run from GitHub (curl pipe)

Replace `YOUR_USER` / `YOUR_REPO` / `BRANCH` with your repo after you push.

**One-liner** (downloads script, installs deps if needed, runs interactively):

```bash
curl -fsSL "https://raw.githubusercontent.com/YOUR_USER/YOUR_REPO/BRANCH/psx_sip_planner.py" | python3 -
```

**Save locally then run** (recommended if you use it often):

```bash
curl -fsSL -o psx_sip_planner.py \
  "https://raw.githubusercontent.com/YOUR_USER/YOUR_REPO/BRANCH/psx_sip_planner.py"
python3 psx_sip_planner.py
```

The script only runs `pip install` when `pandas` or `psxdata` are missing. It does not modify your project files—only optional packages and PSX’s cache directory.

## Push to GitHub (one-time)

From this folder:

```bash
git init
git add psx_sip_planner.py requirements.txt README.md .gitignore
git commit -m "Add PSX index/ETF SIP planner and insights"
gh repo create psx-sip --public --source=. --push
```

Then update the curl URL in this README to match your repo.

## Files

| File | Purpose |
|------|---------|
| `psx_sip_planner.py` | Single entry point (SIP + insights) |
| `requirements.txt` | Pinned minimum deps for venv installs |

## Environment variables

| Variable | Effect |
|----------|--------|
| `PSX_SIP_SKIP_BOOTSTRAP` | Do not attempt `pip install` |
| `PSX_SIP_NO_PIP` | Exit with instructions if deps are missing |
# psx-sip
