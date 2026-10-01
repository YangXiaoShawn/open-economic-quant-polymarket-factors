# Polymarket Arb Scanner

Real-time scanner and factor research platform for Polymarket prediction markets.

> 📄 **阶段性研究报告**：[REPORT.md](REPORT.md) — 原理 → 方法论（五类假象修正）→ 因子结论（MOM/VOL/LIQ 为真实因子）→ 组合佐证（套利 50% + 做空赢家 50%，Sharpe 6.3），插图在 `report_figs/`。

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure secrets
cp .env.example .env
# Edit .env and fill in POLYMARKET_API_KEY (and CLOB credentials if needed)

# 3. Launch the server
start.bat          # Windows
# or:
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

## Pages

| URL | Description |
|-----|-------------|
| `http://localhost:8000/` | Live scanner — pair signals + short-winners TOP5 strip |
| `http://localhost:8000/backtest` | Walk-forward backtest — portfolio arb 50% + short-winners 50% (local data) |
| `http://localhost:8000/factors` | Factor research — expanded-study verdicts + long-term LS curves + Fama-MacBeth |
| `http://localhost:8000/threefactor` | Three-factor model — optimal structure MOM+LIQ+TTR (35-combo search) |
| `http://localhost:8000/shortwinners` | Short-winners live signals — auto-rescan every 10 min, WS push |

## Cache Management

All cached data lives in `cache/`. Individual files expire automatically (4h–24h TTL), but you can force a fresh fetch:

- **API**: `GET /api/factors?force=true` or `GET /api/backtest?force=true`
- **Manual**: delete `cache/*.pkl` and restart the server

## Required Environment Variables (`.env`)

```
POLYMARKET_GAMMA_API=https://gamma-api.polymarket.com
POLYMARKET_CLOB_API=https://clob.polymarket.com
```

## Running Tests

```bash
python -m pytest tests/ -v
```
