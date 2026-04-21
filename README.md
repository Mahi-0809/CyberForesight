# Cyber Threat & PAT Forecasting (B-MTGNN)

End-to-end implementation of the pipeline described in Almahmoud et al. (2025), including:

1. Data collection/preparation (real APIs with synthetic fallbacks)
2. Threats and Pertinent Technologies (TPT) graph construction
3. Bayesian MTGNN model training with MC dropout
4. 2023-2025 forecasting, gap analysis, and recommendation generation
5. Alleviation Technologies Cycle (ATC) plotting

## Quickstart

```bash
pip install -r requirements.txt
python run_pipeline.py
```

## Data Sources

The pipeline collects data from real APIs with automatic synthetic fallback:

| Data Type | Primary Source | Backup Source | Fallback |
|-----------|---------------|---------------|----------|
| **NoI** (Incidents) | NVD CVE API | Hackmageddon scraper | Synthetic |
| **NoM_A** (Threat mentions) | Elsevier Scopus API | Semantic Scholar | Synthetic |
| **NoM_P** (PAT mentions) | Elsevier Scopus API | Semantic Scholar | Synthetic |
| **ACA** (Armed conflicts) | GDELT v2 | ACLED API | Synthetic |
| **PH** (Public holidays) | `holidays` library | — | — |

## API Keys Setup

1. Copy `config.local.yaml.example` to `config.local.yaml`
2. Fill in your API keys:

```yaml
api:
  elsevier_key: "YOUR_KEY"       # from dev.elsevier.com (free)
  nvd_api_key: "YOUR_KEY"        # from nvd.nist.gov (free)
  acled_api_key: "YOUR_KEY"      # from acleddata.com (free)
  acled_email: "your@email.com"
  use_live_apis: true
```

Keys NOT needed (no auth required):
- **GDELT**: Free, no key needed
- **Semantic Scholar**: Free, no key needed

## CLI Options

```bash
python run_pipeline.py                 # Normal run (uses cache)
python run_pipeline.py --refresh-cache # Force re-download all API data
```

## Configuration

Edit `config.yaml` to control:
- `api.use_live_apis`: `true` for real APIs, `false` for synthetic
- `apis.*`: Enable/disable individual data sources and set rate limits
- `cache.refresh`: `true` to re-download cached data

## Outputs

Generated under `outputs/`:

- `b_mtgnn_best.pt`
- `forecast_2023_2025.csv`
- `gap_analysis_report.csv`
- `investment_recommendations.pdf`
- `atc_diagram.png`
- `trend_plots/*.png`

Generated under `data/`:

- `data/processed/monthly_dataset.csv`
- `data/graph/tpt_graph.json`
- `data/raw/` (cached API responses — gitignored)

## Notes

- First run with real APIs may be slow (NVD: ~30 min for 138 months). Subsequent runs load from cache instantly.
- Set `api.use_live_apis: false` in `config.yaml` for reproducible synthetic data mode.
- All collectors gracefully fall back to synthetic data if APIs fail.
