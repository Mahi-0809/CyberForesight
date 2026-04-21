"""Collect Number-of-Mentions (NoM_A / NoM_P) per term per month.

Collection hierarchy:
  1. Elsevier Scopus API  (primary — needs API key from dev.elsevier.com)
  2. Semantic Scholar API  (backup — free, no key needed)
  3. Synthetic fallback    (if both fail or use_live_apis is False)

The output DataFrame has columns [month, term1, term2, ...]
identical to what the original synthetic generator produced, so
downstream renaming to NoM_A_{threat} / NoM_P_{pat} is unaffected.
"""
from __future__ import annotations

import logging
import time
import warnings
from typing import Optional

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)


class ElsevierCollector:
    """Collects academic-mention counts via Scopus or Semantic Scholar.

    Parameters
    ----------
    api_key : str
        Elsevier / Scopus API key (from dev.elsevier.com).
    use_live_api : bool
        If True, attempt real API calls before falling back to synthetic.
    config : dict
        Full pipeline config dict (for rate limits, cache, backup flags).
    """

    def __init__(
        self,
        api_key: str = "",
        use_live_api: bool = False,
        config: dict | None = None,
    ):
        self.api_key = api_key
        self.use_live_api = use_live_api
        self.config = config or {}
        self._cache = None
        if self.config:
            from src.data_collection.cache_manager import CacheManager
            cache_cfg = self.config.get("cache", {})
            self._cache = CacheManager(
                cache_dir=cache_cfg.get("path", "data/raw/"),
                enabled=cache_cfg.get("enabled", True),
                refresh=cache_cfg.get("refresh", False),
            )

    # ------------------------------------------------------------------
    # Public API  (contract: returns DataFrame with [month, term1, ...])
    # ------------------------------------------------------------------

    def monthly_mentions(
        self,
        terms: list[str],
        start_date: str,
        end_date: str,
        seed: int = 42,
    ) -> pd.DataFrame:
        """Get per-month publication counts for each term in *terms*."""
        if not self.use_live_api:
            return self._synthetic_mentions(terms, start_date, end_date, seed)

        apis_cfg = self.config.get("apis", {})
        months = pd.date_range(start_date, end_date, freq="MS")
        result = pd.DataFrame({"month": months})

        # Determine label for logging (attack types vs PATs)
        label = "threats" if len(terms) <= 30 else "PATs"

        # ---- Try Elsevier Scopus (primary) ----
        elsevier_cfg = apis_cfg.get("elsevier", {})
        if elsevier_cfg.get("enabled", True) and self.api_key:
            print(f"  [API] Collecting {label} mentions via Elsevier Scopus API...")
            sleep_sec = elsevier_cfg.get("rate_limit_sleep", 0.2)
            all_ok = True

            for i, term in enumerate(terms):
                counts = self._scopus_monthly_counts(
                    term, start_date, end_date, months, sleep_sec
                )
                if counts is not None:
                    result[term] = counts.values
                    status = "cached" if self._was_cached else "API"
                    print(f"    [{i+1}/{len(terms)}] '{term}' -- {status}")
                else:
                    all_ok = False
                    break  # Quota likely exceeded; fall through to backup

            if all_ok:
                print(f"  [OK] Elsevier: collected mentions for {len(terms)} {label}")
                return result

            # Remove partially collected columns
            result = pd.DataFrame({"month": months})
            print(f"  [WARNING] Elsevier API failed partway -- trying Semantic Scholar")

        # ---- Try Semantic Scholar (backup) ----
        ss_cfg = apis_cfg.get("semantic_scholar", {})
        if ss_cfg.get("enabled", True):
            print(f"  [API] Collecting {label} mentions via Semantic Scholar...")
            sleep_sec = ss_cfg.get("rate_limit_sleep", 3.0)
            all_ok = True

            for i, term in enumerate(terms):
                counts = self._semantic_scholar_monthly_counts(
                    term, start_date, end_date, months, sleep_sec
                )
                if counts is not None:
                    result[term] = counts.values
                    status = "cached" if self._was_cached else "API"
                    print(f"    [{i+1}/{len(terms)}] '{term}' -- {status}")
                else:
                    # For S2, a single term failure isn't fatal; use zeros
                    result[term] = 0
                    print(f"    [{i+1}/{len(terms)}] '{term}' -- failed, using 0s")

            if all_ok:
                print(f"  [OK] Semantic Scholar: collected mentions for {len(terms)} {label}")
            return result

        # ---- Fallback ----
        print(f"  [WARNING] All mention APIs failed -- using synthetic {label} data")
        return self._synthetic_mentions(terms, start_date, end_date, seed)

    # ------------------------------------------------------------------
    # Elsevier Scopus API
    # ------------------------------------------------------------------

    _was_cached: bool = False  # set by each call for logging

    def _scopus_monthly_counts(
        self,
        term: str,
        start_date: str,
        end_date: str,
        months: pd.DatetimeIndex,
        sleep_sec: float,
    ) -> Optional[pd.Series]:
        """Query Scopus for publication counts of *term* per month."""
        cache_key = f"elsevier/{term}_{start_date}_{end_date}"
        if self._cache and self._cache.exists(cache_key):
            self._was_cached = True
            return self._cache.load(cache_key)
        self._was_cached = False

        base_url = "https://api.elsevier.com/content/search/scopus"
        headers = {
            "X-ELS-APIKey": self.api_key,
            "Accept": "application/json",
        }

        start_year = pd.Timestamp(start_date).year
        end_year = pd.Timestamp(end_date).year

        # Build a single Scopus query and use date facets
        query = (
            f'TITLE-ABS-KEY("{term}") '
            f"AND PUBYEAR > {start_year - 1} "
            f"AND PUBYEAR < {end_year + 1}"
        )

        # Collect per-month counts by querying year-by-year
        month_counts: dict[str, int] = {}

        for year in range(start_year, end_year + 1):
            for month_num in range(1, 13):
                month_key = f"{year}-{month_num:02d}"
                ts = pd.Timestamp(f"{year}-{month_num:02d}-01")
                if ts < pd.Timestamp(start_date) or ts > pd.Timestamp(end_date):
                    continue

                # Scopus date filter: PUBDATETXT(month year)
                month_query = (
                    f'TITLE-ABS-KEY("{term}") '
                    f"AND PUBDATETXT({ts.strftime('%B')} {year})"
                )
                params = {
                    "query": month_query,
                    "count": 0,  # We only need totalResults
                }

                try:
                    resp = requests.get(
                        base_url, headers=headers, params=params, timeout=20
                    )
                    if resp.status_code == 429:
                        # Quota exceeded
                        logger.warning("Scopus rate limit hit for '%s'", term)
                        return None
                    resp.raise_for_status()
                    data = resp.json()
                    total = int(
                        data.get("search-results", {})
                        .get("opensearch:totalResults", 0)
                    )
                    month_counts[month_key] = total
                except (requests.RequestException, ValueError, KeyError) as exc:
                    logger.warning("Scopus query failed for '%s' %s: %s", term, month_key, exc)
                    month_counts[month_key] = 0

                time.sleep(sleep_sec)

        # Build a Series aligned to the expected months index
        series = pd.Series(
            [month_counts.get(m.strftime("%Y-%m"), 0) for m in months],
            index=months,
            dtype=int,
        )

        if self._cache:
            self._cache.save(cache_key, series)

        return series

    # ------------------------------------------------------------------
    # Semantic Scholar API  (backup)
    # ------------------------------------------------------------------

    def _semantic_scholar_monthly_counts(
        self,
        term: str,
        start_date: str,
        end_date: str,
        months: pd.DatetimeIndex,
        sleep_sec: float,
    ) -> Optional[pd.Series]:
        """Query Semantic Scholar for papers matching *term*, count by month."""
        cache_key = f"semantic_scholar/{term}_{start_date}_{end_date}"
        if self._cache and self._cache.exists(cache_key):
            self._was_cached = True
            return self._cache.load(cache_key)
        self._was_cached = False

        try:
            from semanticscholar import SemanticScholar
            sch = SemanticScholar()

            results = sch.search_paper(
                term,
                fields_of_study=["Computer Science"],
                limit=100,
            )

            month_counts: dict[str, int] = {
                m.strftime("%Y-%m"): 0 for m in months
            }

            if results and hasattr(results, '__iter__'):
                for paper in results:
                    pub_date = getattr(paper, "publicationDate", None)
                    if pub_date is None:
                        continue
                    if isinstance(pub_date, str):
                        try:
                            pub_date = pd.Timestamp(pub_date)
                        except Exception:
                            continue
                    key = f"{pub_date.year}-{pub_date.month:02d}"
                    if key in month_counts:
                        month_counts[key] += 1

            series = pd.Series(
                [month_counts.get(m.strftime("%Y-%m"), 0) for m in months],
                index=months,
                dtype=int,
            )

            if self._cache:
                self._cache.save(cache_key, series)

            time.sleep(sleep_sec)
            return series

        except Exception as exc:
            logger.warning("Semantic Scholar query failed for '%s': %s", term, exc)
            time.sleep(sleep_sec)
            return None

    # ------------------------------------------------------------------
    # Synthetic fallback  (original mock — UNCHANGED logic)
    # ------------------------------------------------------------------

    def _synthetic_mentions(
        self,
        terms: list[str],
        start_date: str,
        end_date: str,
        seed: int = 42,
    ) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        months = pd.date_range(start_date, end_date, freq="MS")
        n = len(months)
        data: dict = {"month": months}
        trend = np.linspace(20, 120, n)
        season = 8 * np.sin(np.linspace(0, 8 * np.pi, n))
        for term in terms:
            noise = rng.normal(0, 7, n)
            shift = rng.integers(-10, 20)
            vals = np.maximum(0, trend + season + noise + shift).round().astype(int)
            data[term] = vals
        return pd.DataFrame(data)
