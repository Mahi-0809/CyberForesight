"""Collect Armed Conflict Area (ACA) events per country per month.

Collection hierarchy:
  1. ACLED API  (primary -- needs email+password from acleddata.com)
  2. Synthetic fallback (if ACLED fails or use_live_apis is False)

The filename is kept as twitter_api.py to avoid breaking existing imports
in dataset_builder.py.  The class is renamed internally but an alias is
provided for backward compatibility.

Output format: DataFrame with columns [month, ACA_C_{country}..., ACA]
identical to what the original synthetic generator produced.
"""
from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np
import pandas as pd
import requests

from src.constants import COUNTRIES_36

logger = logging.getLogger(__name__)

# ISO 3166-1 alpha-3 to alpha-2 for ACLED country codes
_ISO3_TO_ISO2: dict[str, str] = {
    "USA": "US", "GBR": "UK", "DEU": "DE", "FRA": "FR", "ESP": "ES",
    "ITA": "IT", "NLD": "NL", "SWE": "SE", "NOR": "NO", "FIN": "FI",
    "POL": "PL", "IRL": "IE", "CHE": "CH", "AUT": "AT", "BEL": "BE",
    "PRT": "PT", "RUS": "RU", "UKR": "UA", "TUR": "TR", "IND": "IN",
    "CHN": "CN", "JPN": "JP", "KOR": "KR", "SGP": "SG", "AUS": "AU",
    "NZL": "NZ", "CAN": "CA", "MEX": "MX", "BRA": "BR", "ARG": "AR",
    "ZAF": "ZA", "EGY": "EG", "NGA": "NG", "ARE": "AE", "SAU": "SA",
    "ISR": "IL",
}

# ACLED also provides country names; map common ones to ISO-2 as fallback
_ACLED_NAME_TO_ISO2: dict[str, str] = {
    "United States": "US", "United Kingdom": "UK", "Germany": "DE",
    "France": "FR", "Spain": "ES", "Italy": "IT", "Netherlands": "NL",
    "Sweden": "SE", "Norway": "NO", "Finland": "FI", "Poland": "PL",
    "Ireland": "IE", "Switzerland": "CH", "Austria": "AT", "Belgium": "BE",
    "Portugal": "PT", "Russia": "RU", "Ukraine": "UA", "Turkey": "TR",
    "India": "IN", "China": "CN", "Japan": "JP", "South Korea": "KR",
    "Singapore": "SG", "Australia": "AU", "New Zealand": "NZ",
    "Canada": "CA", "Mexico": "MX", "Brazil": "BR", "Argentina": "AR",
    "South Africa": "ZA", "Egypt": "EG", "Nigeria": "NG",
    "United Arab Emirates": "AE", "Saudi Arabia": "SA", "Israel": "IL",
}


class ConflictCollector:
    """Collects armed conflict events via ACLED API.

    Parameters
    ----------
    bearer_token : str
        Legacy parameter (was Twitter bearer). Ignored but kept for compatibility.
    use_live_api : bool
        If True, attempt real API calls before falling back to synthetic.
    config : dict
        Full pipeline config dict (for API keys, rate limits, cache settings).
    """

    def __init__(
        self,
        bearer_token: str = "",
        use_live_api: bool = False,
        config: dict | None = None,
    ):
        self.bearer_token = bearer_token  # legacy, ignored
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
    # Public API  (contract: returns DataFrame [month, ACA_C_{c}..., ACA])
    # ------------------------------------------------------------------

    def monthly_conflict_counts(
        self,
        countries: list[str],
        start_date: str,
        end_date: str,
        seed: int = 42,
    ) -> pd.DataFrame:
        """Collect monthly conflict event counts per country."""
        if not self.use_live_api:
            return self._synthetic_conflicts(countries, start_date, end_date, seed)

        apis_cfg = self.config.get("apis", {})

        # --- Try ACLED (primary -- structured conflict data) ---
        acled_cfg = apis_cfg.get("acled", {})
        acled_email = self.config.get("api", {}).get("acled_email", "")
        acled_password = self.config.get("api", {}).get("acled_password", "")
        if acled_cfg.get("enabled", True) and acled_email and acled_password:
            try:
                print("  [API] Collecting conflict data via ACLED API...")
                df = self._collect_acled(countries, start_date, end_date)
                if df is not None and len(df) > 0:
                    print(f"  [OK] ACLED: collected conflict data for {len(df)} months")
                    return df
            except Exception as exc:
                print(f"  [WARNING] ACLED collection failed: {exc}")
                logger.exception("ACLED collection failed")
        else:
            if not acled_email or not acled_password:
                print("  [WARNING] ACLED credentials not configured in config.local.yaml")

        # --- Fallback ---
        print("  [WARNING] ACLED unavailable -- using synthetic ACA data")
        return self._synthetic_conflicts(countries, start_date, end_date, seed)

    # ------------------------------------------------------------------
    # ACLED API  (primary)
    # ------------------------------------------------------------------

    def _collect_acled(
        self,
        countries: list[str],
        start_date: str,
        end_date: str,
    ) -> Optional[pd.DataFrame]:
        """Query ACLED API for conflict events per country per month.

        Queries by year for efficiency (12 API calls for 2011-2022).
        ACLED API docs: https://apidocs.acleddata.com/
        """
        cache_key = f"acled/conflict_{start_date}_to_{end_date}"
        if self._cache and self._cache.exists(cache_key):
            print("  [CACHE] Loading ACLED data from cache...")
            return self._cache.load(cache_key)

        email = self.config.get("api", {}).get("acled_email", "")
        password = self.config.get("api", {}).get("acled_password", "")

        if not email or not password:
            logger.warning("ACLED email/password not configured")
            return None

        base_url = "https://api.acleddata.com/acled/read"
        months = pd.date_range(start_date, end_date, freq="MS")
        month_data: dict[str, dict[str, int]] = {}
        country_set = set(countries)

        years = sorted(set(m.year for m in months))
        consecutive_failures = 0

        for year in years:
            print(f"    ACLED: querying year {year}...", end="", flush=True)

            params = {
                "email": email,
                "password": password,
                "event_type": "Battles|Explosions/Remote violence|Violence against civilians",
                "year": year,
                "limit": 0,  # 0 = no limit, return all
            }

            try:
                resp = requests.get(base_url, params=params, timeout=60)
                resp.raise_for_status()
                data = resp.json()

                # Check for ACLED-specific error in response
                if isinstance(data, dict) and not data.get("success", True):
                    error_msg = data.get("error", {}).get("message", "Unknown error")
                    print(f" API error: {error_msg}")
                    consecutive_failures += 1
                    if consecutive_failures >= 3:
                        print("    [WARNING] 3 consecutive failures, stopping ACLED queries")
                        break
                    continue

                events = data.get("data", [])
                consecutive_failures = 0  # reset on success
                print(f" {len(events)} events")

                for event in events:
                    event_date = event.get("event_date", "")
                    try:
                        ed = pd.Timestamp(event_date)
                    except Exception:
                        continue

                    month_key = ed.strftime("%Y-%m")
                    month_ts = pd.Timestamp(f"{ed.year}-{ed.month:02d}-01")
                    if month_ts < pd.Timestamp(start_date) or month_ts > pd.Timestamp(end_date):
                        continue

                    if month_key not in month_data:
                        month_data[month_key] = {c: 0 for c in countries}

                    # Try ISO-3 code first, then country name
                    iso2 = ""
                    iso3 = str(event.get("iso3", ""))
                    if iso3:
                        iso2 = _ISO3_TO_ISO2.get(iso3, "")
                    if not iso2:
                        country_name = str(event.get("country", ""))
                        iso2 = _ACLED_NAME_TO_ISO2.get(country_name, "")

                    if iso2 in country_set:
                        month_data[month_key][iso2] += 1

            except requests.exceptions.ConnectionError as exc:
                print(f" connection error (DNS/network)")
                logger.warning("ACLED connection failed for %s: %s", year, exc)
                # DNS/network issue -- no point retrying other years
                print("    [WARNING] Network error -- cannot reach api.acleddata.com")
                print("    Try connecting via mobile hotspot or VPN")
                return None

            except Exception as exc:
                logger.warning("ACLED query failed for %s: %s", year, exc)
                print(f" error: {exc}")
                consecutive_failures += 1
                if consecutive_failures >= 3:
                    print("    [WARNING] 3 consecutive failures, stopping ACLED queries")
                    break

            time.sleep(1)

        if not month_data:
            return None

        result = self._build_output_df(months, countries, month_data)
        if result is not None and self._cache:
            self._cache.save(cache_key, result)
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_output_df(
        self,
        months: pd.DatetimeIndex,
        countries: list[str],
        month_data: dict[str, dict[str, int]],
    ) -> Optional[pd.DataFrame]:
        """Convert {month_key: {country: count}} to the expected DataFrame."""
        if not month_data:
            return None

        rows = []
        for month in months:
            key = month.strftime("%Y-%m")
            counts = month_data.get(key, {c: 0 for c in countries})
            row = {"month": month}
            for c in countries:
                row[f"ACA_C_{c}"] = counts.get(c, 0)
            rows.append(row)

        df = pd.DataFrame(rows)
        aca_cols = [col for col in df.columns if col.startswith("ACA_C_")]
        df["ACA"] = df[aca_cols].sum(axis=1)
        return df

    # ------------------------------------------------------------------
    # Synthetic fallback  (original mock -- UNCHANGED logic)
    # ------------------------------------------------------------------

    def _synthetic_conflicts(
        self,
        countries: list[str],
        start_date: str,
        end_date: str,
        seed: int = 42,
    ) -> pd.DataFrame:
        rng = np.random.default_rng(seed + 7)
        months = pd.date_range(start_date, end_date, freq="MS")
        n = len(months)
        data: dict = {"month": months}
        base = np.linspace(200, 450, n) + 35 * np.sin(np.linspace(0, 6 * np.pi, n))
        for c in countries:
            counts = np.maximum(
                0, base + rng.normal(0, 25, n) + rng.integers(-70, 70)
            ).round().astype(int)
            data[f"ACA_C_{c}"] = counts
        df = pd.DataFrame(data)
        df["ACA"] = df[[col for col in df.columns if col.startswith("ACA_C_")]].sum(axis=1)
        return df


# Backward-compatible alias
TwitterConflictCollector = ConflictCollector
