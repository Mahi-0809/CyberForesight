"""Collect Number-of-Incidents (NoI) data per attack type per month.

Collection hierarchy:
  1. NVD CVE API  (primary — reliable, structured JSON)
  2. Hackmageddon scraper (bonus — fragile, may break)
  3. Synthetic fallback (if both fail or use_live_apis is False)

The output DataFrame has columns [date, attack_type, country, description]
identical to what the original synthetic generator produced, so downstream
code (wfc.py → dataset_builder.py) is unaffected.
"""
from __future__ import annotations

import logging
import re
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import requests

from src.constants import COUNTRIES_36, THREATS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Keyword map: scan NVD CVE descriptions for these patterns to classify
# each CVE into one of our 26 attack types.  Order matters — first match wins.
# ---------------------------------------------------------------------------
_THREAT_KEYWORDS: dict[str, list[str]] = {
    "Ransomware":        ["ransomware", "ransom"],
    "Phishing":          ["phishing", "spear-phishing", "spearphishing"],
    "DDoS":              ["ddos", "denial of service", "denial-of-service"],
    "Malware":           ["malware", "malicious software", "virus", "worm"],
    "Trojan":            ["trojan", "trojanized"],
    "Botnet":            ["botnet", "bot network", "command and control", "c2 server"],
    "Backdoor":          ["backdoor", "back door", "back-door"],
    "Dropper":           ["dropper", "payload delivery"],
    "Brute Force Attack": ["brute force", "brute-force", "credential stuffing"],
    "Password Attack":   ["password", "credential", "authentication bypass"],
    "SQL Injection":     ["sql injection", "sqli"],
    "Zero-day":          ["zero-day", "zeroday", "zero day", "0-day", "0day"],
    "APT":               ["advanced persistent threat", "apt", "state-sponsored"],
    "Supply Chain Attack": ["supply chain", "supply-chain"],
    "Cryptojacking":     ["cryptojacking", "crypto mining", "cryptomining"],
    "Session Hijacking": ["session hijack", "session fixation", "cookie theft"],
    "MITM":              ["man-in-the-middle", "man in the middle", "mitm"],
    "DNS Spoofing":      ["dns spoof", "dns poison", "dns hijack", "dns cache"],
    "Account Hijacking": ["account hijack", "account takeover"],
    "Insider Threat":    ["insider threat", "insider attack", "privileged user"],
    "IoT Device Attack": ["iot", "internet of things", "smart device"],
    "Deepfake":          ["deepfake", "deep fake"],
    "Disinformation":    ["disinformation", "misinformation", "fake news"],
    "Data Poisoning":    ["data poisoning", "training data"],
    "Adversarial Attack": ["adversarial", "evasion attack"],
    "Targeted Attack":   ["targeted attack", "spear", "watering hole"],
    "Vulnerability":     ["vulnerability", "buffer overflow", "use-after-free",
                          "out-of-bounds", "heap overflow", "stack overflow",
                          "integer overflow", "race condition", "null pointer",
                          "cross-site scripting", "xss", "remote code execution",
                          "rce", "privilege escalation", "path traversal",
                          "directory traversal", "information disclosure",
                          "memory corruption"],
}


def _classify_description(desc: str) -> str:
    """Return the best-matching threat name for a CVE description string."""
    lower = desc.lower()
    for threat, keywords in _THREAT_KEYWORDS.items():
        for kw in keywords:
            if kw in lower:
                # Map "SQL Injection" etc. back to the canonical 26 THREATS
                if threat in THREATS:
                    return threat
                return "Vulnerability"  # catch-all for non-canonical
    return "Vulnerability"  # default if nothing matches


@dataclass
class HackmageddonCollector:
    """Collects cyber-incident data via NVD (primary) or Hackmageddon (bonus).

    Parameters
    ----------
    use_live_api : bool
        If True, attempt real API calls before falling back to synthetic.
    config : dict
        Full pipeline config dict (for API keys, rate limits, cache settings).
    """
    use_live_api: bool = False
    config: dict = field(default_factory=dict)

    # Cache manager is injected if available
    _cache: object = field(default=None, repr=False)

    def __post_init__(self):
        if self._cache is None and self.config:
            from src.data_collection.cache_manager import CacheManager
            cache_cfg = self.config.get("cache", {})
            self._cache = CacheManager(
                cache_dir=cache_cfg.get("path", "data/raw/"),
                enabled=cache_cfg.get("enabled", True),
                refresh=cache_cfg.get("refresh", False),
            )

    # ------------------------------------------------------------------
    # Public API  (contract: returns DataFrame with [date, attack_type,
    #              country, description])
    # ------------------------------------------------------------------

    def collect(
        self,
        start_date: str,
        end_date: str,
        approx_records: int = 18000,
    ) -> pd.DataFrame:
        """Collect incident data.  Tries NVD → Hackmageddon → Synthetic."""
        if not self.use_live_api:
            return self._mock_incidents(start_date, end_date, approx_records)

        apis_cfg = self.config.get("apis", {})

        # --- Try NVD (primary) ---
        nvd_cfg = apis_cfg.get("nvd", {})
        if nvd_cfg.get("enabled", True):
            try:
                print("  [API] Collecting incidents via NVD CVE API (primary)...")
                df = self._collect_nvd(start_date, end_date, nvd_cfg)
                if df is not None and len(df) > 0:
                    print(f"  [OK] NVD: collected {len(df)} incident records")
                    return df
            except Exception as exc:
                print(f"  [WARNING] NVD API failed: {exc}")
                logger.exception("NVD collection failed")

        # --- Try Hackmageddon (bonus) ---
        hm_cfg = apis_cfg.get("hackmageddon", {})
        if hm_cfg.get("enabled", True):
            try:
                print("  [API] Trying Hackmageddon scraper (bonus)...")
                df = self._collect_hackmageddon(start_date, end_date)
                if df is not None and len(df) > 0:
                    print(f"  [OK] Hackmageddon: collected {len(df)} incident records")
                    return df
            except Exception as exc:
                print(f"  [WARNING] Hackmageddon scraper failed: {exc}")
                logger.exception("Hackmageddon scrape failed")

        # --- Fallback to synthetic ---
        print("  [WARNING] All real incident sources failed -- using synthetic data")
        return self._mock_incidents(start_date, end_date, approx_records)

    # ------------------------------------------------------------------
    # NVD CVE API  (primary)
    # ------------------------------------------------------------------

    def _collect_nvd(
        self,
        start_date: str,
        end_date: str,
        nvd_cfg: dict,
    ) -> Optional[pd.DataFrame]:
        """Query NVD CVE 2.0 API month-by-month, classify CVEs by attack type."""
        cache_key = f"nvd/incidents_{start_date}_to_{end_date}"
        if self._cache and self._cache.exists(cache_key):
            print("  [CACHE] Loading NVD data from cache...")
            return self._cache.load(cache_key)

        api_key = self.config.get("api", {}).get("nvd_api_key", "")
        sleep_sec = nvd_cfg.get("rate_limit_sleep", 0.6)
        base_url = "https://services.nvd.nist.gov/rest/json/cves/2.0"

        months = pd.date_range(start_date, end_date, freq="MS")
        all_rows: list[dict] = []

        for i, month_start in enumerate(months):
            month_end = (month_start + pd.offsets.MonthEnd(1))
            pub_start = month_start.strftime("%Y-%m-%dT00:00:00.000")
            pub_end = month_end.strftime("%Y-%m-%dT23:59:59.999")

            params = {
                "pubStartDate": pub_start,
                "pubEndDate": pub_end,
                "resultsPerPage": 2000,
                "startIndex": 0,
            }
            headers = {}
            if api_key:
                headers["apiKey"] = api_key

            month_label = month_start.strftime("%Y-%m")
            print(f"    NVD: querying {month_label} ({i+1}/{len(months)})...", end="")

            try:
                total_fetched = 0
                start_index = 0

                while True:
                    params["startIndex"] = start_index
                    resp = requests.get(
                        base_url, params=params, headers=headers, timeout=30
                    )
                    resp.raise_for_status()
                    data = resp.json()

                    vulns = data.get("vulnerabilities", [])
                    total_results = data.get("totalResults", 0)

                    for item in vulns:
                        cve = item.get("cve", {})
                        pub_date = cve.get("published", "")[:10]
                        descs = cve.get("descriptions", [])
                        desc_en = ""
                        for d in descs:
                            if d.get("lang") == "en":
                                desc_en = d.get("value", "")
                                break
                        if not desc_en and descs:
                            desc_en = descs[0].get("value", "")

                        attack_type = _classify_description(desc_en)

                        all_rows.append({
                            "date": pub_date,
                            "attack_type": attack_type,
                            "country": None,  # NVD doesn't have country info
                            "description": desc_en[:200],
                        })

                    total_fetched += len(vulns)
                    start_index += len(vulns)

                    # Check if we've fetched all results
                    if start_index >= total_results or len(vulns) == 0:
                        break

                    time.sleep(sleep_sec)

                print(f" {total_fetched} CVEs")
                time.sleep(sleep_sec)

            except requests.exceptions.RequestException as exc:
                print(f" error: {exc}")
                logger.warning("NVD request failed for %s: %s", month_label, exc)
                time.sleep(sleep_sec * 2)
                continue

        if not all_rows:
            return None

        df = pd.DataFrame(all_rows)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

        if self._cache:
            self._cache.save(cache_key, df)

        return df

    # ------------------------------------------------------------------
    # Hackmageddon scraper  (bonus — fragile)
    # ------------------------------------------------------------------

    def _collect_hackmageddon(
        self,
        start_date: str,
        end_date: str,
    ) -> Optional[pd.DataFrame]:
        """Scrape Hackmageddon blog for monthly attack stats.

        This is a bonus/best-effort scraper.  The site structure may change
        at any time, so failures here are expected and handled gracefully.
        """
        cache_key = f"hackmageddon/incidents_{start_date}_to_{end_date}"
        if self._cache and self._cache.exists(cache_key):
            print("  [CACHE] Loading Hackmageddon data from cache...")
            return self._cache.load(cache_key)

        try:
            from bs4 import BeautifulSoup
        except ImportError:
            logger.warning("beautifulsoup4 not installed — skipping Hackmageddon")
            return None

        try:
            from thefuzz import fuzz
        except ImportError:
            logger.warning("thefuzz not installed — skipping Hackmageddon")
            return None

        archive_url = "https://www.hackmageddon.com/category/security/cyber-attacks-statistics/"
        all_rows: list[dict] = []

        try:
            resp = requests.get(archive_url, timeout=30)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "lxml")

            # Look for links to monthly timeline posts
            links = []
            for a_tag in soup.find_all("a", href=True):
                href = a_tag["href"]
                text = a_tag.get_text(strip=True).lower()
                if "timeline" in text and ("hackmageddon" in href):
                    links.append(href)

            # Paginate through archive pages
            page = 2
            while page <= 20:
                try:
                    page_url = f"{archive_url}page/{page}/"
                    resp = requests.get(page_url, timeout=20)
                    if resp.status_code != 200:
                        break
                    page_soup = BeautifulSoup(resp.text, "lxml")
                    found_any = False
                    for a_tag in page_soup.find_all("a", href=True):
                        href = a_tag["href"]
                        text = a_tag.get_text(strip=True).lower()
                        if "timeline" in text and ("hackmageddon" in href):
                            links.append(href)
                            found_any = True
                    if not found_any:
                        break
                    page += 1
                    time.sleep(1)
                except Exception:
                    break

            links = list(set(links))
            print(f"    Hackmageddon: found {len(links)} timeline post links")

            if not links:
                return None

            # Parse each timeline post for attack data
            months = pd.date_range(start_date, end_date, freq="MS")
            rng = np.random.default_rng(42)

            for link in links[:50]:  # cap to avoid infinite scraping
                try:
                    resp = requests.get(link, timeout=20)
                    resp.raise_for_status()
                    post_soup = BeautifulSoup(resp.text, "lxml")

                    # Extract date from URL or content
                    date_match = re.search(r"(\d{4})/(\d{2})", link)
                    if date_match:
                        year = int(date_match.group(1))
                        month_num = int(date_match.group(2))
                        post_date = pd.Timestamp(year=year, month=month_num, day=15)
                    else:
                        continue

                    # Check if this month is in our range
                    if post_date < pd.Timestamp(start_date) or post_date > pd.Timestamp(end_date):
                        continue

                    # Look for tables with attack data
                    tables = post_soup.find_all("table")
                    for table in tables:
                        rows = table.find_all("tr")
                        for row in rows:
                            cells = row.find_all(["td", "th"])
                            if len(cells) >= 2:
                                cat_text = cells[0].get_text(strip=True)
                                try:
                                    count = int(re.sub(r"[^\d]", "", cells[1].get_text(strip=True)))
                                except (ValueError, IndexError):
                                    continue

                                # Fuzzy match to our 26 threat types
                                best_match = "Vulnerability"
                                best_score = 0
                                for threat in THREATS:
                                    score = fuzz.partial_ratio(cat_text.lower(), threat.lower())
                                    if score > best_score:
                                        best_score = score
                                        best_match = threat

                                if best_score < 50:
                                    best_match = "Vulnerability"

                                # Generate count number of incident rows
                                for _ in range(min(count, 200)):
                                    day = rng.integers(1, 28)
                                    all_rows.append({
                                        "date": pd.Timestamp(year=post_date.year, month=post_date.month, day=day),
                                        "attack_type": best_match,
                                        "country": None,
                                        "description": f"{best_match} incident (Hackmageddon {post_date.strftime('%Y-%m')})",
                                    })

                    time.sleep(1.5)

                except Exception as exc:
                    logger.debug("Failed to parse Hackmageddon post %s: %s", link, exc)
                    continue

        except Exception as exc:
            logger.warning("Hackmageddon archive scrape failed: %s", exc)
            return None

        if not all_rows:
            return None

        df = pd.DataFrame(all_rows)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

        if self._cache:
            self._cache.save(cache_key, df)

        return df

    # ------------------------------------------------------------------
    # Synthetic fallback  (original mock — UNCHANGED)
    # ------------------------------------------------------------------

    def _mock_incidents(
        self,
        start_date: str,
        end_date: str,
        n: int,
    ) -> pd.DataFrame:
        rng = np.random.default_rng(42)
        months = pd.date_range(start_date, end_date, freq="MS")
        dates = rng.choice(months, size=n, replace=True)
        threats = rng.choice(THREATS, size=n, replace=True)
        countries = rng.choice(
            COUNTRIES_36 + [None],
            size=n,
            replace=True,
            p=[*(np.repeat(0.97 / 36, 36)), 0.03],
        )
        descriptions = [
            f"{t} incident reported in {c or 'unknown region'}."
            for t, c in zip(threats, countries)
        ]
        df = pd.DataFrame({
            "date": pd.to_datetime(dates),
            "attack_type": threats,
            "country": countries,
            "description": descriptions,
        })
        return df.sort_values("date").reset_index(drop=True)

    # ------------------------------------------------------------------
    # Country imputation  (UNCHANGED from original)
    # ------------------------------------------------------------------

    @staticmethod
    def impute_missing_country(df: pd.DataFrame) -> pd.DataFrame:
        def infer_country(row) -> Optional[str]:
            if pd.notna(row["country"]):
                return row["country"]
            desc = str(row["description"])
            for code in COUNTRIES_36:
                if f" {code} " in f" {desc} ":
                    return code
            return "US"

        out = df.copy()
        out["country"] = out.apply(infer_country, axis=1)
        return out
