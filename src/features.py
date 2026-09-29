"""
URL feature extraction utilities.

Extracts the same 23 numeric features used by the training dataset
from a raw URL string, so that live single-URL predictions match
the model's training distribution.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from urllib.parse import urlparse
from typing import Dict, Any

import pandas as pd


# Regex to detect an IPv4 / IPv6 host.  We keep it simple but
# robust enough for the vast majority of real-world URLs.
_IP_RE = re.compile(
    r"^(?:\d{1,3}\.){3}\d{1,3}$"          # IPv4
    r"|\[?[0-9a-fA-F:]+(?:\.[0-9a-fA-F:]+)*\]?$"  # IPv6
)


def _shannon_entropy(text: str) -> float:
    """Shannon entropy (base-2) of a string."""
    if not text:
        return 0.0
    counts = Counter(text)
    length = len(text)
    return -sum((c / length) * math.log2(c / length) for c in counts.values())


def _is_ip(host: str) -> int:
    """Return 1 if the host string is an IP address, else 0."""
    if not host:
        return 0
    return 1 if _IP_RE.match(host) else 0


def _special_char_count(url: str) -> int:
    """Count special characters (non-alphanumeric, non-path-delimiter)."""
    return sum(
        1
        for c in url
        if not (c.isalnum() or c in "/?=#&.-_:")
    )


def extract_url_features(url: str) -> Dict[str, Any]:
    """Extract all 23 numeric features from a raw URL string.

    Returns a dict keyed by the same column names as the training
    dataset (excluding `url`, `dom`, `tld`, and `label`).
    """
    if not url or not isinstance(url, str):
        url = ""

    # Ensure scheme so urlparse behaves correctly
    raw = url.strip()
    if not raw.startswith(("http://", "https://", "ftp://")):
        raw = "http://" + raw

    parsed = urlparse(raw)
    host = (parsed.hostname or "").lower()
    path = parsed.path or ""
    query = parsed.query or ""
    scheme = (parsed.scheme or "").lower()

    is_https = 1 if scheme == "https" else 0
    is_ip = _is_ip(host)

    # Domain / TLD extraction
    # Strip leading "www." for cleaner domain parsing
    domain_for_tld = host[4:] if host.startswith("www.") else host
    parts = domain_for_tld.split(".")
    if len(parts) >= 2:
        tld = ".".join(parts[-2:]) if parts[-1] in {"uk", "au", "jp", "nz", "za"} and len(parts) >= 3 else parts[-1]
        # Re-derive tld after potential join
        if len(parts) >= 3 and parts[-1] in {"uk", "au", "jp", "nz", "za"} and parts[-2] in {"co", "com", "org", "ac", "gov", "edu", "net", "nhs"}:
            tld = ".".join(parts[-2:])
        else:
            tld = parts[-1]
        dom = domain_for_tld
    else:
        tld = ""
        dom = host

    url_len = len(url)
    dom_len = len(dom)
    tld_len = len(tld)
    subdom_cnt = max(0, len(parts) - 2) if parts else 0

    letter_cnt = sum(1 for c in url if c.isalpha())
    digit_cnt = sum(1 for c in url if c.isdigit())
    special_cnt = _special_char_count(url)

    eq_cnt = url.count("=")
    qm_cnt = url.count("?")
    amp_cnt = url.count("&")
    dot_cnt = url.count(".")
    dash_cnt = url.count("-")
    under_cnt = url.count("_")

    total_chars = max(1, len(url))
    letter_ratio = letter_cnt / total_chars
    digit_ratio = digit_cnt / total_chars
    spec_ratio = special_cnt / total_chars

    slash_cnt = url.count("/")
    path_len = len(path)
    query_len = len(query)
    entropy = _shannon_entropy(url)

    return {
        "url_len": url_len,
        "dom_len": dom_len,
        "is_ip": is_ip,
        "tld": tld,
        "tld_len": tld_len,
        "subdom_cnt": subdom_cnt,
        "letter_cnt": letter_cnt,
        "digit_cnt": digit_cnt,
        "special_cnt": special_cnt,
        "eq_cnt": eq_cnt,
        "qm_cnt": qm_cnt,
        "amp_cnt": amp_cnt,
        "dot_cnt": dot_cnt,
        "dash_cnt": dash_cnt,
        "under_cnt": under_cnt,
        "letter_ratio": round(letter_ratio, 9),
        "digit_ratio": round(digit_ratio, 9),
        "spec_ratio": round(spec_ratio, 9),
        "is_https": is_https,
        "slash_cnt": slash_cnt,
        "entropy": round(entropy, 9),
        "path_len": path_len,
        "query_len": query_len,
    }


NUMERIC_FEATURES = [
    "url_len",
    "dom_len",
    "is_ip",
    "tld_len",
    "subdom_cnt",
    "letter_cnt",
    "digit_cnt",
    "special_cnt",
    "eq_cnt",
    "qm_cnt",
    "amp_cnt",
    "dot_cnt",
    "dash_cnt",
    "under_cnt",
    "letter_ratio",
    "digit_ratio",
    "spec_ratio",
    "is_https",
    "slash_cnt",
    "entropy",
    "path_len",
    "query_len",
]

CATEGORICAL_FEATURES = ["tld"]

ALL_FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def urls_to_features_dataframe(urls: list[str]) -> pd.DataFrame:
    """Convert a list of URLs into a feature DataFrame matching training schema."""
    rows = [extract_url_features(u) for u in urls]
    return pd.DataFrame(rows, columns=ALL_FEATURE_COLUMNS)
