import math
import re
from typing import Dict, Iterable
import numpy as np
import pandas as pd

def season_from_today():
    import datetime as dt
    today = dt.date.today()
    # If you want to force a year, set env YEAR and read it in compute_rankings.py
    return today.year

def zscore(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    return (s - s.mean()) / (s.std(ddof=0) if s.std(ddof=0) else 1.0)

def minmax01(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    mn, mx = float(s.min()), float(s.max())
    if mx - mn == 0:
        return pd.Series([0.5]*len(s), index=s.index)
    return (s - mn) / (mx - mn)

def scale_0_100(series: pd.Series) -> pd.Series:
    return (minmax01(series) * 100).round(2)

def invert(series: pd.Series) -> pd.Series:
    # For metrics where lower is better: invert by negating before zscore
    return -pd.to_numeric(series, errors="coerce")

def slugify_team(name: str) -> str:
    name = name.strip().lower()
    name = name.replace("&", "and")
    name = re.sub(r"[^a-z0-9 ]+", " ", name)
    name = re.sub(r"\s+", "-", name).strip("-")
    return name

def mean_ignore_none(values: Iterable[float]) -> float:
    vals = [v for v in values if v is not None and not math.isnan(v)]
    return float(np.mean(vals)) if vals else 0.0

def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None
