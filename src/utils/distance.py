from __future__ import annotations
import numpy as np
import pandas as pd

def ensure_strictly_increasing(series: pd.Series, eps: float = 1e-6) -> pd.Series:
    """
    Ensure a distance-like series is strictly increasing by nudging duplicates upward.
    This avoids divide-by-zero in np.gradient and similar operations.
    """
    x = series.to_numpy(dtype=float).copy()
    for i in range(1, len(x)):
        if x[i] <= x[i - 1]:
            x[i] = x[i - 1] + eps
    return pd.Series(x, index=series.index, name=series.name)