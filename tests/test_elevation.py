import numpy as np
import pandas as pd

from src.elevation import clean_elevation


def test_clean_elevation_interpolates_missing():
    df = pd.DataFrame(
        {
            "cum_distance": [0, 20, 40, 60, 80],
            "elev": [10, np.nan, np.nan, 13, 14],
        }
    )

    cleaned, q = clean_elevation(
        df,
        elev_col="elev",
        dist_col="cum_distance",
        apply_savgol=False,
    )

    arr = np.asarray(cleaned, dtype=float)
    assert np.isfinite(arr).all()
    assert isinstance(q, dict)
    assert "missing_frac" in q


def test_clean_elevation_removes_spikes():
    df = pd.DataFrame(
        {
            "cum_distance": [0, 20, 40, 60, 80, 100],
            "elev": [10, 11, 200, 12, 13, 14],  # obvious spike
        }
    )

    cleaned, q = clean_elevation(
        df,
        elev_col="elev",
        dist_col="cum_distance",
        apply_savgol=False,
    )

    arr = np.asarray(cleaned, dtype=float)

    # Spike should be meaningfully reduced
    assert arr[2] < 100
    assert isinstance(q, dict)