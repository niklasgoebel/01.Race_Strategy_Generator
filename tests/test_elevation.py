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
    cleaned, q = clean_elevation(df, elev_col="elev", dist_col="cum_distance", apply_savgol=False)
    assert np.isfinite(cleaned.to_numpy()).all()
    assert q["missing_frac"] > 0


def test_clean_elevation_removes_spikes():
    df = pd.DataFrame(
        {
            "cum_distance": [0, 20, 40, 60, 80, 100],
            "elev": [10, 11, 200, 12, 13, 14],  # spike at 40m
        }
    )
    cleaned, q = clean_elevation(df, elev_col="elev", dist_col="cum_distance", apply_savgol=False, spike_z_thresh=6.0)
    # spike should be fixed (value near local trend)
    assert q["spikes_fixed"] >= 1
    assert cleaned.iloc[2] < 50