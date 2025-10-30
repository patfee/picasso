import os
import io
import csv
import numpy as np
import pandas as pd
from functools import lru_cache
from scipy.interpolate import RegularGridInterpolator

# -------- Config (defaults assume project tree: ./data next to ./app) --------
DATA_DIR = os.getenv("DATA_DIR", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data")))
HEIGHT_CSV = os.getenv("HEIGHT_CSV", "height.csv")
OUTREACH_CSV = os.getenv("OUTREACH_CSV", "outreach.csv")
LOAD_CSV = os.getenv("LOAD_CSV", "Harbour_Cdyn115.csv")  # Harbour Cdyn=1.15
PEDESTAL_HEIGHT_M = float(os.getenv("PEDESTAL_HEIGHT_M", "6"))

# Display / perf caps (kept for parity with monolithic version)
MAX_POINTS_FOR_DISPLAY  = 15000
MAX_POINTS_FOR_ENVELOPE = 6000
MAX_POINTS_FOR_KNN      = 1200
KNN_K                   = 8


def _sniff_delimiter(sample: str) -> str:
    try:
        return csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t"]).delimiter
    except Exception:
        return ";"


def _clean_angles(vals):
    cleaned = (
        pd.Series(vals)
        .astype(str)
        .str.replace("°", "", regex=False)
        .str.replace("deg", "", case=False, regex=False)
        .str.replace(",", ".", regex=False)
        .str.strip()
    )
    return pd.to_numeric(cleaned, errors="coerce")


def load_matrix_csv(path: str) -> pd.DataFrame:
    """
    Load matrix CSV with:
      - columns = Main-jib angles
      - index   = Folding-jib angles
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        raw = f.read()
    delim = _sniff_delimiter(raw[:4096])

    try:
        df = pd.read_csv(io.StringIO(raw), sep=None, engine="python", index_col=0)
    except Exception:
        df = pd.read_csv(io.StringIO(raw), sep=delim, engine="python")
        first_col = df.columns[0]
        if (
            first_col.lower().strip()
            in {"", "unnamed: 0", "foldingjib", "fold", "fold_angle", "folding"}
            or df[first_col]
            .dropna()
            .astype(str)
            .str.replace(",", ".", regex=False)
            .str.match(r"^\s*-?\d+(\.\d+)?\s*$")
            .all()
        ):
            df = df.set_index(first_col)

    df.columns = _clean_angles(df.columns)
    df.index   = _clean_angles(df.index)
    df = df.apply(lambda s: pd.to_numeric(s.astype(str).str.replace(",", ".", regex=False), errors="coerce"))
    df = df.dropna(how="all").dropna(axis=1, how="all").sort_index().sort_index(axis=1)
    if df.empty:
        raise ValueError(f"{path} parsed to empty DataFrame.")
    return df


def build_interpolators(height_df: pd.DataFrame, outreach_df: pd.DataFrame, load_df: pd.DataFrame):
    # Align to same angle axes
    fold_angles = height_df.index.values.astype(float)
    main_angles = height_df.columns.values.astype(float)
    outreach_df = outreach_df.loc[fold_angles, main_angles]
    load_df     = load_df.loc[fold_angles, main_angles]

    height_itp = RegularGridInterpolator((fold_angles, main_angles), height_df.values, bounds_error=False, fill_value=None)
    outre_itp  = RegularGridInterpolator((fold_angles, main_angles), outreach_df.values, bounds_error=False, fill_value=None)
    load_itp   = RegularGridInterpolator((fold_angles, main_angles), load_df.values, bounds_error=False, fill_value=None)

    return fold_angles, main_angles, height_itp, outre_itp, load_itp


def resample_grid_by_factors(fold_angles, main_angles, fold_factor, main_factor):
    def expand(arr, f):
        arr = np.unique(np.asarray(arr, float))
        arr.sort()
        if len(arr) < 2 or int(f) <= 1:
            return arr
        out = [arr[0]]
        for i in range(len(arr) - 1):
            a, b = arr[i], arr[i + 1]
            seg = np.linspace(a, b, int(f) + 1, endpoint=True)[1:]  # keep originals, add equal subdivisions
            out.extend(seg.tolist())
        return np.array(out, float)

    fnew = expand(fold_angles, fold_factor)
    mnew = expand(main_angles, main_factor)
    F, M = np.meshgrid(fnew, mnew, indexing="ij")
    pts = np.column_stack([F.ravel(), M.ravel()])  # [Folding, Main]
    return fnew, mnew, F, M, pts


@lru_cache(maxsize=1)
def get_context():
    """
    Lazy-loaded dataset & interpolators.
    Returns dict with:
      height_df, outreach_df, load_df,
      fold_angles, main_angles, height_itp, outre_itp, load_itp,
      constants
    """
    height_df   = load_matrix_csv(os.path.join(DATA_DIR, HEIGHT_CSV))
    outreach_df = load_matrix_csv(os.path.join(DATA_DIR, OUTREACH_CSV))
    load_df     = load_matrix_csv(os.path.join(DATA_DIR, LOAD_CSV))
    # Align load to height axes
    load_df = load_df.reindex(index=height_df.index, columns=height_df.columns)

    fold_angles, main_angles, height_itp, outre_itp, load_itp = build_interpolators(height_df, outreach_df, load_df)

    return {
        "height_df": height_df,
        "outreach_df": outreach_df,
        "load_df": load_df,
        "fold_angles": fold_angles,
        "main_angles": main_angles,
        "height_itp": height_itp,
        "outre_itp": outre_itp,
        "load_itp": load_itp,
        "PEDESTAL_HEIGHT_M": PEDESTAL_HEIGHT_M,
        "MAX_POINTS_FOR_DISPLAY": MAX_POINTS_FOR_DISPLAY,
        "MAX_POINTS_FOR_ENVELOPE": MAX_POINTS_FOR_ENVELOPE,
        "MAX_POINTS_FOR_KNN": MAX_POINTS_FOR_KNN,
        "KNN_K": KNN_K,
    }
