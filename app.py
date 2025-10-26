import csv
import pandas as pd
import numpy as np
from io import StringIO

def _sniff_delimiter(sample: str) -> str:
    try:
        return csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t"]).delimiter
    except Exception:
        return ";"  # EU fallback

def _clean_angles(vals):
    cleaned = (
        pd.Series(vals)
        .astype(str)
        .str.replace("°", "", regex=False)
        .str.replace("deg", "", case=False, regex=False)
        .str.replace(",", ".", regex=False)  # allow decimal comma in headers
        .str.strip()
    )
    return pd.to_numeric(cleaned, errors="coerce")

def load_matrix_csv(path: str) -> pd.DataFrame:
    """
    Loads a jib matrix where:
      - Columns = Main-jib angles (deg)
      - Rows (index) = Folding-jib angles (deg)
    Robust to semicolon/comma/tab delimiters and messy headers.
    """
    # Read original text once
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        raw = f.read()

    # Delimiter detection
    delim = _sniff_delimiter(raw[:4096])

    # First attempt: pandas auto-detect with sep=None (python engine) + index_col=0
    try:
        df = pd.read_csv(StringIO(raw), sep=None, engine="python", index_col=0)
    except Exception:
        # Fallback to detected delimiter
        df = pd.read_csv(StringIO(raw), sep=delim, engine="python")

        # If index not set, but the first column looks like angles or a common label, make it the index
        first_col = df.columns[0]
        if (
            first_col.lower().strip() in {"", "unnamed: 0", "foldingjib", "fold", "fold_angle", "folding"}
            or df[first_col].dropna().astype(str).str.replace(",", ".", regex=False)
               .str.match(r"^\s*-?\d+(\.\d+)?\s*$").all()
        ):
            df = df.set_index(first_col)

    # Clean angles (headers/index) → float degrees
    df.columns = _clean_angles(df.columns)
    df.index   = _clean_angles(df.index)

    # Coerce values to numeric (supports decimal comma inside cells)
    # Replace decimal commas only in the value area
    df = df.apply(lambda s: pd.to_numeric(s.astype(str).str.replace(",", ".", regex=False), errors="coerce"))

    # Drop fully-empty rows/cols and sort axes
    df = df.dropna(how="all").dropna(axis=1, how="all")
    df = df.sort_index().sort_index(axis=1)

    print(f"[load_matrix_csv] {path}: shape={df.shape}, delim='{delim}'")
    if not df.empty:
        print(
            f"[load_matrix_csv] fold_angles={df.index.min()}..{df.index.max()} | "
            f"main_angles={df.columns.min()}..{df.columns.max()}"
        )

    if df.empty:
        raise ValueError(f"{path} parsed to an empty DataFrame. Check delimiter/layout.")
    return df
