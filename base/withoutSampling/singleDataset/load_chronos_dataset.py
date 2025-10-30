#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
load_chronos_csv.py

Demo script that loads a Chronos time‑series CSV from S3 into a long‑format
pandas DataFrame (and optionally into an AutoGluon TimeSeriesDataFrame).

Supported URLs (examples):
    • https://autogluon.s3.amazonaws.com/datasets/timeseries/m4_hourly/train.csv
    • https://autogluon.s3.amazonaws.com/datasets/timeseries/monash_tourism_yearly/train.csv

Author:  ChatGPT  – 2025‑10‑30
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

# ----------------------------------------------------------------------
# Helper: download & tidy a CSV from the public S3 bucket
# ----------------------------------------------------------------------
def load_long_csv(csv_url: str) -> pd.DataFrame:
    """
    Reads a Chronos CSV (long format) from ``csv_url`` into a pandas DataFrame
    and makes sure the ``timestamp`` column is a proper datetime.

    Expected columns (the canonical Chronos schema):
        - item_id   (string / identifier of the series)
        - timestamp (ISO‑8601 date‑time string)
        - target    (numeric observation)

    Returns
    -------
    pd.DataFrame
        Long‑format dataframe with ``timestamp`` as ``datetime64[ns]``.
    """
    # --------------------------------------------------------------
    # 1️⃣  Read CSV straight from the URL.
    #    ``low_memory=False`` forces pandas to infer dtypes in a single pass –
    #    this avoids the dreaded “mixed‑type column” warning for large files.
    # --------------------------------------------------------------
    print(f"📥 Downloading {csv_url} …")
    df = pd.read_csv(csv_url, low_memory=False)

    # --------------------------------------------------------------
    # 2️⃣  Sanity‑check that the three core columns exist.
    # --------------------------------------------------------------
    required_cols = {"item_id", "timestamp", "target"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"The CSV at {csv_url} is missing the required columns: {missing}"
        )

    # --------------------------------------------------------------
    # 3️⃣  Convert timestamp strings → pandas datetime (timezone‑naïve UTC)
    # --------------------------------------------------------------
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    # --------------------------------------------------------------
    # 4️⃣  OPTIONAL – enforce numeric target (coerce non‑numeric to NaN, then ffill)
    # --------------------------------------------------------------
    df["target"] = pd.to_numeric(df["target"], errors="coerce")
    if df["target"].isna().any():
        # Forward‑fill missing values; if the first value is NaN we replace it with 0.
        df["target"] = df["target"].ffill().fillna(0)

    # --------------------------------------------------------------
    # 5️⃣  Sort for reproducibility (not strictly required for Chronos)
    # --------------------------------------------------------------
    df = df.sort_values(["item_id", "timestamp"]).reset_index(drop=True)

    return df


# ----------------------------------------------------------------------
# Optional: convert to AutoGluon’s TimeSeriesDataFrame (if you have AutoGluon)
# ----------------------------------------------------------------------
def to_autogluon_tsdf(df: pd.DataFrame):
    """
    Convert a long‑format pandas DataFrame into an AutoGluon TimeSeriesDataFrame.
    Only works when the ``autogluon.timeseries`` package is installed.
    """
    try:
        from autogluon.timeseries import TimeSeriesDataFrame
    except Exception as exc:
        raise ImportError(
            "AutoGluon‑TS is not installed. Install it via:\n"
            "    pip install \"autogluon[timeseries]\""
        ) from exc

    # AutoGluon’s loader expects columns named exactly as we have them.
    ts_df = TimeSeriesDataFrame.from_data_frame(df, item_id="item_id", timestamp="timestamp")
    return ts_df


# ----------------------------------------------------------------------
# Optional: write the (clean) dataframe to Parquet for fast future loads
# ----------------------------------------------------------------------
def write_parquet(df: pd.DataFrame, out_path: Path):
    """
    Save ``df`` as a single parquet file (compressed with gzip).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, compression="gzip")
    print(f"✅  Parquet written to {out_path}")


# ----------------------------------------------------------------------
# Command‑line interface
# ----------------------------------------------------------------------
def _cli() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Load a Chronos CSV (long format) from S3 into pandas. "
            "Optionally convert to AutoGluon TimeSeriesDataFrame and/or write Parquet."
        )
    )
    parser.add_argument(
        "url",
        help=(
            "Public S3 URL of the CSV. Examples:\n"
            "  https://autogluon.s3.amazonaws.com/datasets/timeseries/m4_hourly/train.csv\n"
            "  https://autogluon.s3.amazonaws.com/datasets/timeseries/monash_tourism_yearly/train.csv"
        ),
    )
    parser.add_argument(
        "--to-tsdf",
        action="store_true",
        help="Also return an Autogluon TimeSeriesDataFrame (requires autogluon[timeseries])",
    )
    parser.add_argument(
        "--parquet",
        type=Path,
        metavar="FILE",
        help="Write the cleaned data to this Parquet file (gzip compression).",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1️⃣  Load CSV → pandas
    # ------------------------------------------------------------------
    df = load_long_csv(args.url)

    print("\n📊 Input dataframe shape :", df.shape)
    print("🔎 First 5 rows:")
    print(df.head(5).to_string(index=False))

    # ------------------------------------------------------------------
    # 2️⃣  Optional: write Parquet
    # ------------------------------------------------------------------
    if args.parquet:
        write_parquet(df, args.parquet)

    # ------------------------------------------------------------------
    # 3️⃣  Optional: convert to AutoGluon TimeSeriesDataFrame
    # ------------------------------------------------------------------
    if args.to_tsdf:
        ts_df = to_autogluon_tsdf(df)
        print("\n🚀 AutoGluon TimeSeriesDataFrame summary:")
        print(ts_df.info())
        # Example: show the first series only
        first_item = df["item_id"].iloc[0]
        print(f"\nFirst series (item_id = {first_item}):")
        print(ts_df.loc[first_item].head())
        # If you want to keep it for later use:
        # ts_df.save("my_tsdf.agts")   # AutoGluon native binary format

    print("\n✅  All done.")


if __name__ == "__main__":
    _cli()
# python load_chronos_csv.py \
#     https://autogluon.s3.amazonaws.com/datasets/timeseries/monash_tourism_yearly/train.csv

# python load_chronos_csv.py \
#     https://autogluon.s3.amazonaws.com/datasets/timeseries/monash_tourism_yearly/train.csv \
#     --parquet ./tourism_hourly_clean.parquet.gz
