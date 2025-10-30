#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
load_chronos_hf.py

Download a Chronos dataset from the Hugging‑Face hub,
convert it to the **long** Chronos format
(item_id, timestamp, target) and optionally write a gzip‑compressed Parquet.

The script works for all Chronos datasets that are stored in the
“list‑of‑timestamps / list‑of‑targets” representation
(e.g. monash_tourism_yearly, m4_hourly, etc.).
"""

import argparse
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from tqdm.auto import tqdm


def _load_hf_dataset(name: str, split: str = "train") -> pd.DataFrame:
    """
    Load a dataset from the `autogluon/chronos_datasets` hub
    and return a **long‑format** pandas DataFrame.

    Parameters
    ----------
    name: str
        The config name, e.g. ``monash_tourism_yearly`` or ``m4_hourly``.
    split: str, default "train"
        Which split to load (most Chronos datasets have only a train split).

    Returns
    -------
    pd.DataFrame
        Columns: item_id (str), timestamp (datetime64[ns, UTC]), target (float)
    """
    # --------------------------------------------------------------
    # 1️⃣  Load the raw hub dataset (list‑of‑timestamps / list‑of‑targets)
    # --------------------------------------------------------------
    raw = load_dataset(
        "autogluon/chronos_datasets",
        name,
        split=split,
        # `trust_remote_code` was removed in 🤗 datasets ≥2.19, so we simply omit it.
    )
    # ------------------------------------------------------------------
    # 2️⃣  Convert to pandas – the three columns are `id`, `timestamp`, `target`
    # ------------------------------------------------------------------
    df = raw.to_pandas()

    # ------------------------------------------------------------------
    # 3️⃣  Rename `id` → `item_id` to match Chronos expectations
    # ------------------------------------------------------------------
    if "id" not in df.columns:
        raise RuntimeError("The hub dataset does not contain an `id` column.")
    df = df.rename(columns={"id": "item_id"})

    # ------------------------------------------------------------------
    # 4️⃣  Explode the list‑columns so that each (timestamp, target) pair
    #     becomes a separate row.
    # ------------------------------------------------------------------
    #   pandas.explode works on a column of list‑like objects.
    #   We explode both columns at the same time to keep the alignment.
    df = df.explode(["timestamp", "target"]).reset_index(drop=True)

    # ------------------------------------------------------------------
    # 5️⃣  Clean / cast the columns
    # ------------------------------------------------------------------
    #   * timestamp → pandas datetime (UTC‑naïve)
    #   * target    → numeric (coerce errors, forward‑fill any NaNs)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["target"] = pd.to_numeric(df["target"], errors="coerce")
    if df["target"].isna().any():
        df["target"] = df["target"].ffill().fillna(0)

    # ------------------------------------------------------------------
    # 6️⃣  Final sanity check – we now have exactly the three columns we need
    # ------------------------------------------------------------------
    required = {"item_id", "timestamp", "target"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing columns after processing: {missing}")

    # ------------------------------------------------------------------
    # 7️⃣  Optional: sort for reproducibility (not required for Chronos)
    # ------------------------------------------------------------------
    df = df.sort_values(["item_id", "timestamp"]).reset_index(drop=True)

    return df


def _write_parquet(df: pd.DataFrame, out_path: Path) -> None:
    """Save the long DataFrame as a gzip‑compressed Parquet file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # `compression="gzip"` writes a single gzip‑compressed parquet file
    df.to_parquet(out_path, compression="gzip")
    print(f"✅  Parquet written to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download a Chronos dataset from Hugging‑Face, turn it into "
            "Chronos long format, and optionally write a gzip‑compressed Parquet."
        )
    )
    parser.add_argument(
        "config",
        help=(
            "Config name of the Chronos dataset, e.g. "
            "`monash_tourism_yearly`, `m4_hourly`, `electricity_15min` ..."
        ),
    )
    parser.add_argument(
        "--parquet",
        type=Path,
        metavar="FILE",
        help="Write the long DataFrame to this parquet file (gzip compressed).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Print a tiny preview of the long DataFrame after processing.",
    )
    args = parser.parse_args()

    # --------------------------------------------------------------
    # 1️⃣  Load & reshape the hub dataset
    # --------------------------------------------------------------
    print(f"🚚  Loading Hugging‑Face config `{args.config}` …")
    df_long = _load_hf_dataset(args.config)

    print("\n📊  Long‑format DataFrame shape :", df_long.shape)
    if args.show:
        print("🔎  First 5 rows:")
        print(df_long.head(5).to_string(index=False))

    # --------------------------------------------------------------
    # 2️⃣  Write parquet if the user asked for it
    # --------------------------------------------------------------
    if args.parquet:
        _write_parquet(df_long, args.parquet)

    print("\n✅  All done.")


if __name__ == "__main__":
    main()
# chmod +x load_chronos_hf.py
# ./load_chronos_hf.py monash_cif_2016 \
#     --parquet cif_2016_clean.parquet.gz \
#     --show