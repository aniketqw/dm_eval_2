#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
download_london_smart_meters.py

Stream the Monash London Smart‑Meters Chronos dataset, clean NaNs and write it
to disk (JSON‑Lines, Parquet or CSV). Works on Python >=3.8.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from datasets import load_dataset
from tqdm.auto import tqdm


# ----------------------------------------------------------------------
# CLI parsing
# ----------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download the Monash London Smart Meters Chronos dataset "
            "without truncation and write it to disk."
        )
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="./london_smart_meters_proper",
        help="Directory where the processed files will be stored.",
    )
    parser.add_argument(
        "-f",
        "--format",
        choices=["jsonl", "parquet", "csv"],
        default="jsonl",
        help="Output format. JSONL is the default because it streams naturally.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help=(
            "How many series to buffer before flushing to disk "
            "(only used for Parquet/CSV output)."
        ),
    )
    return parser.parse_args()


# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def _clean_series(item: Dict, series_idx: int) -> Optional[Dict]:
    """Convert a raw HuggingFace entry to the canonical Chronos representation."""
    target = item.get("target")
    if not target:
        return None

    # 1) numpy array (float32)
    target_arr = np.asarray(target, dtype=np.float32)

    # 2) forward‑fill NaNs (if any)
    if np.isnan(target_arr).any():
        series_pd = pd.Series(target_arr)
        target_arr = series_pd.ffill().fillna(0).values.astype(np.float32)

    # 3) final dict
    return {
        "item_id": f"series_{series_idx}",
        "start": str(item.get("start", "2011-01-01 00:00:00")),
        "target": target_arr.tolist(),
    }


def _write_jsonl(series: Dict, fh):
    json_line = json.dumps(series, separators=(",", ":"))
    fh.write(json_line + "\n")


def _save_batch_parquet(batch: List[Dict], out_path: Path):
    df = pd.DataFrame(batch)
    df = df.explode("target")          # one row per timestamp
    df.to_parquet(out_path, index=False)


def _save_batch_csv(batch: List[Dict], out_path: Path):
    df = pd.DataFrame(batch)
    df = df.explode("target")
    df.to_csv(out_path, index=False)


# ----------------------------------------------------------------------
# Main download routine
# ----------------------------------------------------------------------
def download_proper_dataset(
    output_dir: str,
    out_format: str = "jsonl",
    batch_size: int = 1000,
) -> Optional[Path]:
    print("📥 Streaming London Smart Meters dataset from HuggingFace …")
    try:
        dataset = load_dataset(
            "autogluon/chronos_datasets",
            "monash_london_smart_meters",
            split="train",
            streaming=True,         # low‑memory mode
        )
        print(f"🔎 Dataset reports {getattr(dataset, 'num_rows', 'unknown')} series")

        out_dir = Path(output_dir).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        jsonl_fh = None
        if out_format == "jsonl":
            jsonl_path = out_dir / "train.jsonl"
            jsonl_fh = jsonl_path.open("w", encoding="utf-8")

        buffer: List[Dict] = []          # only used for parquet/csv
        stats = {"written": 0}

        with tqdm(dataset, unit="series", desc="Downloading") as pbar:
            for idx, raw_item in enumerate(pbar):
                cleaned = _clean_series(raw_item, idx)
                if cleaned is None:
                    continue

                if out_format == "jsonl":
                    _write_jsonl(cleaned, jsonl_fh)          # immediate write
                else:
                    buffer.append(cleaned)
                    if len(buffer) >= batch_size:
                        batch_id = idx // batch_size
                        batch_path = out_dir / f"train_batch_{batch_id:04d}.{out_format}"
                        if out_format == "parquet":
                            _save_batch_parquet(buffer, batch_path)
                        else:   # csv
                            _save_batch_csv(buffer, batch_path)
                        buffer.clear()

                stats["written"] += 1
                if idx % 1000 == 0 and idx > 0:
                    pbar.set_postfix(written=stats["written"])

        # Flush last (possibly incomplete) batch for parquet/csv
        if out_format != "jsonl" and buffer:
            batch_id = stats["written"] // batch_size
            batch_path = out_dir / f"train_batch_{batch_id:04d}.{out_format}"
            if out_format == "parquet":
                _save_batch_parquet(buffer, batch_path)
            else:
                _save_batch_csv(buffer, batch_path)

        if jsonl_fh:
            jsonl_fh.close()

        print("\n✅ Finished!")
        print(f"🗂  Output directory : {out_dir}")
        print(f"🔢 Total series written : {stats['written']}")
        if out_format != "jsonl":
            files = sorted(out_dir.glob(f"*.{out_format}"))
            print(f"📁 {len(files)} batch files created ({out_format})")
        return out_dir

    except KeyboardInterrupt:
        print("\n⚠️  Download interrupted by user – partial data may remain.")
        return None
    except Exception as exc:      # pragma: no cover
        print(f"\n❌  Unexpected error: {exc}", file=sys.stderr)
        return None


def main() -> None:
    args = _parse_args()
    download_proper_dataset(
        output_dir=args.output_dir,
        out_format=args.format,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
