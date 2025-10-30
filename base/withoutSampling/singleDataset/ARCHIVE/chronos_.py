#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Chronos M4‑Hourly Evaluation – single‑GPU (probabilistic)  
   *First tries to load the locally saved parquet file
   (m4_hourly_clean.parquet.gz). If the parquet does not exist,
   it downloads the official CSV from HuggingFace.*"""

# ----------------------------------------------------------------------
# 0️⃣  ONE‑GPU ENVIRONMENT
# ----------------------------------------------------------------------
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"   # use GPU 0 only
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ----------------------------------------------------------------------
# 1️⃣  IMPORTS
# ----------------------------------------------------------------------
import logging, pickle, time, sys, signal
from pathlib import Path

import numpy as np, pandas as pd, torch
from scipy.stats import gmean
from chronos import ChronosPipeline

# ----------------------------------------------------------------------
# 2️⃣  LOGGING
# ----------------------------------------------------------------------
def setup_logging():
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"chronos_eval_{int(time.time())}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger(__name__)

log = setup_logging()

# ----------------------------------------------------------------------
# 3️⃣  CONFIGURATION
# ----------------------------------------------------------------------
CONFIG = {
    # ------------------------------------------------------------------
    #   INPUT DATA
    # ------------------------------------------------------------------
    "CSV_URL": "https://autogluon.s3.amazonaws.com/datasets/timeseries/monash_tourism_yearly/train.csv",
    # Path to the parquet you already created
    "PARQUET_PATH": "/home/h20250169/study/modelTraining/dm_eval_2/base/withoutSampling/singleDataset/TOURISM_YEARLY/tourism_hourly_clean.parquet.gz",

    # ------------------------------------------------------------------
    #   MODEL / INFERENCE SETTINGS
    # ------------------------------------------------------------------
    "PREDICTION_LENGTH": 48,          # how many steps ahead we forecast
    "NUM_SAMPLES": 500,               # number of Monte‑Carlo draws per series
    "QUANTILE_LEVELS": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],

    # ------------------------------------------------------------------
    #   DISK / CHECKPOINT SETTINGS
    # ------------------------------------------------------------------
    "RESULTS_DIR": "results",
    "CHECKPOINT_INTERVAL": 500,       # write a checkpoint every N series
    "MAX_MEMORY_GB": 24,

    # ------------------------------------------------------------------
    #   OTHER TUNABLES
    # ------------------------------------------------------------------
    "BATCH_SIZE": 1,
    "CONTEXT_LENGTH": 512,
}
# ----------------------------------------------------------------------
# 4️⃣  GRACEFUL SHUTDOWN
# ----------------------------------------------------------------------
class GracefulShutdown:
    def __init__(self):
        self.shutdown_requested = False
        signal.signal(signal.SIGINT, self._handler)
        signal.signal(signal.SIGTERM, self._handler)

    def _handler(self, signum, frame):
        log.info(f"Received signal {signum}. Initiating graceful shutdown …")
        self.shutdown_requested = True

shutdown_manager = GracefulShutdown()

# ----------------------------------------------------------------------
# 5️⃣  CHECKPOINT / PROGRESS TRACKER (fully implemented)
# ----------------------------------------------------------------------
class ProgressTracker:
    """
    Tracks which series have already been processed, stores per‑series WQL /
    MASE values and writes a pickle checkpoint every ``checkpoint_interval``.
    """
    def __init__(self, results_dir: str, checkpoint_interval: int):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.checkpoint_interval = checkpoint_interval
        self.checkpoint_file = self.results_dir / "progress_checkpoint.pkl"

        self.processed: set[int] = set()
        self.wqls: list[float] = []
        self.mases: list[float] = []

        self._load()

    # --------------------------------------------------------------
    def _load(self):
        """Load a previously‑saved checkpoint (if any)."""
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, "rb") as f:
                    data = pickle.load(f)
                self.processed = set(data.get("processed", []))
                self.wqls = data.get("wqls", [])
                self.mases = data.get("mases", [])
                log.info(f"Loaded checkpoint – {len(self.processed)} series already done")
            except Exception as e:
                log.warning(f"Could not read checkpoint: {e}")

    # --------------------------------------------------------------
    def save(self, last_index: int):
        """Write a checkpoint containing everything we have so far."""
        try:
            payload = {
                "processed": list(self.processed),
                "wqls": self.wqls,
                "mases": self.mases,
                "last_index": last_index,
                "timestamp": time.time(),
            }
            with open(self.checkpoint_file, "wb") as f:
                pickle.dump(payload, f)
            log.info(f"Checkpoint saved after processing index {last_index}")
        except Exception as e:
            log.error(f"Failed to write checkpoint: {e}")

    # --------------------------------------------------------------
    def should_process(self, idx: int) -> bool:
        """True if this series has not been processed yet."""
        return idx not in self.processed

    # --------------------------------------------------------------
    def add(self, idx: int, wql_val: float, mase_val: float):
        """Append per‑series metrics and mark the series as done."""
        self.processed.add(idx)
        if not np.isnan(wql_val):
            self.wqls.append(wql_val)
        if not np.isnan(mase_val):
            self.mases.append(mase_val)

# ----------------------------------------------------------------------
# 6️⃣  METRICS (unchanged)
# ----------------------------------------------------------------------
def wql(actual: np.ndarray, pred: np.ndarray, qs: list[float]) -> float:
    """Weighted Quantile Loss (as used in the Chronos paper)."""
    if np.sum(np.abs(actual)) == 0:
        return np.nan
    actual = actual.reshape(-1, 1) if actual.ndim == 1 else actual
    losses = [
        np.maximum(q * (actual - pred[:, i]), (q - 1) * (actual - pred[:, i]))
        for i, q in enumerate(qs)
    ]
    return np.sum(np.stack(losses, axis=1)) / np.sum(np.abs(actual))

def mase(actual: np.ndarray, pred: np.ndarray, seasonal_err: np.ndarray) -> float:
    """Mean Absolute Scaled Error."""
    mae = np.mean(np.abs(actual - pred))
    denom = np.mean(np.abs(seasonal_err))
    return mae / denom if denom != 0 else (np.inf if mae > 0 else 0.0)

# ----------------------------------------------------------------------
# 7️⃣  DATA LOADER – parquet first, CSV fallback
# ----------------------------------------------------------------------
def load_m4_hourly(csv_url: str, parquet_path: str):
    """
    Returns three parallel lists:
        * ``contexts`` – everything except the last PREDICTION_LENGTH points
        * ``futures``  – the last PREDICTION_LENGTH points
        * ``valid_idx`` – integer indices (0 … N‑1)
    """
    # --------------------------------------------------------------
    # 1️⃣  Try to read the local parquet
    # --------------------------------------------------------------
    if Path(parquet_path).exists():
        log.info(f"✅ Loading data from local Parquet: {parquet_path}")
        df = pd.read_parquet(parquet_path)
    else:
        # --------------------------------------------------------------
        # 2️⃣  Parquet missing → download CSV (original behaviour)
        # --------------------------------------------------------------
        log.info(f"Parquet not found – downloading CSV from {csv_url}")
        df = pd.read_csv(csv_url, parse_dates=["timestamp"])

    # --------------------------------------------------------------
    # Common post‑processing (identical for both sources)
    # --------------------------------------------------------------
    log.info(f"Input dataframe shape : {df.shape}")
    log.info("First 5 rows:")
    log.info("\n" + df.head().to_string(index=False))

    # Make sure each series is ordered by timestamp
    df = df.sort_values(["item_id", "timestamp"]).reset_index(drop=True)

    contexts, futures, valid_idx = [], [], []

    for i, (item_id, grp) in enumerate(df.groupby("item_id")):
        if shutdown_manager.shutdown_requested:
            break

        target_arr = grp["target"].astype(np.float32).values

        # Defensive NaN handling (should be very rare)
        if np.isnan(target_arr).any():
            nan_cnt = np.isnan(target_arr).sum()
            target_arr = np.nan_to_num(target_arr, nan=0.0)
            log.warning(f"Series {i} ({item_id}) had {nan_cnt} NaNs → replaced with 0")

        # Split into context / future
        if len(target_arr) > CONFIG["PREDICTION_LENGTH"]:
            ctx = target_arr[:-CONFIG["PREDICTION_LENGTH"]]
            fut = target_arr[-CONFIG["PREDICTION_LENGTH"] :]
        else:
            pad = CONFIG["PREDICTION_LENGTH"] - len(target_arr)
            ctx = target_arr
            fut = (
                np.concatenate([target_arr[-pad:], np.zeros(pad)])
                if pad > 0
                else np.zeros(CONFIG["PREDICTION_LENGTH"])
            )
            log.warning(
                f"Series {i} ({item_id}) shorter than prediction length – padded with zeros"
            )

        contexts.append(ctx)
        futures.append(fut)
        valid_idx.append(i)

    log.info(f"Prepared {len(valid_idx)} series for inference")
    return contexts, futures, valid_idx

# ----------------------------------------------------------------------
# 8️⃣  PROCESS ONE SERIES – unchanged
# ----------------------------------------------------------------------
def process_one_series(pipeline, ctx, fut, idx, tracker):
    """Run Chronos on a single series and store WQL / MASE."""
    try:
        # (batch = 1) → shape (1, len(ctx))
        tensor = torch.from_numpy(ctx.astype(np.float32)).unsqueeze(0)

        with torch.no_grad():
            samples = pipeline.predict(
                tensor,
                prediction_length=CONFIG["PREDICTION_LENGTH"],
                num_samples=CONFIG["NUM_SAMPLES"],
            )
        # samples: (1, num_samples, prediction_length)
        samples_np = samples.cpu().numpy().squeeze(0)   # (num_samples, pred_len)

        # Quantiles across the sample dimension
        quantile_preds = np.column_stack(
            [np.quantile(samples_np, q, axis=0) for q in CONFIG["QUANTILE_LEVELS"]]
        )   # (pred_len, n_quantiles)

        # Median (0.5) is the 5‑th column (index 4)
        median_pred = quantile_preds[:, 4]

        # Metrics
        w = wql(fut, quantile_preds, CONFIG["QUANTILE_LEVELS"])
        seasonal_err = np.diff(ctx) if len(ctx) > 1 else np.array([1.0])
        m = mase(fut, median_pred, seasonal_err)

        # Record the result
        tracker.add(idx, w, m)

        if (idx + 1) % 100 == 0:
            log.info(f"Series {idx} → WQL={w:.4f}, MASE={m:.4f}")

    except Exception as exc:
        log.error(f"Error processing series {idx}: {exc}")

# ----------------------------------------------------------------------
# 9️⃣  MAIN – orchestrates everything
# ----------------------------------------------------------------------
def main():
    log.info("=== Chronos M4‑Hourly Evaluation (single‑GPU, probabilistic) ===")
    log.info(f"Visible GPUs: {torch.cuda.device_count()} (CUDA {torch.version.cuda})")
    log.info(f"Configuration: {CONFIG}")

    tracker = ProgressTracker(CONFIG["RESULTS_DIR"], CONFIG["CHECKPOINT_INTERVAL"])

    # --------------------------------------------------------------
    # 1️⃣  Load data (parquet preferred)
    # --------------------------------------------------------------
    contexts, futures, indices = load_m4_hourly(
        CONFIG["CSV_URL"], CONFIG["PARQUET_PATH"]
    )

    # --------------------------------------------------------------
    # 2️⃣  Load Chronos model (T5‑Base, fp16, auto‑device‑map)
    # --------------------------------------------------------------
    log.info("Loading Chronos model…")
    max_mem = {0: f"{CONFIG['MAX_MEMORY_GB']}GiB"}
    pipeline = ChronosPipeline.from_pretrained(
        "amazon/chronos-t5-base",
        device_map="auto",
        dtype=torch.float16,
        max_memory=max_mem,
    )

    total = len(indices)
    log.info(f"Will run inference on {total} series ( {CONFIG['NUM_SAMPLES']} samples each )")

    # --------------------------------------------------------------
    # 3️⃣  Inference loop (checkpoint every CONFIG["CHECKPOINT_INTERVAL"])
    # --------------------------------------------------------------
    for i, idx in enumerate(indices):
        if shutdown_manager.shutdown_requested:
            log.info("Graceful shutdown requested → breaking loop")
            break

        if tracker.should_process(idx):
            process_one_series(pipeline, contexts[i], futures[i], idx, tracker)

        # checkpoint
        if (i + 1) % CONFIG["CHECKPOINT_INTERVAL"] == 0 or (i + 1) == total:
            tracker.save(i + 1)

        # light progress output
        if (i + 1) % 100 == 0:
            log.info(f"Progress: {i + 1}/{total} ({(i + 1) / total * 100:.1f}%)")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # --------------------------------------------------------------
    # 4️⃣  Final aggregation & dump results
    # --------------------------------------------------------------
    if tracker.wqls and tracker.mases:
        wql_geo = gmean(np.maximum(tracker.wqls, 1e-9))
        mase_geo = gmean(np.maximum(tracker.mases, 1e-9))

        results_path = Path(CONFIG["RESULTS_DIR"]) / "m4_hourly_results.pkl"
        summary = {
            "wql_geo": wql_geo,
            "mase_geo": mase_geo,
            "series_processed": len(tracker.wqls),
            "total_series": total,
            "num_samples_per_series": CONFIG["NUM_SAMPLES"],
            "timestamp": time.time(),
        }
        with open(results_path, "wb") as f:
            pickle.dump(summary, f)

        # Remove checkpoint – we have a clean, successful run
        if tracker.checkpoint_file.exists():
            tracker.checkpoint_file.unlink()

        log.info("=== FINAL METRICS ===")
        log.info(f"WQL (geometric mean) : {wql_geo:.6f}")
        log.info(f"MASE (geometric mean): {mase_geo:.6f}")
        log.info(f"Results written to {results_path}")
    else:
        log.warning("No metrics were collected – something went wrong earlier")

    log.info("=== Evaluation finished ===")


# ----------------------------------------------------------------------
# 🔟  ENTRY POINT
# ----------------------------------------------------------------------
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log.info("KeyboardInterrupt caught – exiting gracefully")
    except Exception as exc:
        log.error(f"Unexpected fatal error: {exc}")
        sys.exit(1)
    finally:
        log.info("Script execution completed")
# nohup python3 chronos_m4_hourly.py  > eval_output.log 2>&1 &
# kill %1