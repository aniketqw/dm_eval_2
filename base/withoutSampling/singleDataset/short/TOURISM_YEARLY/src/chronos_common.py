#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
chronos_common.py - SIMPLEST POSSIBLE CORRECT IMPLEMENTATION

Going back to basics with the EXACT formula from the paper.
"""

import logging, pickle, signal, sys, time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.stats import gmean
from chronos import ChronosPipeline


# ----------------------------------------------------------------------
# 0️⃣  LOGGING
# ----------------------------------------------------------------------
def setup_logging() -> logging.Logger:
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


# ----------------------------------------------------------------------
# 1️⃣  GRACEFUL SHUTDOWN
# ----------------------------------------------------------------------
class GracefulShutdown:
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.shutdown_requested = False
        signal.signal(signal.SIGINT, self._handler)
        signal.signal(signal.SIGTERM, self._handler)

    def _handler(self, signum, frame):
        self.logger.info(f"Received signal {signum}. Initiating graceful shutdown …")
        self.shutdown_requested = True


# ----------------------------------------------------------------------
# 2️⃣  PROGRESS TRACKER
# ----------------------------------------------------------------------
class ProgressTracker:
    def __init__(self, results_dir: str, checkpoint_interval: int, logger: logging.Logger):
        self.logger = logger
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.checkpoint_interval = checkpoint_interval
        self.checkpoint_file = self.results_dir / "progress_checkpoint.pkl"

        self.processed: set[int] = set()
        self.wqls: List[float] = []
        self.mases: List[float] = []
        self.baseline_wqls: List[float] = []
        self.baseline_mases: List[float] = []

        self._load()

    def _load(self):
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, "rb") as f:
                    data = pickle.load(f)
                self.processed = set(data.get("processed", []))
                self.wqls = data.get("wqls", [])
                self.mases = data.get("mases", [])
                self.baseline_wqls = data.get("baseline_wqls", [])
                self.baseline_mases = data.get("baseline_mases", [])
                self.logger.info(f"Loaded checkpoint – {len(self.processed)} series already done")
            except Exception as e:
                self.logger.warning(f"Could not read checkpoint: {e}")

    def save(self, last_index: int):
        try:
            payload = {
                "processed": list(self.processed),
                "wqls": self.wqls,
                "mases": self.mases,
                "baseline_wqls": self.baseline_wqls,
                "baseline_mases": self.baseline_mases,
                "last_index": last_index,
                "timestamp": time.time(),
            }
            with open(self.checkpoint_file, "wb") as f:
                pickle.dump(payload, f)
            self.logger.info(f"Checkpoint saved after processing index {last_index}")
        except Exception as e:
            self.logger.error(f"Failed to write checkpoint: {e}")

    def should_process(self, idx: int) -> bool:
        return idx not in self.processed

    def add(self, idx: int, wql_val: float, mase_val: float, baseline_wql: float, baseline_mase: float):
        self.processed.add(idx)
        
        if not np.isnan(wql_val) and wql_val > 0:
            self.wqls.append(wql_val)
            
        if not np.isnan(mase_val) and mase_val > 0:
            self.mases.append(mase_val)
            
        if not np.isnan(baseline_wql) and baseline_wql > 0:
            self.baseline_wqls.append(baseline_wql)
            
        if not np.isnan(baseline_mase) and baseline_mase > 0:
            self.baseline_mases.append(baseline_mase)


# ----------------------------------------------------------------------
# 3️⃣  METRICS
# ----------------------------------------------------------------------

def seasonal_naive_forecast(context: np.ndarray, pred_len: int, season_length: int) -> np.ndarray:
    """Generate seasonal naive forecast."""
    if season_length >= len(context):
        return np.full(pred_len, context[-1])
    
    forecast = []
    for i in range(pred_len):
        idx = len(context) - season_length + (i % season_length)
        if idx < 0:
            idx = -1
        forecast.append(context[idx])
    return np.array(forecast)


def wql(actual: np.ndarray, pred: np.ndarray, qs: List[float]) -> float:
    """Weighted Quantile Loss."""
    if np.sum(np.abs(actual)) == 0:
        return np.nan
    
    actual = actual.reshape(-1, 1) if actual.ndim == 1 else actual
    
    wql_sum = 0.0
    for i, q in enumerate(qs):
        errors = actual - pred[:, i:i+1]
        ql = np.where(errors >= 0, q * errors, (q - 1) * errors)
        wql_q = 2.0 * np.sum(ql) / np.sum(np.abs(actual))
        wql_sum += wql_q
    
    return wql_sum / len(qs)


def mase(actual: np.ndarray, pred: np.ndarray, context: np.ndarray, season_length: int) -> float:
    """
    Mean Absolute Scaled Error - SIMPLEST CORRECT IMPLEMENTATION
    
    From paper Appendix D:
    MASE = mean(|forecast_error|) / mean(|seasonal_naive_error_on_history|)
    
    For yearly data (S=1): seasonal_naive_error = |x_t - x_{t+1}|
    """
    # MAE of the forecast
    mae_forecast = np.mean(np.abs(actual - pred))
    
    # MAE of seasonal naive on the historical context
    # For S=1 (yearly): this is just |x_t - x_{t+1}|
    if len(context) <= season_length:
        # Not enough data, use simple differences
        seasonal_errors = np.abs(np.diff(context))
    else:
        # Seasonal differences: |x_t - x_{t+S}|
        seasonal_errors = np.abs(context[:-season_length] - context[season_length:])
    
    mae_seasonal = np.mean(seasonal_errors) if len(seasonal_errors) > 0 else 1.0
    
    if mae_seasonal == 0:
        return np.inf if mae_forecast > 0 else 0.0
    
    return mae_forecast / mae_seasonal


# ----------------------------------------------------------------------
# 4️⃣  DATA LOADER
# ----------------------------------------------------------------------
def load_dataset(
    csv_url: str, 
    parquet_path: str, 
    pred_len: int,  
    logger: logging.Logger
) -> Tuple[List[np.ndarray], List[np.ndarray], List[int]]:
    """Load dataset."""
    if Path(parquet_path).exists():
        logger.info(f"✅ Loading data from local Parquet: {parquet_path}")
        df = pd.read_parquet(parquet_path)
    else:
        logger.info(f"Parquet not found – downloading CSV from {csv_url}")
        df = pd.read_csv(csv_url, parse_dates=["timestamp"])

    if "id" in df.columns and "item_id" not in df.columns:
        df = df.rename(columns={"id": "item_id"})

    required = {"item_id", "timestamp", "target"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"The dataset is missing required columns: {missing}")

    logger.info(f"Input dataframe shape : {df.shape}")
    logger.info("First 5 rows:\n" + df.head().to_string(index=False))

    df = df.sort_values(["item_id", "timestamp"]).reset_index(drop=True)

    contexts, futures, indices = [], [], []
    for i, (item_id, grp) in enumerate(df.groupby("item_id")):
        target_arr = grp["target"].astype(np.float32).values

        if len(target_arr) > pred_len:
            ctx = target_arr[:-pred_len]
            fut = target_arr[-pred_len:]
        else:
            pad = pred_len - len(target_arr)
            ctx = target_arr
            fut = np.concatenate([target_arr[-pad:], np.zeros(pad)]) if pad > 0 else np.zeros(pred_len)
            logger.warning(
                f"Series {i} ({item_id}) shorter than prediction length – padded with zeros"
            )

        contexts.append(ctx)
        futures.append(fut)
        indices.append(i)

    logger.info(f"Prepared {len(indices)} series for inference")
    return contexts, futures, indices


# ----------------------------------------------------------------------
# 5️⃣  INFERENCE LOOP
# ----------------------------------------------------------------------
def run_inference(
    pipeline: ChronosPipeline,
    contexts: List[np.ndarray],
    futures: List[np.ndarray],
    tracker: ProgressTracker,
    shutdown: GracefulShutdown,
    logger: logging.Logger,
    config: dict,
) -> None:
    """Run inference loop."""
    total = len(contexts)
    logger.info(f"Will run inference on {total} series ( {config['NUM_SAMPLES']} samples each )")
    
    season_length = config.get("SEASON_LENGTH", 1)
    pred_len = config["PREDICTION_LENGTH"]
    logger.info(f"Using season_length={season_length} for Seasonal Naive baseline")

    for i, (ctx, fut) in enumerate(zip(contexts, futures)):
        if shutdown.shutdown_requested:
            logger.info("Graceful shutdown requested → breaking loop")
            break

        if tracker.should_process(i):
            tensor = torch.from_numpy(ctx).unsqueeze(0)

            with torch.no_grad():
                samples = pipeline.predict(
                    tensor,
                    prediction_length=pred_len,
                    num_samples=config["NUM_SAMPLES"],
                )
            samples_np = samples.cpu().numpy().squeeze(0)

            quantile_preds = np.column_stack(
                [np.quantile(samples_np, q, axis=0) for q in config["QUANTILE_LEVELS"]]
            )

            median_pred = quantile_preds[:, 4]

            naive_forecast = seasonal_naive_forecast(ctx, pred_len, season_length)
            naive_quantile_preds = np.tile(naive_forecast.reshape(-1, 1), (1, len(config["QUANTILE_LEVELS"])))
            
            model_wql = wql(fut, quantile_preds, config["QUANTILE_LEVELS"])
            model_mase = mase(fut, median_pred, ctx, season_length)

            baseline_wql = wql(fut, naive_quantile_preds, config["QUANTILE_LEVELS"])
            baseline_mase = mase(fut, naive_forecast, ctx, season_length)

            tracker.add(i, model_wql, model_mase, baseline_wql, baseline_mase)

            if (i + 1) % 100 == 0:
                logger.info(f"Series {i} → Model WQL={model_wql:.4f}, Model MASE={model_mase:.4f} | "
                          f"Baseline WQL={baseline_wql:.4f}, Baseline MASE={baseline_mase:.4f}")

        if (i + 1) % config["CHECKPOINT_INTERVAL"] == 0 or (i + 1) == total:
            tracker.save(i + 1)

        if (i + 1) % 100 == 0:
            logger.info(f"Progress: {i + 1}/{total} ({(i + 1) / total * 100:.1f}%)")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


# ----------------------------------------------------------------------
# 6️⃣  FINAL METRICS
# ----------------------------------------------------------------------
def write_final_summary(tracker: ProgressTracker, logger: logging.Logger, config: dict) -> None:
    """Write final summary."""
    if not (tracker.wqls and tracker.mases):
        logger.warning("No metrics were collected")
        return

    wql_geo_abs = gmean(np.maximum(tracker.wqls, 1e-9))
    mase_geo_abs = gmean(np.maximum(tracker.mases, 1e-9))
    
    baseline_wql_geo = gmean(np.maximum(tracker.baseline_wqls, 1e-9))
    baseline_mase_geo = gmean(np.maximum(tracker.baseline_mases, 1e-9))

    wql_relative = wql_geo_abs / baseline_wql_geo if baseline_wql_geo > 0 else np.nan
    mase_relative = mase_geo_abs / baseline_mase_geo if baseline_mase_geo > 0 else np.nan

    results_path = tracker.results_dir / "chronos_results.pkl"
    summary = {
        "wql_absolute": wql_geo_abs,
        "mase_absolute": mase_geo_abs,
        "wql_relative": wql_relative,
        "mase_relative": mase_relative,
        "baseline_wql": baseline_wql_geo,
        "baseline_mase": baseline_mase_geo,
        "series_processed": len(tracker.wqls),
        "total_series": len(tracker.wqls),
        "num_samples_per_series": config["NUM_SAMPLES"],
        "timestamp": time.time(),
    }
    with open(results_path, "wb") as f:
        pickle.dump(summary, f)

    if tracker.checkpoint_file.exists():
        tracker.checkpoint_file.unlink()

    logger.info("=" * 70)
    logger.info("=== FINAL METRICS ===")
    logger.info("=" * 70)
    logger.info(f"MODEL - ABSOLUTE SCORES:")
    logger.info(f"  WQL  (geometric mean) : {wql_geo_abs:.6f}")
    logger.info(f"  MASE (geometric mean) : {mase_geo_abs:.6f}")
    logger.info("")
    logger.info(f"BASELINE (Seasonal Naive) - ABSOLUTE SCORES:")
    logger.info(f"  WQL  (geometric mean) : {baseline_wql_geo:.6f}")
    logger.info(f"  MASE (geometric mean) : {baseline_mase_geo:.6f}")
    logger.info("")
    logger.info(f"RELATIVE SCORES (Model/Baseline):")
    logger.info(f"  WQL  (relative) : {wql_relative:.6f}")
    logger.info(f"  MASE (relative) : {mase_relative:.6f}")
    logger.info("=" * 70)
    logger.info(f"Results written to {results_path}")