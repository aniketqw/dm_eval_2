# #!/usr/bin/env python
# # -*- coding: utf-8 -*-

# """
# chronos_common.py

# Utility functions that are *identical* for every Chronos evaluation
# (script‑agnostic).  Import this module from any experiment script
# (e.g. evaluate_chronos.py) and you get:

# * logging helper
# * graceful‑shutdown handler
# * checkpoint / progress‑tracker
# * WQL and MASE metric implementations
# * generic inference loop (runs the model series‑by‑series)
# * generic CSV‑/Parquet‑loader that works for all Chronos "long‑format"
#   datasets (item_id, timestamp, target).
# """

# import logging, pickle, signal, sys, time
# from pathlib import Path
# from typing import List, Tuple

# import numpy as np
# import pandas as pd
# import torch
# from scipy.stats import gmean
# from chronos import ChronosPipeline


# # ----------------------------------------------------------------------
# # 0️⃣  LOGGING (shared)
# # ----------------------------------------------------------------------
# def setup_logging() -> logging.Logger:
#     log_dir = Path("logs")
#     log_dir.mkdir(exist_ok=True)
#     log_file = log_dir / f"chronos_eval_{int(time.time())}.log"
#     logging.basicConfig(
#         level=logging.INFO,
#         format="%(asctime)s | %(levelname)-8s | %(message)s",
#         datefmt="%Y-%m-%d %H:%M:%S",
#         handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
#     )
#     return logging.getLogger(__name__)


# # ----------------------------------------------------------------------
# # 1️⃣  GRACEFUL SHUTDOWN (shared)
# # ----------------------------------------------------------------------
# class GracefulShutdown:
#     """Catches SIGINT / SIGTERM and flips a flag that the loop can inspect."""
#     def __init__(self, logger: logging.Logger):
#         self.logger = logger
#         self.shutdown_requested = False
#         signal.signal(signal.SIGINT, self._handler)
#         signal.signal(signal.SIGTERM, self._handler)

#     def _handler(self, signum, frame):
#         self.logger.info(f"Received signal {signum}. Initiating graceful shutdown …")
#         self.shutdown_requested = True


# # ----------------------------------------------------------------------
# # 2️⃣  CHECKPOINT / PROGRESS TRACKER (shared)
# # ----------------------------------------------------------------------
# class ProgressTracker:
#     """
#     Persists three lists (processed indices, per‑series WQL, per‑series MASE)
#     together with a small "processed set".  A pickle file is written every
#     `checkpoint_interval` steps.
#     """
#     def __init__(self, results_dir: str, checkpoint_interval: int, logger: logging.Logger):
#         self.logger = logger
#         self.results_dir = Path(results_dir)
#         self.results_dir.mkdir(exist_ok=True)
#         self.checkpoint_interval = checkpoint_interval
#         self.checkpoint_file = self.results_dir / "progress_checkpoint.pkl"

#         self.processed: set[int] = set()
#         self.wqls: List[float] = []
#         self.mases: List[float] = []

#         self._load()

#     # --------------------------------------------------------------
#     def _load(self):
#         if self.checkpoint_file.exists():
#             try:
#                 with open(self.checkpoint_file, "rb") as f:
#                     data = pickle.load(f)
#                 self.processed = set(data.get("processed", []))
#                 self.wqls = data.get("wqls", [])
#                 self.mases = data.get("mases", [])
#                 self.logger.info(f"Loaded checkpoint – {len(self.processed)} series already done")
#             except Exception as e:
#                 self.logger.warning(f"Could not read checkpoint: {e}")

#     # --------------------------------------------------------------
#     def save(self, last_index: int):
#         try:
#             payload = {
#                 "processed": list(self.processed),
#                 "wqls": self.wqls,
#                 "mases": self.mases,
#                 "last_index": last_index,
#                 "timestamp": time.time(),
#             }
#             with open(self.checkpoint_file, "wb") as f:
#                 pickle.dump(payload, f)
#             self.logger.info(f"Checkpoint saved after processing index {last_index}")
#         except Exception as e:
#             self.logger.error(f"Failed to write checkpoint: {e}")

#     # --------------------------------------------------------------
#     def should_process(self, idx: int) -> bool:
#         return idx not in self.processed

#     # --------------------------------------------------------------
#     def add(self, idx: int, wql_val: float, mase_val: float):
#         self.processed.add(idx)
#         if not np.isnan(wql_val):
#             self.wqls.append(wql_val)
#         if not np.isnan(mase_val):
#             self.mases.append(mase_val)


# # ----------------------------------------------------------------------
# # 3️⃣  METRICS (shared) -------------------------------------------------
# def wql(actual: np.ndarray, pred: np.ndarray, qs: List[float]) -> float:
#     """Weighted Quantile Loss – identical to the Chronos paper."""
#     if np.sum(np.abs(actual)) == 0:
#         return np.nan
#     actual = actual.reshape(-1, 1) if actual.ndim == 1 else actual
#     losses = [
#         np.maximum(q * (actual - pred[:, i]), (q - 1) * (actual - pred[:, i]))
#         for i, q in enumerate(qs)
#     ]
#     return np.sum(np.stack(losses, axis=1)) / np.sum(np.abs(actual))


# def mase(actual: np.ndarray, pred: np.ndarray, seasonal_err: np.ndarray) -> float:
#     """Mean Absolute Scaled Error."""
#     mae = np.mean(np.abs(actual - pred))
#     denom = np.mean(np.abs(seasonal_err))
#     return mae / denom if denom != 0 else (np.inf if mae > 0 else 0.0)


# # ----------------------------------------------------------------------
# # 4️⃣  GENERIC DATA‑LOADER (shared) ------------------------------------
# def load_dataset(
#     csv_url: str, 
#     parquet_path: str, 
#     pred_len: int,  
#     logger: logging.Logger
# ) -> Tuple[List[np.ndarray], List[np.ndarray], List[int]]:
#     """
#     Reads a **Chronos long‑format** CSV (or a cached parquet) and returns three
#     parallel lists required by the inference loop:

#     * `contexts` – all observations **except** the last `pred_len`
#     * `futures` – the last `pred_len` values
#     * `indices` – integer identifiers (0 … N‑1)

#     The function is completely data‑agnostic: it only assumes the three core
#     columns `item_id`, `timestamp`, `target`.  If the source uses a different
#     column name for the identifier (e.g. `id` instead of `item_id`) you can
#     rename it *inside* this function – the calling script does not need to know.
#     """
#     # --------------------------------------------------------------
#     # 1️⃣  Prefer the local parquet (fast, no network)
#     # --------------------------------------------------------------
#     if Path(parquet_path).exists():
#         logger.info(f"✅ Loading data from local Parquet: {parquet_path}")
#         df = pd.read_parquet(parquet_path)
#     else:
#         logger.info(f"Parquet not found – downloading CSV from {csv_url}")
#         df = pd.read_csv(csv_url, parse_dates=["timestamp"])

#     # --------------------------------------------------------------
#     # 2️⃣  Normalise column names (some Chronos repos use `id` instead of `item_id`)
#     # --------------------------------------------------------------
#     if "id" in df.columns and "item_id" not in df.columns:
#         df = df.rename(columns={"id": "item_id"})

#     # --------------------------------------------------------------
#     # 3️⃣  Basic sanity check
#     # --------------------------------------------------------------
#     required = {"item_id", "timestamp", "target"}
#     missing = required - set(df.columns)
#     if missing:
#         raise RuntimeError(f"The dataset is missing required columns: {missing}")

#     logger.info(f"Input dataframe shape : {df.shape}")
#     logger.info("First 5 rows:\n" + df.head().to_string(index=False))

#     # --------------------------------------------------------------
#     # 4️⃣  Ensure each series is sorted chronologically
#     # --------------------------------------------------------------
#     df = df.sort_values(["item_id", "timestamp"]).reset_index(drop=True)

#     # --------------------------------------------------------------
#     # 5️⃣  Build the three parallel lists
#     # --------------------------------------------------------------
#     contexts, futures, indices = [], [], []
#     for i, (item_id, grp) in enumerate(df.groupby("item_id")):
#         target_arr = grp["target"].astype(np.float32).values

#         # Use the pred_len parameter passed to the function
#         if len(target_arr) > pred_len:
#             ctx = target_arr[:-pred_len]
#             fut = target_arr[-pred_len:]
#         else:
#             pad = pred_len - len(target_arr)
#             ctx = target_arr
#             fut = np.concatenate([target_arr[-pad:], np.zeros(pad)]) if pad > 0 else np.zeros(pred_len)
#             logger.warning(
#                 f"Series {i} ({item_id}) shorter than prediction length – padded with zeros"
#             )

#         contexts.append(ctx)
#         futures.append(fut)
#         indices.append(i)

#     logger.info(f"Prepared {len(indices)} series for inference")
#     return contexts, futures, indices


# # ----------------------------------------------------------------------
# # 5️⃣  INFERENCE LOOP (shared) -----------------------------------------
# def run_inference(
#     pipeline: ChronosPipeline,
#     contexts: List[np.ndarray],
#     futures: List[np.ndarray],
#     tracker: ProgressTracker,
#     shutdown: GracefulShutdown,
#     logger: logging.Logger,
#     config: dict,
# ) -> None:
#     """
#     Walk through the series, call `pipeline.predict`, compute WQL/MASE,
#     store the results in `tracker`, and checkpoint every
#     `config["CHECKPOINT_INTERVAL"]` steps.
#     """
#     total = len(contexts)
#     logger.info(f"Will run inference on {total} series ( {config['NUM_SAMPLES']} samples each )")

#     for i, (ctx, fut) in enumerate(zip(contexts, futures)):
#         if shutdown.shutdown_requested:
#             logger.info("Graceful shutdown requested → breaking loop")
#             break

#         if tracker.should_process(i):
#             # ------------------------------------------------------------------
#             # 1️⃣  Build a 1‑dim tensor (batch‑size = 1)
#             # ------------------------------------------------------------------
#             tensor = torch.from_numpy(ctx).unsqueeze(0)   # shape (1, len(ctx))

#             # ------------------------------------------------------------------
#             # 2️⃣  Model prediction (no gradient tracking)
#             # ------------------------------------------------------------------
#             with torch.no_grad():
#                 samples = pipeline.predict(
#                     tensor,
#                     prediction_length=config["PREDICTION_LENGTH"],
#                     num_samples=config["NUM_SAMPLES"],
#                 )
#             # samples: (1, num_samples, prediction_length)
#             samples_np = samples.cpu().numpy().squeeze(0)   # (num_samples, pred_len)

#             # ------------------------------------------------------------------
#             # 3️⃣  Quantile aggregation
#             # ------------------------------------------------------------------
#             quantile_preds = np.column_stack(
#                 [np.quantile(samples_np, q, axis=0) for q in config["QUANTILE_LEVELS"]]
#             )   # (pred_len, n_quantiles)

#             # Median is the 5‑th column (index 4) because we have 9 quantiles
#             median_pred = quantile_preds[:, 4]

#             # ------------------------------------------------------------------
#             # 4️⃣  Compute metrics
#             # ------------------------------------------------------------------
#             w = wql(fut, quantile_preds, config["QUANTILE_LEVELS"])
#             seasonal_err = np.diff(ctx) if len(ctx) > 1 else np.array([1.0])
#             m = mase(fut, median_pred, seasonal_err)

#             # ------------------------------------------------------------------
#             # 5️⃣  Store results
#             # ------------------------------------------------------------------
#             tracker.add(i, w, m)

#             if (i + 1) % 100 == 0:
#                 logger.info(f"Series {i} → WQL={w:.4f}, MASE={m:.4f}")

#         # ------------------------------------------------------------------
#         # 6️⃣  Periodic checkpoint
#         # ------------------------------------------------------------------
#         if (i + 1) % config["CHECKPOINT_INTERVAL"] == 0 or (i + 1) == total:
#             tracker.save(i + 1)

#         # ------------------------------------------------------------------
#         # 7️⃣  Light progress output + optional GPU memory cleanup
#         # ------------------------------------------------------------------
#         if (i + 1) % 100 == 0:
#             logger.info(f"Progress: {i + 1}/{total} ({(i + 1) / total * 100:.1f}%)")
#             if torch.cuda.is_available():
#                 torch.cuda.empty_cache()


# # ----------------------------------------------------------------------
# # 6️⃣  FINAL METRICS AGGREGATION (shared)
# # ----------------------------------------------------------------------
# def write_final_summary(tracker: ProgressTracker, logger: logging.Logger, config: dict) -> None:
#     if not (tracker.wqls and tracker.mases):
#         logger.warning("No metrics were collected – something went wrong earlier")
#         return

#     wql_geo = gmean(np.maximum(tracker.wqls, 1e-9))
#     mase_geo = gmean(np.maximum(tracker.mases, 1e-9))

#     results_path = tracker.results_dir / "chronos_results.pkl"
#     summary = {
#         "wql_geo": wql_geo,
#         "mase_geo": mase_geo,
#         "series_processed": len(tracker.wqls),
#         "total_series": len(tracker.wqls),   # same as processed after a clean run
#         "num_samples_per_series": config["NUM_SAMPLES"],
#         "timestamp": time.time(),
#     }
#     with open(results_path, "wb") as f:
#         pickle.dump(summary, f)

#     # delete the checkpoint – we have a clean successful run
#     if tracker.checkpoint_file.exists():
#         tracker.checkpoint_file.unlink()

#     logger.info("=== FINAL METRICS ===")
#     logger.info(f"WQL (geometric mean) : {wql_geo:.6f}")
#     logger.info(f"MASE (geometric mean): {mase_geo:.6f}")
#     logger.info(f"Results written to {results_path}")

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
chronos_common.py

Utility functions that are *identical* for every Chronos evaluation
(script‑agnostic).  Import this module from any experiment script
(e.g. evaluate_chronos.py) and you get:

* logging helper
* graceful‑shutdown handler
* checkpoint / progress‑tracker
* WQL and MASE metric implementations
* generic inference loop (runs the model series‑by‑series)
* generic CSV‑/Parquet‑loader that works for all Chronos "long‑format"
  datasets (item_id, timestamp, target).
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
# 0️⃣  LOGGING (shared)
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
# 1️⃣  GRACEFUL SHUTDOWN (shared)
# ----------------------------------------------------------------------
class GracefulShutdown:
    """Catches SIGINT / SIGTERM and flips a flag that the loop can inspect."""
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.shutdown_requested = False
        signal.signal(signal.SIGINT, self._handler)
        signal.signal(signal.SIGTERM, self._handler)

    def _handler(self, signum, frame):
        self.logger.info(f"Received signal {signum}. Initiating graceful shutdown …")
        self.shutdown_requested = True


# ----------------------------------------------------------------------
# 2️⃣  CHECKPOINT / PROGRESS TRACKER (shared)
# ----------------------------------------------------------------------
class ProgressTracker:
    """
    Persists metrics (processed indices, per‑series WQL, per‑series MASE)
    together with BASELINE metrics for normalization.
    A pickle file is written every `checkpoint_interval` steps.
    """
    def __init__(self, results_dir: str, checkpoint_interval: int, logger: logging.Logger):
        self.logger = logger
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.checkpoint_interval = checkpoint_interval
        self.checkpoint_file = self.results_dir / "progress_checkpoint.pkl"

        self.processed: set[int] = set()
        self.wqls: List[float] = []
        self.mases: List[float] = []
        
        # Add baseline metrics for normalization
        self.baseline_wqls: List[float] = []
        self.baseline_mases: List[float] = []

        self._load()

    # --------------------------------------------------------------
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

    # --------------------------------------------------------------
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

    # --------------------------------------------------------------
    def should_process(self, idx: int) -> bool:
        return idx not in self.processed

    # --------------------------------------------------------------
    def add(self, idx: int, wql_val: float, mase_val: float, baseline_wql: float, baseline_mase: float):
        """Add metrics for both model and baseline"""
        self.processed.add(idx)
        if not np.isnan(wql_val):
            self.wqls.append(wql_val)
        if not np.isnan(mase_val):
            self.mases.append(mase_val)
        if not np.isnan(baseline_wql):
            self.baseline_wqls.append(baseline_wql)
        if not np.isnan(baseline_mase):
            self.baseline_mases.append(baseline_mase)


# ----------------------------------------------------------------------
# 3️⃣  METRICS (shared) -------------------------------------------------

def seasonal_naive_forecast(context: np.ndarray, pred_len: int, season_length: int) -> np.ndarray:
    """
    Generate seasonal naive forecast.
    For yearly data, season_length=1 (uses last observation).
    For monthly data, season_length=12
    For quarterly data, season_length=4
    """
    if season_length >= len(context):
        # If not enough history, just repeat the last value
        return np.full(pred_len, context[-1])
    
    forecast = []
    for i in range(pred_len):
        # Look back season_length steps
        idx = len(context) - season_length + (i % season_length)
        if idx < 0:
            idx = -1
        forecast.append(context[idx])
    return np.array(forecast)


def wql(actual: np.ndarray, pred: np.ndarray, qs: List[float]) -> float:
    """Weighted Quantile Loss – identical to the Chronos paper."""
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
# 4️⃣  GENERIC DATA‑LOADER (shared) ------------------------------------
def load_dataset(
    csv_url: str, 
    parquet_path: str, 
    pred_len: int,  
    logger: logging.Logger
) -> Tuple[List[np.ndarray], List[np.ndarray], List[int]]:
    """
    Reads a **Chronos long‑format** CSV (or a cached parquet) and returns three
    parallel lists required by the inference loop:

    * `contexts` – all observations **except** the last `pred_len`
    * `futures` – the last `pred_len` values
    * `indices` – integer identifiers (0 … N‑1)

    The function is completely data‑agnostic: it only assumes the three core
    columns `item_id`, `timestamp`, `target`.  If the source uses a different
    column name for the identifier (e.g. `id` instead of `item_id`) you can
    rename it *inside* this function – the calling script does not need to know.
    """
    # --------------------------------------------------------------
    # 1️⃣  Prefer the local parquet (fast, no network)
    # --------------------------------------------------------------
    if Path(parquet_path).exists():
        logger.info(f"✅ Loading data from local Parquet: {parquet_path}")
        df = pd.read_parquet(parquet_path)
    else:
        logger.info(f"Parquet not found – downloading CSV from {csv_url}")
        df = pd.read_csv(csv_url, parse_dates=["timestamp"])

    # --------------------------------------------------------------
    # 2️⃣  Normalise column names (some Chronos repos use `id` instead of `item_id`)
    # --------------------------------------------------------------
    if "id" in df.columns and "item_id" not in df.columns:
        df = df.rename(columns={"id": "item_id"})

    # --------------------------------------------------------------
    # 3️⃣  Basic sanity check
    # --------------------------------------------------------------
    required = {"item_id", "timestamp", "target"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"The dataset is missing required columns: {missing}")

    logger.info(f"Input dataframe shape : {df.shape}")
    logger.info("First 5 rows:\n" + df.head().to_string(index=False))

    # --------------------------------------------------------------
    # 4️⃣  Ensure each series is sorted chronologically
    # --------------------------------------------------------------
    df = df.sort_values(["item_id", "timestamp"]).reset_index(drop=True)

    # --------------------------------------------------------------
    # 5️⃣  Build the three parallel lists
    # --------------------------------------------------------------
    contexts, futures, indices = [], [], []
    for i, (item_id, grp) in enumerate(df.groupby("item_id")):
        target_arr = grp["target"].astype(np.float32).values

        # Use the pred_len parameter passed to the function
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
# 5️⃣  INFERENCE LOOP (shared) -----------------------------------------
def run_inference(
    pipeline: ChronosPipeline,
    contexts: List[np.ndarray],
    futures: List[np.ndarray],
    tracker: ProgressTracker,
    shutdown: GracefulShutdown,
    logger: logging.Logger,
    config: dict,
) -> None:
    """
    Walk through the series, call `pipeline.predict`, compute WQL/MASE,
    store the results in `tracker`, and checkpoint every
    `config["CHECKPOINT_INTERVAL"]` steps.
    
    **CRITICAL**: Also compute Seasonal Naive baseline for normalization!
    """
    total = len(contexts)
    logger.info(f"Will run inference on {total} series ( {config['NUM_SAMPLES']} samples each )")
    
    # Get season length from config (default to 1 for yearly data)
    season_length = config.get("SEASON_LENGTH", 1)
    logger.info(f"Using season_length={season_length} for Seasonal Naive baseline")

    for i, (ctx, fut) in enumerate(zip(contexts, futures)):
        if shutdown.shutdown_requested:
            logger.info("Graceful shutdown requested → breaking loop")
            break

        if tracker.should_process(i):
            # ------------------------------------------------------------------
            # 1️⃣  Build a 1‑dim tensor (batch‑size = 1)
            # ------------------------------------------------------------------
            tensor = torch.from_numpy(ctx).unsqueeze(0)   # shape (1, len(ctx))

            # ------------------------------------------------------------------
            # 2️⃣  Model prediction (no gradient tracking)
            # ------------------------------------------------------------------
            with torch.no_grad():
                samples = pipeline.predict(
                    tensor,
                    prediction_length=config["PREDICTION_LENGTH"],
                    num_samples=config["NUM_SAMPLES"],
                )
            # samples: (1, num_samples, prediction_length)
            samples_np = samples.cpu().numpy().squeeze(0)   # (num_samples, pred_len)

            # ------------------------------------------------------------------
            # 3️⃣  Quantile aggregation for MODEL
            # ------------------------------------------------------------------
            quantile_preds = np.column_stack(
                [np.quantile(samples_np, q, axis=0) for q in config["QUANTILE_LEVELS"]]
            )   # (pred_len, n_quantiles)

            # Median is the 5‑th column (index 4) because we have 9 quantiles
            median_pred = quantile_preds[:, 4]

            # ------------------------------------------------------------------
            # 4️⃣  Compute SEASONAL NAIVE BASELINE
            # ------------------------------------------------------------------
            naive_forecast = seasonal_naive_forecast(ctx, config["PREDICTION_LENGTH"], season_length)
            
            # For WQL baseline, create quantile predictions (all identical for naive)
            naive_quantile_preds = np.tile(naive_forecast.reshape(-1, 1), (1, len(config["QUANTILE_LEVELS"])))
            
            # ------------------------------------------------------------------
            # 5️⃣  Compute metrics for MODEL
            # ------------------------------------------------------------------
            model_wql = wql(fut, quantile_preds, config["QUANTILE_LEVELS"])
            seasonal_err = np.diff(ctx) if len(ctx) > 1 else np.array([1.0])
            model_mase = mase(fut, median_pred, seasonal_err)

            # ------------------------------------------------------------------
            # 6️⃣  Compute metrics for BASELINE
            # ------------------------------------------------------------------
            baseline_wql = wql(fut, naive_quantile_preds, config["QUANTILE_LEVELS"])
            baseline_mase = mase(fut, naive_forecast, seasonal_err)

            # ------------------------------------------------------------------
            # 7️⃣  Store results (both model and baseline)
            # ------------------------------------------------------------------
            tracker.add(i, model_wql, model_mase, baseline_wql, baseline_mase)

            if (i + 1) % 100 == 0:
                logger.info(f"Series {i} → Model WQL={model_wql:.4f}, Model MASE={model_mase:.4f} | "
                          f"Baseline WQL={baseline_wql:.4f}, Baseline MASE={baseline_mase:.4f}")

        # ------------------------------------------------------------------
        # 8️⃣  Periodic checkpoint
        # ------------------------------------------------------------------
        if (i + 1) % config["CHECKPOINT_INTERVAL"] == 0 or (i + 1) == total:
            tracker.save(i + 1)

        # ------------------------------------------------------------------
        # 9️⃣  Light progress output + optional GPU memory cleanup
        # ------------------------------------------------------------------
        if (i + 1) % 100 == 0:
            logger.info(f"Progress: {i + 1}/{total} ({(i + 1) / total * 100:.1f}%)")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


# ----------------------------------------------------------------------
# 6️⃣  FINAL METRICS AGGREGATION (shared) - WITH NORMALIZATION!
# ----------------------------------------------------------------------
def write_final_summary(tracker: ProgressTracker, logger: logging.Logger, config: dict) -> None:
    if not (tracker.wqls and tracker.mases):
        logger.warning("No metrics were collected – something went wrong earlier")
        return

    # Compute ABSOLUTE geometric means
    wql_geo_abs = gmean(np.maximum(tracker.wqls, 1e-9))
    mase_geo_abs = gmean(np.maximum(tracker.mases, 1e-9))
    
    baseline_wql_geo = gmean(np.maximum(tracker.baseline_wqls, 1e-9))
    baseline_mase_geo = gmean(np.maximum(tracker.baseline_mases, 1e-9))

    # Compute RELATIVE scores (normalized by baseline) - THIS IS WHAT THE PAPER REPORTS!
    wql_relative = wql_geo_abs / baseline_wql_geo
    mase_relative = mase_geo_abs / baseline_mase_geo

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

    # delete the checkpoint – we have a clean successful run
    if tracker.checkpoint_file.exists():
        tracker.checkpoint_file.unlink()

    logger.info("=" * 70)
    logger.info("=== FINAL METRICS ===")
    logger.info("=" * 70)
    logger.info(f"MODEL (absolute):")
    logger.info(f"  WQL  (geometric mean) : {wql_geo_abs:.6f}")
    logger.info(f"  MASE (geometric mean) : {mase_geo_abs:.6f}")
    logger.info("")
    logger.info(f"BASELINE (Seasonal Naive):")
    logger.info(f"  WQL  (geometric mean) : {baseline_wql_geo:.6f}")
    logger.info(f"  MASE (geometric mean) : {baseline_mase_geo:.6f}")
    logger.info("")
    logger.info(f"RELATIVE SCORES (Model/Baseline) - **PAPER METRIC**:")
    logger.info(f"  WQL  (relative) : {wql_relative:.6f}")
    logger.info(f"  MASE (relative) : {mase_relative:.6f}")
    logger.info("=" * 70)
    logger.info(f"Results written to {results_path}")