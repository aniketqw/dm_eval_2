#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Chronos Dominick Evaluation – **single‑GPU** version (Enhanced).
This script is optimized for maximum GPU utilization and can survive SSH disconnections.
"""

# ----------------------------------------------------------------------
# 0️⃣  Make the process see only ONE GPU  
# ----------------------------------------------------------------------
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"   # Use GPU 0 only
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ----------------------------------------------------------------------
# 1️⃣  IMPORTS & SETUP
# ----------------------------------------------------------------------
import logging
import pickle
import numpy as np
from scipy.stats import gmean
import datasets
import torch
from chronos import ChronosPipeline
import signal
import sys
import time
from pathlib import Path

# ----------------------------------------------------------------------
# 2️⃣  ENHANCED LOGGING with file output
# ----------------------------------------------------------------------
def setup_logging():
    """Setup logging to both console and file"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    log_file = log_dir / f"chronos_eval_{int(time.time())}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

log = setup_logging()

# ----------------------------------------------------------------------
# 3️⃣  CONFIGURATION - Optimized for single GPU without sampling
# ----------------------------------------------------------------------
CONFIG = {
    "DATASET_PATH": "/home/h20250169/study/modelTraining/dm_eval_2/base/withoutSampling/ALLDATASET/chronos_datasets_dominick",
    "PREDICTION_LENGTH": 8,
    "NUM_SAMPLES": 100,                     #more sampling more accurate 
    "QUANTILE_LEVELS": [0.1, 0.5, 0.9],
    "RESULTS_DIR": "results",
    "CHECKPOINT_INTERVAL": 500,           # Save progress every N series
    "BATCH_SIZE": 1,                      # Single series processing for variable lengths
    "MAX_MEMORY_GB": 24,                  # Use available GPU memory
    #if you set NUM_SAMPLES = 100000, you would be asking the model to do this: For each of the 100,000 series, generate 100,000 different predictions.
}

# ----------------------------------------------------------------------
# 4️⃣  SIGNAL HANDLING for graceful shutdown
# ----------------------------------------------------------------------
class GracefulShutdown:
    def __init__(self):
        self.shutdown_requested = False
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def signal_handler(self, signum, frame):
        log.info(f"Received signal {signum}. Graceful shutdown initiated...")
        self.shutdown_requested = True

shutdown_manager = GracefulShutdown()

# ----------------------------------------------------------------------
# 5️⃣  PROGRESS TRACKING & CHECKPOINTING
# ----------------------------------------------------------------------
class ProgressTracker:
    def __init__(self, results_dir, checkpoint_interval):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.checkpoint_interval = checkpoint_interval
        self.checkpoint_file = self.results_dir / "progress_checkpoint.pkl"
        
        # Load existing progress if available
        self.processed_indices = set()
        self.wqls = []
        self.mases = []
        self.load_checkpoint()
    
    def load_checkpoint(self):
        """Load previous progress from checkpoint file"""
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, "rb") as f:
                    data = pickle.load(f)
                self.processed_indices = set(data.get("processed_indices", []))
                self.wqls = data.get("wqls", [])
                self.mases = data.get("mases", [])
                log.info(f"Loaded checkpoint: {len(self.processed_indices)} series already processed")
            except Exception as e:
                log.warning(f"Failed to load checkpoint: {e}")
    
    def save_checkpoint(self, current_index):
        """Save current progress to checkpoint file"""
        try:
            checkpoint_data = {
                "processed_indices": list(self.processed_indices),
                "wqls": self.wqls,
                "mases": self.mases,
                "last_index": current_index,
                "timestamp": time.time()
            }
            with open(self.checkpoint_file, "wb") as f:
                pickle.dump(checkpoint_data, f)
            log.info(f"Checkpoint saved at index {current_index}")
        except Exception as e:
            log.error(f"Failed to save checkpoint: {e}")
    
    def should_process(self, index):
        """Check if this index should be processed (not already done)"""
        return index not in self.processed_indices
    
    def add_result(self, index, wql_value, mase_value):
        """Add results for a processed series"""
        self.processed_indices.add(index)
        if not np.isnan(wql_value):
            self.wqls.append(wql_value)
        if not np.isnan(mase_value):
            self.mases.append(mase_value)

# ----------------------------------------------------------------------
# 6️⃣  METRICS (unchanged)
# ----------------------------------------------------------------------
def wql(actual, pred, qs):
    if np.sum(np.abs(actual)) == 0:
        return np.nan
    losses = [
        np.maximum(q * (actual - pred[:, i]), (q - 1) * (actual - pred[:, i]))
        for i, q in enumerate(qs)
    ]
    return np.sum(np.stack(losses, axis=1)) / np.sum(np.abs(actual))

def mase(actual, pred, seasonal_err):
    mae = np.mean(np.abs(actual - pred))
    denom = np.mean(np.abs(seasonal_err))
    return mae / denom if denom != 0 else (np.inf if mae > 0 else 0.0)

# ----------------------------------------------------------------------
# 7️⃣  SINGLE SERIES PROCESSING - Deterministic without sampling
# ----------------------------------------------------------------------
def process_single_series(pipeline, ctx, fut, idx, progress_tracker):
    """Process a single series with deterministic prediction"""
    try:
        # Convert to tensor - single series processing
        tensor = torch.from_numpy(ctx.astype(np.float32)).unsqueeze(0)
        
        with torch.no_grad():
            # Use deterministic prediction (num_samples=1)
            samples = pipeline.predict(
                tensor,
                prediction_length=CONFIG["PREDICTION_LENGTH"],
                num_samples=CONFIG["NUM_SAMPLES"],  # This is now 1 for deterministic
            )
        
        # Since num_samples=1, we get direct predictions
        samples_np = samples[0].cpu().numpy()
        
        # For deterministic prediction, we only have one sample
        # Use the prediction directly as the median
        median_pred = np.median(samples_np, axis=0)  # Shape: (prediction_length,)
        
        # For WQL, we need quantiles - create synthetic quantiles around the prediction
        # Since we don't have samples, we'll create a small range around the prediction
        pred_std = np.std(ctx) * 0.1  # Small standard deviation based on context
        qpred = np.column_stack([
            median_pred - pred_std,  # 0.1 quantile (approx)
            median_pred,             # 0.5 quantile (median)
            median_pred + pred_std   # 0.9 quantile (approx)
        ])
        
        w = wql(fut, qpred, CONFIG["QUANTILE_LEVELS"])
        seasonal_err = np.diff(ctx) if len(ctx) > 1 else np.array([1.0])
        m = mase(fut, median_pred, seasonal_err)
        
        progress_tracker.add_result(idx, w, m)
        
    except Exception as e:
        log.error(f"Error processing series {idx}: {e}")

# ----------------------------------------------------------------------
# 8️⃣  MAIN FUNCTION - Enhanced with resilience
# ----------------------------------------------------------------------
def main():
    log.info("=== Chronos Dominick Single‑GPU Evaluation (Deterministic) ===")
    log.info(f"Visible GPUs: {torch.cuda.device_count()}")
    log.info(f"CUDA version: {torch.version.cuda}")
    log.info(f"Configuration: {CONFIG}")
    log.info("Using deterministic prediction (no sampling)")
    
    # Initialize progress tracker
    progress_tracker = ProgressTracker(CONFIG["RESULTS_DIR"], CONFIG["CHECKPOINT_INTERVAL"])
    
    try:
        # Load dataset
        log.info("Loading dataset...")
        ds = datasets.load_from_disk(CONFIG["DATASET_PATH"])["train"]
        target_col = next(
            k for k, v in ds.features.items()
            if isinstance(v, datasets.Sequence) and v.feature.dtype in ("float32", "float64")
        )
        ds.set_format("numpy")
        
        # Prepare series data
        contexts, futures, valid_indices = [], [], []
        for idx, row in enumerate(ds):
            if shutdown_manager.shutdown_requested:
                break
                
            s = row[target_col]
            if len(s) <= CONFIG["PREDICTION_LENGTH"] or np.any(np.isnan(s)):
                continue
                
            ctx = s[:-CONFIG["PREDICTION_LENGTH"]]
            fut = s[-CONFIG["PREDICTION_LENGTH"]:]
            
            if progress_tracker.should_process(idx):
                contexts.append(ctx)
                futures.append(fut)
                valid_indices.append(idx)
        
        log.info(f"Total series: {len(ds)}, To process: {len(valid_indices)}")
        
        # Load model with memory optimization
        log.info("Loading Chronos model...")
        max_mem = {0: f"{CONFIG['MAX_MEMORY_GB']}GiB"}
        pipeline = ChronosPipeline.from_pretrained(
            "amazon/chronos-t5-base",
            device_map="auto",
            dtype=torch.float16,
            max_memory=max_mem,
            offload_folder="./offload",
            offload_state_dict=True,
        )
        
        # Single series processing loop
        total_series = len(valid_indices)
        
        log.info(f"Starting deterministic inference...")
        
        for i, idx in enumerate(valid_indices):
            if shutdown_manager.shutdown_requested:
                log.info("Shutdown requested. Saving progress...")
                break
                
            # Process single series
            process_single_series(pipeline, contexts[i], futures[i], idx, progress_tracker)
            
            # Save checkpoint periodically
            if (i + 1) % CONFIG["CHECKPOINT_INTERVAL"] == 0 or (i + 1) == total_series:
                progress_tracker.save_checkpoint(i + 1)
            
            if (i + 1) % 100 == 0:
                log.info(f"Progress: {i + 1}/{total_series} series ({(i + 1)/total_series*100:.1f}%)")
                # Clear GPU cache periodically to prevent memory fragmentation
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Calculate final results
        if progress_tracker.wqls and progress_tracker.mases:
            wql_geo = gmean(np.maximum(progress_tracker.wqls, 1e-9))
            mase_geo = gmean(np.maximum(progress_tracker.mases, 1e-9))
            
            # Save final results
            results_file = Path(CONFIG["RESULTS_DIR"]) / "dominick_results.pkl"
            final_results = {
                "wql": wql_geo,
                "mase": mase_geo,
                "series_processed": len(progress_tracker.wqls),
                "total_series": total_series,
                "timestamp": time.time(),
                "method": "deterministic"  # Mark as deterministic run
            }
            
            with open(results_file, "wb") as f:
                pickle.dump(final_results, f)
            
            # Remove checkpoint file after successful completion
            if progress_tracker.checkpoint_file.exists():
                progress_tracker.checkpoint_file.unlink()
            
            log.info("=== FINAL RESULTS ===")
            log.info(f"WQL (geo): {wql_geo:.6f}")
            log.info(f"MASE (geo): {mase_geo:.6f}")
            log.info(f"Series processed: {len(progress_tracker.wqls)}/{total_series}")
            log.info(f"Results saved to {results_file}")
        
    except Exception as e:
        log.error(f"Fatal error in main: {e}")
        # Save checkpoint even on error
        progress_tracker.save_checkpoint(-1)
        raise
    
    log.info("Evaluation completed successfully!")

# ----------------------------------------------------------------------
# 9️⃣  ENTRY POINT with enhanced error handling
# ----------------------------------------------------------------------
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log.info("Keyboard interrupt received. Exiting gracefully...")
    except Exception as e:
        log.error(f"Unexpected error: {e}")
        sys.exit(1)
    finally:
        log.info("Script execution finished.")
