#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Chronos London Smart Meter Evaluation – **single‑GPU** version (Enhanced).
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
    "DATASET_PATH": "/home/h20250169/study/modelTraining/dm_eval_2/base/withoutSampling/ALLDATASET/chronos_datasets_monash_london_smart_meters",
    "PREDICTION_LENGTH": 8,
    "NUM_SAMPLES": 500,  # For probabilistic forecasting (paper uses sampling)
    "NUM_SERIES": 5560,  # entire of the dataset . Number of time series to process (NOT samples per series)
    "QUANTILE_LEVELS": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],  # Paper uses 9 quantiles
    "RESULTS_DIR": "results",
    "CHECKPOINT_INTERVAL": 500,
    "BATCH_SIZE": 1,
    "MAX_MEMORY_GB": 24,
    "CONTEXT_LENGTH": 512,  # From Chronos paper (Section 5.2)
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
    # Ensure actual has the right shape for broadcasting
    actual = actual.reshape(-1, 1) if len(actual.shape) == 1 else actual
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
# 7️⃣  SINGLE SERIES PROCESSING - Probabilistic with sampling (as in paper)
# ----------------------------------------------------------------------
def process_single_series(pipeline, ctx, fut, idx, progress_tracker):
    """Process a single series with probabilistic prediction (as in Chronos paper)"""
    try:
        # Convert to tensor - single series processing
        # Shape: (1, context_length) - adding batch dimension
        tensor = torch.from_numpy(ctx.astype(np.float32)).unsqueeze(0)
        
        with torch.no_grad():
            # Use probabilistic prediction with multiple samples (as in paper)
            # The pipeline returns samples with shape: (1, num_samples, prediction_length)
            # Where: 1 = batch size, 100 = num_samples, 8 = prediction_length
            samples = pipeline.predict(
                tensor,
                prediction_length=CONFIG["PREDICTION_LENGTH"],
                num_samples=CONFIG["NUM_SAMPLES"],  # Generate multiple samples for probabilistic forecast
            )
        
        # STEP 1: FIX THE SHAPE ISSUE
        # Original shape: (1, 100, 8) - we need to remove the batch dimension
        # Using squeeze(0) removes the first dimension (batch dimension)
        samples_np = samples.cpu().numpy().squeeze(0)  # Shape becomes: (100, 8)
        
        # STEP 2: CALCULATE QUANTILES FROM SAMPLES
        # We calculate quantiles across the sample dimension (axis=0)
        # For each of the 8 prediction steps, calculate 9 quantiles from 100 samples
        # Result shape: (8, 9) - 8 time steps × 9 quantile levels
        quantile_predictions = np.column_stack([
            np.quantile(samples_np, q, axis=0) for q in CONFIG["QUANTILE_LEVELS"]
        ])
        
        # STEP 3: EXTRACT MEDIAN FOR POINT FORECASTS
        # The median is the 0.5 quantile, which is at index 4 in our 9 quantile levels
        # Shape: (8,) - one median value for each of the 8 prediction steps
        median_pred = quantile_predictions[:, 4]  # 0.5 quantile is at index 4
        
        # STEP 4: CALCULATE METRICS
        # WQL: Compares quantile predictions (shape: 8×9) with actual future values (shape: 8)
        w = wql(fut, quantile_predictions, CONFIG["QUANTILE_LEVELS"])
        
        # MASE: Compares median predictions (shape: 8) with actual future values (shape: 8)
        seasonal_err = np.diff(ctx) if len(ctx) > 1 else np.array([1.0])
        m = mase(fut, median_pred, seasonal_err)
        
        # STEP 5: STORE RESULTS
        progress_tracker.add_result(idx, w, m)
        
        # Optional: Log success for debugging
        if (idx + 1) % 100 == 0:  # Log every 100 series to avoid too much output
            log.info(f"Successfully processed series {idx}: WQL={w:.4f}, MASE={m:.4f}")
        
    except Exception as e:
        log.error(f"Error processing series {idx}: {e}")

# ----------------------------------------------------------------------
# 8️⃣  MAIN FUNCTION - Enhanced with resilience
# ----------------------------------------------------------------------
def main():
    log.info("=== Chronos London Smart Meter Single‑GPU Evaluation (Probabilistic) ===")
    log.info(f"Visible GPUs: {torch.cuda.device_count()}")
    log.info(f"CUDA version: {torch.version.cuda}")
    log.info(f"Configuration: {CONFIG}")
    log.info(f"Using probabilistic prediction with {CONFIG['NUM_SAMPLES']} samples per series")
    
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
        
        # 🔥 FIXED: ADD DATASET ANALYSIS TO UNDERSTAND WHY ONLY 74 SERIES
        log.info("Analyzing dataset filtering...")
        short_count = 0
        nan_count = 0 
        valid_count = 0

        for idx, row in enumerate(ds):
            s = row[target_col]
            if len(s) <= CONFIG["PREDICTION_LENGTH"]:
                short_count += 1
            elif np.any(np.isnan(s)):
                nan_count += 1
            else:
                valid_count += 1

        log.info(f"Dataset analysis - Valid: {valid_count}, Too short: {short_count}, Has NaN: {nan_count}")
        
        # 🔥 FIXED: LENIENT FILTERING - PROCESS ALL POSSIBLE SERIES
        contexts, futures, valid_indices = [], [], []
        series_count = 0
        
        for idx, row in enumerate(ds):
            if shutdown_manager.shutdown_requested or series_count >= CONFIG["NUM_SERIES"]:
                break
                
            s = row[target_col]
            
            # 🔥 FIXED: Only exclude series with NaN values
            if np.any(np.isnan(s)):
                continue
                
            # 🔥 FIXED: Use available context, even if short
            if len(s) > CONFIG["PREDICTION_LENGTH"]:
                ctx = s[:-CONFIG["PREDICTION_LENGTH"]]
                fut = s[-CONFIG["PREDICTION_LENGTH"]:]
            else:
                # Use entire series as context, create zero future
                ctx = s
                fut = np.zeros(CONFIG["PREDICTION_LENGTH"])
                log.warning(f"Series {idx} is too short ({len(s)}), using zero future")
            
            if progress_tracker.should_process(idx):
                contexts.append(ctx)
                futures.append(fut)
                valid_indices.append(idx)
                series_count += 1
        
        log.info(f"After lenient filtering: {len(valid_indices)} series to process (should be close to {CONFIG['NUM_SERIES']})")
        
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
        
        log.info(f"Starting probabilistic inference with {CONFIG['NUM_SAMPLES']} samples per series...")
        
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
            results_file = Path(CONFIG["RESULTS_DIR"]) / "london_smart_meter_results.pkl"
            final_results = {
                "wql": wql_geo,
                "mase": mase_geo,
                "series_processed": len(progress_tracker.wqls),
                "total_series": total_series,
                "num_samples_per_series": CONFIG["NUM_SAMPLES"],
                "timestamp": time.time(),
                "method": f"probabilistic_{CONFIG['NUM_SAMPLES']}_samples"
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
            log.info(f"Samples per series: {CONFIG['NUM_SAMPLES']}")
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
# nohup python3 1.py  > eval_output.log 2>&1 &
# kill %1