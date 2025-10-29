#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Chronos Dominick Evaluation – **Single-GPU, Batched & Optimized**

This version is optimized for a single, powerful GPU (like a V100).
It removes all memory offloading and implements batched inference
to maximize GPU utilization and throughput.
"""

# ----------------------------------------------------------------------
# 0️⃣  Make the process see only ONE GPU
# ----------------------------------------------------------------------
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"   # <‑‑ Set to your target GPU

# ----------------------------------------------------------------------
# 1️⃣  Imports
# ----------------------------------------------------------------------
import logging
import pickle
import numpy as np
from scipy.stats import gmean
import datasets
import torch
from chronos import ChronosPipeline

# ----------------------------------------------------------------------
# 2️⃣  LOGGING
# ----------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# 3️⃣  SETTINGS (Optimized for V100)
# ----------------------------------------------------------------------
DATASET_PATH      = "/home/h20250169/study/modelTraining/dm_eval_2/base/withoutSampling/ALLDATASET/chronos_datasets_electricity_15min"
PREDICTION_LENGTH = 8
NUM_SAMPLES       = 50                # Increased for better accuracy, V100 can handle it
QUANTILE_LEVELS    = [0.1, 0.5, 0.9]
RESULTS_DIR       = "results"
BATCH_SIZE        = 8                 # 🚀 **REDUCED: To avoid OOM errors**

log.info("=== Chronos Dominick Single-GPU (Batched) Evaluation ===")
log.info(f"GPUs visible to this process: {torch.cuda.device_count()} | CUDA: {torch.version.cuda}")
log.info(f"Batch Size: {BATCH_SIZE} | Num Samples: {NUM_SAMPLES}")

# ----------------------------------------------------------------------
# 4️⃣  METRICS (Unchanged)
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
# 5️⃣  MAIN – BATCHED EXECUTION
# ----------------------------------------------------------------------
def main():
    # --------------------------------------------------------------
    # a) Load the dataset (Unchanged)
    # --------------------------------------------------------------
    log.info("Loading dataset...")
    ds = datasets.load_from_disk(DATASET_PATH)["train"]
    target_col = next(
        k
        for k, v in ds.features.items()
        if isinstance(v, datasets.Sequence) and v.feature.dtype in ("float32", "float64")
    )
    ds.set_format("numpy")

    ctx_all, fut_all = [], []
    for row in ds:
        s = row[target_col]
        ctx_all.append(s[:-PREDICTION_LENGTH])
        fut_all.append(s[-PREDICTION_LENGTH:])
    log.info(f"Loaded {len(ctx_all)} series")

    # --------------------------------------------------------------
    # b) Load the Chronos model (🚀 FIX #1: Removed all offloading)
    # --------------------------------------------------------------
    log.info("Loading model to V100 (no offloading)...")
    pipeline = ChronosPipeline.from_pretrained(
        "amazon/chronos-t5-base",
        device_map="auto",              # Automatically uses the visible GPU
        dtype=torch.float16,           # 🚀 FIX: Use 'dtype' instead of 'torch_dtype'
    )
    log.info("Model loaded – starting batched inference...")

    # --------------------------------------------------------------
    # c) Inference loop (🚀 FIX #2: Simplified - let Chronos handle device placement)
    # --------------------------------------------------------------
    all_samples_list = [] # Store results from each batch
    all_futs_list = []    # Store corresponding ground truths
    
    # Filter out empty/NaN series *before* batching
    valid_indices = [
        i for i, (ctx, fut) in enumerate(zip(ctx_all, fut_all))
        if len(ctx) > 0 and not np.any(np.isnan(fut))
    ]
    log.info(f"Processing {len(valid_indices)} valid series...")

    # Clear any cached memory before starting
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Use a variable batch size that can be adjusted dynamically
    current_batch_size = BATCH_SIZE
    i = 0
    
    while i < len(valid_indices):
        batch_indices = valid_indices[i : i + current_batch_size]
        
        # 🚀 FIX: Don't manually move to GPU - let Chronos handle device placement
        # Convert numpy arrays to PyTorch tensors (keep on CPU)
        batch_ctx = [torch.tensor(ctx_all[idx], dtype=torch.float32) for idx in batch_indices]
        batch_fut = [fut_all[idx] for idx in batch_indices]
        
        # --- Predict on the batch ---
        with torch.no_grad():
            try:
                samples = pipeline.predict(
                    batch_ctx,
                    prediction_length=PREDICTION_LENGTH,
                    num_samples=NUM_SAMPLES,
                )
                
                # Move samples to CPU immediately to free GPU memory
                samples_cpu = samples.cpu().numpy()
                all_samples_list.append(samples_cpu)
                all_futs_list.extend(batch_fut)
                
                # Clear GPU cache after each successful batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Successfully processed this batch, move to next
                i += current_batch_size
                
                # Reset batch size to original if we successfully processed
                current_batch_size = BATCH_SIZE
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    log.error(f"OOM error at index {i}. Reducing batch size from {current_batch_size}...")
                    # Clear cache and reduce batch size
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    # Reduce batch size but don't skip indices
                    current_batch_size = max(1, current_batch_size // 2)
                    log.info(f"Reduced batch size to {current_batch_size}")
                    
                    # Don't increment i - we'll retry the same indices with smaller batch
                else:
                    # For device mismatch errors, try processing one series at a time
                    if "same device" in str(e).lower():
                        log.error(f"Device mismatch error. Trying with batch size 1...")
                        current_batch_size = 1
                    else:
                        raise e

        # Progress logging
        if i % 100 < current_batch_size or i >= len(valid_indices):
            log.info(f"[GPU 0] {min(i, len(valid_indices))}/{len(valid_indices)} series processed")

    log.info("[GPU 0] INFERENCE DONE – Calculating metrics...")

    # --------------------------------------------------------------
    # d) Calculate Metrics (Post-loop)
    # --------------------------------------------------------------
    wqls, mases = [], []
    
    # Combine all batch results into one big array
    if all_samples_list:
        all_samples_np = np.concatenate(all_samples_list, axis=0)

        # Make sure we have the same number of samples as future values
        if len(all_samples_np) != len(all_futs_list):
            log.error(f"Mismatch between samples ({len(all_samples_np)}) and futures ({len(all_futs_list)})")
            # Use the minimum length to avoid index errors
            min_length = min(len(all_samples_np), len(all_futs_list))
            all_samples_np = all_samples_np[:min_length]
            all_futs_list = all_futs_list[:min_length]

        for idx, fut in enumerate(all_futs_list):
            # We get the original context (ctx) using the *original* index
            if idx < len(valid_indices):
                original_idx = valid_indices[idx]
                ctx = ctx_all[original_idx]
                
                samples = all_samples_np[idx]  # (num_samples, prediction_length)
                
                # ----- Quantiles & Metrics -----
                qpred = np.quantile(samples, QUANTILE_LEVELS, axis=0).T   # (H, 3)

                w = wql(fut, qpred, QUANTILE_LEVELS)
                if not np.isnan(w):
                    wqls.append(w)

                median = qpred[:, 1]
                seasonal_err = np.diff(ctx) if len(ctx) > 1 else np.array([1.0])
                m = mase(fut, median, seasonal_err)
                if not np.isnan(m):
                    mases.append(m)

        log.info(f"[CPU] METRICS DONE – {len(wqls)} WQL, {len(mases)} MASE")
    else:
        log.error("No samples were processed successfully!")
        return

    # --------------------------------------------------------------
    # e) Reduce & save results
    # --------------------------------------------------------------
    if wqls and mases:
        wql_geo = gmean(np.maximum(wqls, 1e-9))
        mase_geo = gmean(np.maximum(mases, 1e-9))

        os.makedirs(RESULTS_DIR, exist_ok=True)
        out = {"wql": wql_geo, "mase": mase_geo, "series": len(wqls)}
        with open(f"{RESULTS_DIR}/dominick_results.pkl", "wb") as f:
            pickle.dump(out, f)

        log.info("=== FINAL RESULTS ===")
        log.info(f"WQL (geo): {wql_geo:.6f}")
        log.info(f"MASE (geo): {mase_geo:.6f}")
        log.info(f"Series: {len(wqls)}")
        log.info(f"Results saved to {RESULTS_DIR}/dominick_results.pkl")
    else:
        log.error("No valid metrics calculated!")


if __name__ == "__main__":
    main()
# 02:39:44 | INFO     | === FINAL RESULTS ===
# 02:39:44 | INFO     | WQL (geo): 0.062607
# 02:39:44 | INFO     | MASE (geo): 0.568221
# 02:39:44 | INFO     | Series: 368
