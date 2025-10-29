#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Chronos Dominick Evaluation – **single‑GPU** version.
This script runs on the GPU whose index is given in the
CUDA_VISIBLE_DEVICES line (default = 0).  It avoids the OOM
that occurs when trying to start several processes that each
load a full Chronos model.

The logic is identical to the original Map‑Reduce version,
only the parallel part has been removed.
"""

# ----------------------------------------------------------------------
# 0️⃣  Make the process see only ONE GPU  (change the number if you
#      want to run on a different card – e.g. "1", "2", "3").
# ----------------------------------------------------------------------
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"   # <‑‑ edit if you want a different GPU

# ----------------------------------------------------------------------
# 1️⃣  Imports – torch is imported **after** the env‑var is set,
#      so the CUDA context is created with a single visible device.
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
# 3️⃣  SETTINGS
# ----------------------------------------------------------------------
DATASET_PATH      = "/home/h20250169/study/modelTraining/dm_eval_2/base/withoutSampling/ALLDATASET/chronos_datasets_dominick"
PREDICTION_LENGTH = 8
NUM_SAMPLES       = 20                # lower if you still hit memory limits
QUANTILE_LEVELS    = [0.1, 0.5, 0.9]
RESULTS_DIR       = "results"

log.info("=== Chronos Dominick Single‑GPU Evaluation ===")
log.info(f"GPUs visible to this process: {torch.cuda.device_count()} | CUDA: {torch.version.cuda}")

# ----------------------------------------------------------------------
# 4️⃣  METRICS
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
# 5️⃣  MAIN – everything runs on the single GPU
# ----------------------------------------------------------------------
def main():
    # --------------------------------------------------------------
    # a) Load the dataset
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
    # b) Load the Chronos model (single GPU, limited memory)
    # --------------------------------------------------------------
    max_mem = {0: "5GiB"}                 # safe ceiling for a Tesla M10
    pipeline = ChronosPipeline.from_pretrained(
        "amazon/chronos-t5-base",
        device_map="auto",
        dtype=torch.float16,
        max_memory=max_mem,
        offload_folder="./offload",
        offload_state_dict=True,
    )
    log.info("Model loaded – starting inference...")

    # --------------------------------------------------------------
    # c) Inference loop
    # --------------------------------------------------------------
    wqls, mases = [], []
    for idx, (ctx, fut) in enumerate(zip(ctx_all, fut_all)):
        if len(ctx) == 0 or np.any(np.isnan(fut)):
            continue

        # ----- 1️⃣  keep context on CPU (Chronos moves it) -----
        tensor = torch.from_numpy(ctx.astype(np.float32)).unsqueeze(0)   # (1, L) on CPU

        # ----- 2️⃣  predict -----
        with torch.no_grad():
            samples = pipeline.predict(
                tensor,
                prediction_length=PREDICTION_LENGTH,
                num_samples=NUM_SAMPLES,
            )                     # (1, num_samples, prediction_length)

        samples = samples[0].cpu().numpy()          # Remove batch dimension: (num_samples, prediction_length)

        # ----- 3️⃣  quantiles & metrics -----
        qpred = np.quantile(samples, QUANTILE_LEVELS, axis=0).T   # (prediction_length, len(QUANTILE_LEVELS))

        w = wql(fut, qpred, QUANTILE_LEVELS)
        if not np.isnan(w):
            wqls.append(w)

        # FIX: Extract median correctly - it should be the second column (0.5 quantile)
        median = qpred[:, 1]  # This gives shape (8,) - the median predictions
        seasonal_err = np.diff(ctx) if len(ctx) > 1 else np.array([1.0])
        m = mase(fut, median, seasonal_err)
        if not np.isnan(m):
            mases.append(m)

        if (idx + 1) % 1000 == 0:
            log.info(f"[GPU 0] {idx + 1}/{len(ctx_all)} series processed")

    log.info(f"[GPU 0] DONE – {len(wqls)} WQL, {len(mases)} MASE")

    # --------------------------------------------------------------
    # d) Reduce & save results
    # --------------------------------------------------------------
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


if __name__ == "__main__":
    # Optional – avoid large CUDA fragmentation (same flag you saw before)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    main()
