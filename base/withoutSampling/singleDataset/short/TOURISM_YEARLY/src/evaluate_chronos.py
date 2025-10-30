#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
evaluate_chronos.py

A thin wrapper around *chronos_common.py*.  The only thing you change
when you switch to a different Chronos benchmark is the ``CONFIG`` dict
below – everything else (logging, checkpointing, inference, metrics)
stays exactly the same.
"""

# ----------------------------------------------------------------------
# 0️⃣  ONE‑GPU ENVIRONMENT (optional, keep if you have several GPUs)
# ----------------------------------------------------------------------
import os,sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ----------------------------------------------------------------------
# 1️⃣  IMPORT COMMON UTILITIES
# ----------------------------------------------------------------------
from chronos_common import (
    setup_logging,
    GracefulShutdown,
    ProgressTracker,
    load_dataset,
    run_inference,
    write_final_summary,
)

import torch

# ----------------------------------------------------------------------
# 2️⃣  CONFIGURATION – **ONLY THIS SECTION changes per dataset**
# ----------------------------------------------------------------------
CONFIG = {
    # ------------------------------------------------------------------
    #   INPUT DATA
    # ------------------------------------------------------------------
    "CSV_URL": "https://autogluon.s3.amazonaws.com/datasets/timeseries/monash_tourism_yearly/train.csv",
    # Path to the parquet you produced with `load_chronos_csv.py` (or any other tool)
    "PARQUET_PATH": "/home/h20250169/study/modelTraining/dm_eval_2/base/withoutSampling/singleDataset/short/TOURISM_YEARLY/dataset/tourism_yearly_clean.parquet.gz",

    # ------------------------------------------------------------------
    #   MODEL / INFERENCE SETTINGS
    # ------------------------------------------------------------------
    "PREDICTION_LENGTH": 4,                 # how many steps ahead we forecast
    "NUM_SAMPLES": 20,                      # Monte‑Carlo draws per series (paper uses 20 for quantile estimation)
    "QUANTILE_LEVELS": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],

    # ------------------------------------------------------------------
    #   SEASONAL NAIVE BASELINE SETTINGS (CRITICAL FOR NORMALIZATION!)
    # ------------------------------------------------------------------
    # For YEARLY data: season_length = 1 (last observation)
    # For MONTHLY data: season_length = 12
    # For QUARTERLY data: season_length = 4
    # For DAILY/WEEKLY: depends on your data pattern
    "SEASON_LENGTH": 1,  # <-- **YEARLY DATA**

    # ------------------------------------------------------------------
    #   DISK / CHECKPOINT SETTINGS
    # ------------------------------------------------------------------
    "RESULTS_DIR": "results",
    "CHECKPOINT_INTERVAL": 500,             # write a checkpoint every N series
    "MAX_MEMORY_GB": 24,                    # used when we instantiate the Chronos pipeline

    # ------------------------------------------------------------------
    #   OTHER TUNABLES
    # ------------------------------------------------------------------
    "BATCH_SIZE": 1,        # Chronos only supports batch‑size 1 for the public models
    "CONTEXT_LENGTH": 512,  # not used directly – kept for compatibility with older scripts
}

# ----------------------------------------------------------------------
# 3️⃣  MAIN – orchestrates everything (no changes needed)
# ----------------------------------------------------------------------
def main():
    logger = setup_logging()
    logger.info("=== Chronos Evaluation – CORRECT WQL/MASE Calculation ===")
    logger.info(f"Configuration: {CONFIG}")

    # --------------------------------------------------------------
    # 0️⃣  Graceful‑shutdown manager (shared)
    # --------------------------------------------------------------
    shutdown = GracefulShutdown(logger)

    # --------------------------------------------------------------
    # 1️⃣  Load data (parquet preferred, CSV fallback)
    # --------------------------------------------------------------
    contexts, futures, indices = load_dataset(
        csv_url        = CONFIG["CSV_URL"],
        parquet_path   = CONFIG["PARQUET_PATH"],
        pred_len       = CONFIG["PREDICTION_LENGTH"],
        logger         = logger,
    )

    # --------------------------------------------------------------
    # 2️⃣  Load the Chronos model (T5‑Base, fp16, auto device map)
    # --------------------------------------------------------------
    from chronos import ChronosPipeline
    logger.info("Loading Chronos model …")
    max_mem = {0: f"{CONFIG['MAX_MEMORY_GB']}GiB"}
    pipeline = ChronosPipeline.from_pretrained(
        "amazon/chronos-t5-base",
        device_map="auto",
        torch_dtype=torch.float16,
        max_memory=max_mem,
    )

    # --------------------------------------------------------------
    # 3️⃣  Tracker (checkpoint handling)
    # --------------------------------------------------------------
    tracker = ProgressTracker(CONFIG["RESULTS_DIR"], CONFIG["CHECKPOINT_INTERVAL"], logger)

    # --------------------------------------------------------------
    # 4️⃣  Run the inference loop (shared)
    # --------------------------------------------------------------
    run_inference(pipeline, contexts, futures, tracker, shutdown, logger, CONFIG)

    # --------------------------------------------------------------
    # 5️⃣  Write final summary (shared)
    # --------------------------------------------------------------
    write_final_summary(tracker, logger, CONFIG)

    logger.info("=== Evaluation finished ===")


# ----------------------------------------------------------------------
# 🔟  ENTRY POINT
# ----------------------------------------------------------------------
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚡️  KeyboardInterrupt – exiting gracefully")
    except Exception as exc:
        import traceback
        print("\n❌  Unexpected error:")
        traceback.print_exc()
        sys.exit(1)
    finally:
        print("🚦  Script execution completed")