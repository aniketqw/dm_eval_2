#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
evaluate_chronos_multiseed.py - Run Chronos evaluation over multiple random seeds

This script runs the evaluation 3 times with different random seeds (as done in the paper)
and averages the results to match the paper's methodology.
"""

import sys
import json
import pickle
from pathlib import Path
import numpy as np
import torch
from scipy.stats import gmean

# Import from your chronos_common.py
from chronos_common import (
    setup_logging,
    GracefulShutdown,
    ProgressTracker,
    load_dataset,
    run_inference,
)
from chronos import ChronosPipeline


def run_single_seed(seed: int, config: dict, logger):
    """Run evaluation for a single seed."""
    logger.info(f"\n{'='*70}")
    logger.info(f"=== SEED {seed} ===")
    logger.info(f"{'='*70}\n")
    
    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Create a seed-specific results directory
    seed_results_dir = f"{config['RESULTS_DIR']}_seed{seed}"
    
    # Modify config for this seed
    seed_config = config.copy()
    seed_config['RESULTS_DIR'] = seed_results_dir
    
    # Initialize
    shutdown = GracefulShutdown(logger)
    tracker = ProgressTracker(seed_results_dir, config["CHECKPOINT_INTERVAL"], logger)
    
    # Load dataset
    contexts, futures, indices = load_dataset(
        config["CSV_URL"],
        config["PARQUET_PATH"],
        config["PREDICTION_LENGTH"],
        logger,
    )
    
    # Load model
    logger.info("Loading Chronos model...")
    pipeline = ChronosPipeline.from_pretrained(
        "amazon/chronos-t5-base",
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype=torch.bfloat16,
    )
    
    # Run inference
    run_inference(pipeline, contexts, futures, tracker, shutdown, logger, seed_config)
    
    # Compute metrics for this seed
    wql_geo = gmean(np.maximum(tracker.wqls, 1e-9))
    mase_geo = gmean(np.maximum(tracker.mases, 1e-9))
    baseline_wql_geo = gmean(np.maximum(tracker.baseline_wqls, 1e-9))
    baseline_mase_geo = gmean(np.maximum(tracker.baseline_mases, 1e-9))
    
    logger.info(f"\n{'='*70}")
    logger.info(f"=== SEED {seed} RESULTS ===")
    logger.info(f"  Model WQL:  {wql_geo:.6f}")
    logger.info(f"  Model MASE: {mase_geo:.6f}")
    logger.info(f"  Baseline WQL:  {baseline_wql_geo:.6f}")
    logger.info(f"  Baseline MASE: {baseline_mase_geo:.6f}")
    logger.info(f"{'='*70}\n")
    
    return {
        'wql': wql_geo,
        'mase': mase_geo,
        'baseline_wql': baseline_wql_geo,
        'baseline_mase': baseline_mase_geo,
    }


def main():
    """Main function to run multi-seed evaluation."""
    
    # Configuration - matching the paper
    CONFIG = {
        "CSV_URL": "https://autogluon.s3.amazonaws.com/datasets/timeseries/nn5/train.csv",
        "PARQUET_PATH": "/home/h20250169/study/modelTraining/dm_eval_2/base/withoutSampling/singleDataset/short/NN5_DAILY/dataset/nn5_daily_clean.parquet.gz",
        "NUM_SAMPLES": 20,# no . of different trajectories from t5-base to obtain a predictive distribution and compute quantile
        "QUANTILE_LEVELS": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        "SEASON_LENGTH": 7,
        "RESULTS_DIR": "results",
        "CHECKPOINT_INTERVAL": 500,
        "MAX_MEMORY_GB": 24,
        "BATCH_SIZE": 1,
        "CONTEXT_LENGTH": 512,
        "PREDICTION_LENGTH":56,

    }
    
    # Random seeds to use (as done in the paper)
    SEEDS = [42, 123, 456]
    
    # Setup logging
    logger = setup_logging()
    
    logger.info("="*70)
    logger.info("=== Chronos Multi-Seed Evaluation ===")
    logger.info("="*70)
    logger.info(f"Running evaluation over {len(SEEDS)} seeds: {SEEDS}")
    logger.info(f"Configuration: {json.dumps(CONFIG, indent=2)}")
    logger.info("="*70)
    
    # Run evaluation for each seed
    all_results = []
    for seed in SEEDS:
        try:
            results = run_single_seed(seed, CONFIG, logger)
            all_results.append(results)
        except Exception as e:
            logger.error(f"Error running seed {seed}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if not all_results:
        logger.error("No successful runs! Exiting.")
        sys.exit(1)
    
    # Aggregate results across seeds (using arithmetic mean as per paper)
    logger.info("\n" + "="*70)
    logger.info("=== FINAL AGGREGATED RESULTS (AVERAGED OVER SEEDS) ===")
    logger.info("="*70)
    
    # Individual seed results
    logger.info("\nPer-seed results:")
    for i, (seed, result) in enumerate(zip(SEEDS, all_results)):
        logger.info(f"  Seed {seed}: WQL={result['wql']:.6f}, MASE={result['mase']:.6f}")
    
    # Compute mean across seeds
    mean_wql = np.mean([r['wql'] for r in all_results])
    mean_mase = np.mean([r['mase'] for r in all_results])
    mean_baseline_wql = np.mean([r['baseline_wql'] for r in all_results])
    mean_baseline_mase = np.mean([r['baseline_mase'] for r in all_results])
    
    # Compute relative scores
    relative_wql = mean_wql / mean_baseline_wql if mean_baseline_wql > 0 else np.nan
    relative_mase = mean_mase / mean_baseline_mase if mean_baseline_mase > 0 else np.nan
    
    logger.info("\n" + "-"*70)
    logger.info("FINAL RESULTS (Mean over 3 seeds):")
    logger.info("-"*70)
    logger.info("MODEL - ABSOLUTE SCORES:")
    logger.info(f"  WQL  (mean over seeds) : {mean_wql:.6f}")
    logger.info(f"  MASE (mean over seeds) : {mean_mase:.6f}")
    logger.info("")
    logger.info("BASELINE - ABSOLUTE SCORES:")
    logger.info(f"  WQL  (mean over seeds) : {mean_baseline_wql:.6f}")
    logger.info(f"  MASE (mean over seeds) : {mean_baseline_mase:.6f}")
    logger.info("")
    logger.info("RELATIVE SCORES (Model/Baseline):")
    logger.info(f"  WQL  (relative) : {relative_wql:.6f}")
    logger.info(f"  MASE (relative) : {relative_mase:.6f}")
    logger.info("="*70)
    
    logger.info("\n" + "-"*70)
    logger.info("COMPARISON TO PAPER:")
    logger.info("-"*70)
    logger.info(f"  Paper WQL:  0.161")
    logger.info(f"  Your WQL:   {mean_wql:.6f}")
    logger.info(f"  Difference: {abs(mean_wql - 0.207):.6f} ({abs(mean_wql - 0.207)/0.207*100:.1f}%)")
    logger.info("")
    logger.info(f"  Paper MASE: 0.585")
    logger.info(f"  Your MASE:  {mean_mase:.6f}")
    logger.info(f"  Difference: {abs(mean_mase - 3.9):.6f} ({abs(mean_mase - 3.9)/3.9*100:.1f}%)")
    logger.info("="*70)
    
    # Save aggregated results
    final_results = {
        'seeds': SEEDS,
        'per_seed_results': all_results,
        'mean_wql': mean_wql,
        'mean_mase': mean_mase,
        'mean_baseline_wql': mean_baseline_wql,
        'mean_baseline_mase': mean_baseline_mase,
        'relative_wql': relative_wql,
        'relative_mase': relative_mase,
        'config': CONFIG,
    }
    
    results_path = Path(CONFIG['RESULTS_DIR']) / "chronos_multiseed_results.pkl"
    results_path.parent.mkdir(exist_ok=True)
    with open(results_path, 'wb') as f:
        pickle.dump(final_results, f)
    
    logger.info(f"\nAggregated results saved to: {results_path}")
    logger.info("\n=== Multi-seed evaluation complete! ===\n")


if __name__ == "__main__":
    main()
    # nohup python3 evaluate_chronos_multiseed.py  > eval_output.log 2>&1 &