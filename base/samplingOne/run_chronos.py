import os
import pickle
import glob
import shutil
import time
import signal
from contextlib import contextmanager
# Set environment variables BEFORE importing
os.environ["DATASETS_VERBOSITY"] = "warning"
import datasets
from datasets import DownloadConfig
from datasets.features import Value as DatasetValue ### REFACTOR: Import Value for type checking
from scipy.stats import gmean
import numpy as np
import pandas as pd
from chronos import ChronosPipeline
import logging
import torch
import traceback
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
# --- Global Configuration ---
PREDICTION_LENGTH = 12
NUM_SAMPLES_PER_SERIES = 100 # Increased for 32GB VRAM
RESULTS_DIR = "results" # Local results folder
QUANTILE_LEVELS = [0.1, 0.5, 0.9]
BATCH_SIZE = 4 # Increased for better throughput on V100
CACHE_DIR = os.path.expanduser("~/.cache/huggingface")
MIN_FREE_SPACE_GB = 10 # Adjusted for server
MAX_CACHE_SIZE_GB = 20 # Increased for 32GB VRAM/system
STREAMING_FALLBACK = True
MIN_SERIES_LENGTH = PREDICTION_LENGTH + 1 # 13 points minimum to have at least 1 in context
# --- DATASET SIZE TIERS (NEW!) ---
# REFACTOR: Added 'target_override' key (defaults to None if not present)
# This allows manual specification for datasets that fail automatic discovery
# or have multiple valid target columns.
# Increased workers for V100
DATASET_TIERS = {
    "very_small": {
        "datasets": ["nn5", "exchange_rate", "monash_m1_yearly", "monash_tourism_yearly",
                     "monash_m3_yearly"],
        "workers": 4,
        "timeout": 600,
        "max_cache_gb": 10,
        "target_override": None
    },
    "small": {
        "datasets": ["m4_monthly", "m4_daily", "monash_tourism_quarterly", "monash_tourism_monthly",
                     "monash_m3_quarterly", "monash_m3_monthly", "monash_m1_quarterly", "monash_m1_monthly",
                     "monash_saugeenday", "monash_temperature_rain"],
        "workers": 8,
        "timeout": 960,
        "max_cache_gb": 16,
        "target_override": None
    },
    "medium": {
        "datasets": ["dominick", "ercot", "monash_car_parts", "m4_hourly", "monash_hospital",
                     "monash_traffic", "monash_covid_deaths", "monash_pedestrian_counts", "monash_fred_md"],
        "workers": 12,
        "timeout": 1440,
        "max_cache_gb": 24,
        "target_override": None
    },
    "medium_large": {
        "datasets": ["electricity_15min", "solar_1h", "wind_farms_hourly", "taxi_1h",
                     "mexico_city_bikes", "wind_farms_daily", "ushcn_daily", "uber_tlc_hourly",
                     "uber_tlc_daily", "taxi_30min", "solar"],
        "workers": 16,
        "timeout": 1800,
        "max_cache_gb": 30,
        "target_override": None
    },
    "large": {
        "datasets": ["monash_rideshare", "monash_nn5_weekly", "monash_london_smart_meters",
                     "monash_kdd_cup_2018", "monash_electricity_weekly"],
        "workers": 20,
        "timeout": 2400,
        "max_cache_gb": 40,
        "target_override": None
    },
}
# Skip problematic datasets (NEW!)
SKIP_DATASETS = [
    "wiki_daily_100k",
    "training_corpus_tsmixup_10m",
    "training_corpus_kernel_synth_1m",
    "weatherbench_hourly_vorticity",
    "weatherbench_hourly_total_cloud_cover",
    "weatherbench_hourly_potential_vorticity",
    "weatherbench_hourly_2m_temperature",
    "weatherbench_hourly_total_precipitation",
    "weatherbench_hourly_v_component_of_wind",
    "weatherbench_daily",
    "weatherbench_hourly_10m_u_component_of_wind",
    "weatherbench_hourly_u_component_of_wind",
    "weatherbench_hourly_10m_v_component_of_wind",
    "weatherbench_hourly_temperature",
    "weatherbench_hourly_toa_incident_solar_radiation",
    "weatherbench_weekly",
    "weatherbench_hourly_specific_humidity",
    "weatherbench_hourly_relative_humidity",
    "weatherbench_hourly_geopotential",
]
# --- Helper Functions ---
### REFACTOR: New function for dynamic target column discovery
def find_target_column(dataset_features):
    """
    Inspects dataset features to find the first valid numeric Sequence column.
    This dynamically finds the 'target' column, which isn't always named 'target'.
    """
    # Define valid numeric types
    numeric_types = (
        DatasetValue(dtype='float32', id=None),
        DatasetValue(dtype='float64', id=None),
        DatasetValue(dtype='int32', id=None),
        DatasetValue(dtype='int64', id=None),
        DatasetValue(dtype='int16', id=None),
        DatasetValue(dtype='int8', id=None)
    )
    # Columns to explicitly ignore
    ignore_cols = ['id', 'timestamp', 'category', 'item_id']
    for col_name, feature in dataset_features.items():
        if col_name in ignore_cols:
            continue
    
        # Check if it's a Sequence...
        if isinstance(feature, datasets.Sequence):
            #...and the inner feature is a numeric Value
            if feature.feature in numeric_types:
                logging.info(f" Discovered target column: '{col_name}'")
                return col_name
            
    raise ValueError("Schema Error: No valid numeric Sequence column found.")
def get_dataset_tier(config_name):
    """Determine which tier a dataset belongs to."""
    for tier_name, tier_config in DATASET_TIERS.items():
        if config_name in tier_config["datasets"]:
            return tier_name, tier_config
    return "medium", DATASET_TIERS["medium"]
def get_prioritized_dataset_list():
    """Get all datasets in order from smallest to largest."""
    all_datasets = []
    for tier_name in ["very_small", "small", "medium", "medium_large", "large"]:
        all_datasets.extend(DATASET_TIERS[tier_name]["datasets"])
    return all_datasets
@contextmanager
def timeout(seconds, error_message="Operation timed out"):
    """Context manager for timeout operations (NEW!)"""
    def timeout_handler(signum, frame):
        raise TimeoutError(error_message)
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
def get_cache_size():
    """Get current cache size in GB."""
    try:
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(CACHE_DIR):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                try:
                    total_size += os.path.getsize(filepath)
                except:
                    pass
        return total_size / (1024**3)
    except:
        return 0
def clear_all_cache():
    """Aggressively clear ALL HuggingFace caches."""
    try:
        cache_size = get_cache_size()
        logging.info(f"🧹 CLEARING CACHE (Current: {cache_size:.2f} GB)")
    
        # Clear datasets cache
        datasets_cache = os.path.join(CACHE_DIR, "datasets")
        if os.path.exists(datasets_cache):
            shutil.rmtree(datasets_cache, ignore_errors=True)
            os.makedirs(datasets_cache, exist_ok=True)
    
        # Clear hub cache (model downloads)
        hub_cache = os.path.join(CACHE_DIR, "hub")
        hub_datasets = os.path.join(hub_cache, "datasets--autogluon--chronos_datasets")
        if os.path.exists(hub_datasets):
            shutil.rmtree(hub_datasets, ignore_errors=True)
    
        # Force garbage collection
        import gc
        gc.collect()
    
        new_cache_size = get_cache_size()
        freed = cache_size - new_cache_size
        logging.info(f"✅ Cache cleared! Freed: {freed:.2f} GB")
    
        time.sleep(1)
    
    except Exception as e:
        logging.error(f"⚠️ Error clearing cache: {e}")
def check_disk_space():
    """Check and log available disk space."""
    try:
        total, used, free = shutil.disk_usage("/")
        free_gb = free / (1024**3)
        total_gb = total / (1024**3)
        used_percent = (used / total) * 100
    
        logging.info(f"💾 DISK: {free_gb:.2f} GB free / {total_gb:.2f} GB total ({used_percent:.1f}% used)")
    
        if free_gb < MIN_FREE_SPACE_GB:
            logging.warning(f"⚠️ LOW DISK SPACE! Only {free_gb:.2f} GB free (need {MIN_FREE_SPACE_GB} GB)")
            return False
        return True
    except Exception as e:
        logging.error(f"Error checking disk space: {e}")
        return True
def check_and_manage_cache(required_space_gb=10):
    """Check cache size and clear if needed before download (NEW!)"""
    cache_size = get_cache_size()
    if cache_size > MAX_CACHE_SIZE_GB:
        logging.warning(f"⚠️ Cache size ({cache_size:.2f} GB) exceeds limit ({MAX_CACHE_SIZE_GB} GB)")
        clear_all_cache()
        return True
    total, used, free = shutil.disk_usage("/")
    free_gb = free / (1024**3)
    if free_gb < required_space_gb:
        logging.warning(f"⚠️ Only {free_gb:.2f} GB free, need {required_space_gb} GB")
        clear_all_cache()
        return True
    return False
def get_processed_datasets():
    """Get list of successfully processed datasets."""
    processed = []
    if os.path.exists(RESULTS_DIR):
        pkl_files = glob.glob(os.path.join(RESULTS_DIR, "*_large_results.pkl"))
        for pkl_file in pkl_files:
            try:
                with open(pkl_file, 'rb') as f:
                    data = pickle.load(f)
                    if data.get('num_processed_series', 0) > 0:
                        dataset_name = data['dataset_name']
                        processed.append(dataset_name)
            except Exception as e:
                logging.debug(f"Could not read {pkl_file}: {e}")
    return processed
def dataset_exists(config_name):
    """Check if dataset has already been processed (NEW!)"""
    pkl_path = os.path.join(RESULTS_DIR, f"{config_name}_large_results.pkl")
    if os.path.exists(pkl_path):
        try:
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
                if data.get('num_processed_series', 0) > 0:
                    logging.info(f"⏭️ SKIPPING {config_name} - Already processed")
                    return True
        except Exception as e:
            logging.warning(f"⚠️ Found corrupted.pkl for {config_name}: {e}")
            try:
                os.remove(pkl_path)
                txt_path = pkl_path.replace('.pkl', '.txt')
                if os.path.exists(txt_path):
                    os.remove(txt_path)
                logging.info(f"🗑️ Deleted corrupted files for {config_name}")
            except:
                pass
            return False
    return False
# --- Metric Functions ---
def weighted_quantile_loss(actual, predictions, quantiles):
    """Calculates the Weighted Quantile Loss (WQL)."""
    num_samples, num_quantiles = predictions.shape
    if len(quantiles)!= num_quantiles:
        raise ValueError("Number of quantiles in predictions and quantiles list must match.")
    sum_abs_actual = np.sum(np.abs(actual))
    if sum_abs_actual == 0:
        return np.nan
    losses = []
    for i, q in enumerate(quantiles):
        error = actual - predictions[:, i]
        loss = np.maximum(q * error, (q - 1) * error)
        losses.append(loss)
    losses_stacked = np.stack(losses, axis=1)
    normalized_quantile_loss = np.sum(losses_stacked) / sum_abs_actual
    return normalized_quantile_loss
def mean_absolute_scaled_error(actual, predicted, seasonal_error):
    """Calculates the Mean Absolute Scaled Error (MASE)."""
    mae = np.mean(np.abs(actual - predicted))
    denominator = np.mean(np.abs(seasonal_error))
    if denominator == 0:
        return np.inf if mae > 0 else 0.0
    else:
        return mae / denominator
### REFACTOR: Function signature changed to accept target_column_name
def split_dataset(ds, prediction_length, target_column_name):
    """Splits each time series in a dataset into training and testing sets."""
    train_data = []
    test_data = []
    for i in range(len(ds)):
        ### REFACTOR: Use dynamic target_column_name instead of hard-coded 'target'
        series = ds[i][target_column_name]
        train_data.append(series[:-prediction_length])
        test_data.append(series[-prediction_length:])
    return train_data, test_data
# --- Core Processing Function ---
def process_dataset(config_name, pipeline):
    """Process a single dataset with all fixes applied."""
    # Check if already processed (NEW!)
    if dataset_exists(config_name):
        return "skipped"
    # Get tier configuration (NEW!)
    tier_name, tier_config = get_dataset_tier(config_name)
    timeout_seconds = tier_config["timeout"]
    workers = tier_config["workers"]
    logging.info(f"\n{'='*80}")
    logging.info(f"📊 PROCESSING: {config_name}")
    logging.info(f" Tier: {tier_name} | Timeout: {timeout_seconds}s | Workers: {workers}")
    logging.info(f"{'='*80}")
    # Check disk space before starting
    if not check_disk_space():
        logging.error(f"❌ Insufficient disk space to process {config_name}")
        return None
    start_time = time.time()
    try:
        # Timeout protection (NEW!)
        with timeout(timeout_seconds, f"{config_name} exceeded {timeout_seconds}s timeout"):
        
            # Clear cache if needed (NEW!)
            check_and_manage_cache(tier_config.get("max_cache_gb", 10))
        
            # Load dataset with proper config (NEW!)
            logging.info(f"⬇️ Loading dataset: {config_name}...")
        
            is_streaming = False
            ds_train = None
            try:
                # Try normal loading first
                download_config = DownloadConfig(num_proc=workers)
                ds_train = datasets.load_dataset(
                    "autogluon/chronos_datasets",
                    config_name,
                    split="train",
                    download_config=download_config
                )
            except Exception as e:
                if STREAMING_FALLBACK:
                    logging.warning(f"⚠️ Normal load failed, trying streaming mode: {str(e)[:100]}")
                    is_streaming = True
                    ds_train = datasets.load_dataset(
                        "autogluon/chronos_datasets",
                        config_name,
                        split="train",
                        streaming=True
                    )
                else:
                    raise
        
            ### REFACTOR: Dynamic Target Column Discovery
            target_column_name = None
            try:
                # 1. Check for manual override in config
                target_column_name = tier_config.get("target_override")
                if target_column_name:
                    logging.info(f" Found manual target override: '{target_column_name}'")
            
                # 2. If no override, perform automatic discovery
                else:
                    if is_streaming:
                        # For streaming, we must take 1 element to inspect features
                        ds_head = ds_train.take(1)
                        features = next(iter(ds_head)).keys()
                        # Streaming datasets don't have the rich.features attribute
                        # We must create a mock features dict to inspect
                        mock_features = {k: ds_train.info.features[k] for k in features}
                        target_column_name = find_target_column(mock_features)
                    else:
                        target_column_name = find_target_column(ds_train.features)
            
                # 3. Final check
                if not target_column_name:
                    raise ValueError("Target column discovery failed.")
                
            except Exception as e_schema:
                logging.error(f" ❌ SCHEMA ERROR: {config_name} - {str(e_schema)}")
                return None
            ### END REFACTOR
        
        
            # Convert to pandas DataFrame for sampling
            if is_streaming:
                logging.info("📊 Converting streaming dataset to list...")
                # Must re-init the iterable if we.take() from it
                ds_train = datasets.load_dataset(
                        "autogluon/chronos_datasets",
                        config_name,
                        split="train",
                        streaming=True
                    )
                ds_list = list(ds_train)
                df_ds = pd.DataFrame(ds_list)
            else:
                ds_train.set_format("numpy") # Set format after feature discovery
                df_ds = ds_train.to_pandas()
        
            # Perform sampling
            #stratified sampling if category exist
            if 'category' in df_ds.columns:
                sampled_df = df_ds.groupby('category', group_keys=False).apply(
                    lambda x: x.sample(min(len(x), NUM_SAMPLES_PER_SERIES), random_state=42)
                )
            #otherwise random sampling 
            else:
                sampled_df = df_ds.sample(min(len(df_ds), NUM_SAMPLES_PER_SERIES), random_state=42)
        
            # CRITICAL FIX: Filter out series that are too short!
            logging.info(f"🔍 Filtering series by length (min: {MIN_SERIES_LENGTH} points)...")
            original_count = len(sampled_df)
        
            ### REFACTOR: Use dynamic target_column_name instead of hard-coded 'target'
            sampled_df = sampled_df[sampled_df[target_column_name].apply(lambda x: len(x) >= MIN_SERIES_LENGTH)]
            filtered_count = len(sampled_df)
        
            if filtered_count == 0:
                logging.error(f" ❌ No valid series after filtering (all too short)")
                return None
        
            if filtered_count < original_count:
                logging.warning(f" ⚠️ Filtered out {original_count - filtered_count} series (too short)")
        
            logging.info(f" ✅ Valid series: {filtered_count}")
        
            ds_sampled = datasets.Dataset.from_pandas(sampled_df)
            ds_sampled.set_format("numpy")
        
            # Split into train/test
            ### REFACTOR: Pass target_column_name to split_dataset
            all_series_data_train, all_series_data_test = split_dataset(ds_sampled, PREDICTION_LENGTH, target_column_name)
            logging.info(f"✅ Dataset loaded and split into train/test")
        
            wql_scores = []
            mase_scores = []
            processed_series_ids = []
            too_short_count = 0
            invalid_count = 0
        
            unique_ids = [ds_sampled[i]['id'] for i in range(len(ds_sampled))]
        
            # Process each series with better error handling (IMPROVED!)
            for series_index, series_id in enumerate(unique_ids):
                try:
                    series_data_train = all_series_data_train[series_index]
                    actual_values_test = all_series_data_test[series_index]
                
                    # Additional validation (NEW!)
                    if len(series_data_train) < 1:
                        too_short_count += 1
                        continue
                    
                    # Check for invalid data (NEW!)
                    if np.any(np.isnan(series_data_train)) or np.any(np.isinf(series_data_train)):
                        invalid_count += 1
                        continue
                
                    if np.any(np.isnan(actual_values_test)) or np.any(np.isinf(actual_values_test)):
                        invalid_count += 1
                        continue
                
                    context = torch.tensor(series_data_train).unsqueeze(0).to(dtype=torch.float32)
                    
                    # Generate predictions
                    forecast_samples = pipeline.predict(context, PREDICTION_LENGTH, num_samples=50)  # Increased num_samples for better stats on GPU
                    forecast_samples_np = np.asarray(forecast_samples)
                
                    # Validate predictions (NEW!)
                    if np.any(np.isnan(forecast_samples_np)) or np.any(np.isinf(forecast_samples_np)):
                        invalid_count += 1
                        continue
                
                    # ✅ FIXED: Corrected quantile computation
                    # Extract first batch, compute quantiles over samples (axis=0), then transpose
                    quantile_forecasts = np.quantile(forecast_samples_np[0], QUANTILE_LEVELS, axis=0).T
                
                    wql = weighted_quantile_loss(actual_values_test, quantile_forecasts, QUANTILE_LEVELS)
                
                    # Validate WQL (NEW!)
                    if np.isnan(wql) or np.isinf(wql):
                        invalid_count += 1
                        continue
                
                    wql_scores.append(wql)
                
                    median_forecast = quantile_forecasts[:, 1]
                
                    seasonality = 1
                    if len(series_data_train) > seasonality:
                        # Use naive seasonal error from context
                        seasonal_diff = series_data_train[seasonality:] - series_data_train[:-seasonality]
                    else:
                        seasonal_diff = np.diff(series_data_train)
                
                    if len(seasonal_diff) == 0 or np.all(seasonal_diff == 0):
                        # Fallback for constant series or very short series
                        seasonal_error = np.array([1.0])
                    else:
                        seasonal_error = seasonal_diff
                    mase = mean_absolute_scaled_error(actual_values_test, median_forecast, seasonal_error)
                
                    # Validate MASE (NEW!)
                    if np.isnan(mase) or np.isinf(mase):
                        invalid_count += 1
                        continue
                
                    mase_scores.append(mase)
                    processed_series_ids.append(str(series_id))
                
                except Exception as e_series:
                    logging.error(f"⚠️ Failed on series {series_id}: {str(e_series)}")
                    logging.error(traceback.format_exc())
                    invalid_count += 1
        
            # Log statistics (IMPROVED!)
            logging.info(f" 📊 Processed: {len(processed_series_ids)} series")
            logging.info(f" 📊 Too short: {too_short_count}, Invalid: {invalid_count}")
        
            # Check if we have valid results (NEW!)
            if len(wql_scores) == 0 or len(mase_scores) == 0:
                logging.warning(f" ⚠️ No valid predictions, setting metrics to NaN")
                wql_geometric_mean = np.nan
                mase_geometric_mean = np.nan
            else:
                # Calculate metrics
                wql_geometric_mean = gmean(np.maximum(wql_scores, 1e-9))
                mase_geometric_mean = gmean(np.maximum(mase_scores, 1e-9))
        
            elapsed_time = time.time() - start_time
        
            # Save results
            results = {
                "dataset_name": config_name,
                "tier": tier_name,
                "target_column": target_column_name, ### REFACTOR: Log the discovered column
                "workers_used": workers,
                "wql_geometric_mean": wql_geometric_mean,
                "mase_geometric_mean": mase_geometric_mean,
                "processed_series_ids": processed_series_ids,
                "num_processed_series": len(processed_series_ids),
                "too_short_count": too_short_count,
                "invalid_count": invalid_count,
                "processing_time_seconds": elapsed_time,
                "streaming_mode": is_streaming
            }
        
            results_filename_pkl = os.path.join(RESULTS_DIR, f"{config_name}_large_results.pkl")
            with open(results_filename_pkl, 'wb') as f:
                pickle.dump(results, f)
        
            results_filename_txt = os.path.join(RESULTS_DIR, f"{config_name}_large_results.txt")
            with open(results_filename_txt, 'w') as f:
                f.write(f"Dataset: {config_name}\n")
                f.write(f"Tier: {tier_name}\n")
                f.write(f"Target Column: {target_column_name}\n") ### REFACTOR: Log the discovered column
                f.write(f"Workers: {workers}\n")
                f.write(f"Geometric Mean WQL: {wql_geometric_mean}\n")
                f.write(f"Geometric Mean MASE: {mase_geometric_mean}\n")
                f.write(f"Num Processed Series: {len(processed_series_ids)}\n")
                f.write(f"Too Short Count: {too_short_count}\n")
                f.write(f"Invalid Count: {invalid_count}\n")
                f.write(f"Processing Time: {elapsed_time:.2f}s\n")
                f.write(f"Streaming Mode: {is_streaming}\n")
        
            logging.info(f" ✅ COMPLETE!")
            logging.info(f" 📊 WQL: {wql_geometric_mean:.6f} | MASE: {mase_geometric_mean:.6f}")
            logging.info(f" ⏱️ Time: {elapsed_time:.2f}s")
        
            # CRITICAL: Clean up this dataset from memory immediately
            del ds_train
            del df_ds, sampled_df, ds_sampled
            import gc
            gc.collect()
        
            return results
    except TimeoutError as e:
        logging.error(f" ❌ TIMEOUT: {config_name} - {str(e)}")
        return None
    except Exception as e_dataset:
        logging.error(f" ❌ ERROR: {config_name}")
        logging.error(f" {str(e_dataset)}")
        logging.error(traceback.format_exc())
        return None
# --- Main Execution ---
def main():
    """Main execution function."""
    logging.info("\n" + "🚀"*40)
    logging.info("STARTING FIXED CHRONOS PROCESSING (v2 - Dynamic Schema)")
    logging.info("🚀"*40 + "\n")
    # Create results dir
    os.makedirs(RESULTS_DIR, exist_ok=True)
    logging.info(f"✅ Results directory: {RESULTS_DIR}")
    # Initial cleanup
    logging.info("\n🧹 Initial cache cleanup...")
    clear_all_cache()
    check_disk_space()
    # Load model
    logging.info("\n📥 Loading Chronos-T5-Large model...")
    try:
        pipeline = ChronosPipeline.from_pretrained(
            "amazon/chronos-t5-large",
            device_map="cuda:0",
            torch_dtype="float16",
        )
        logging.info("✅ Model loaded successfully on GPU")
    except Exception as e:
        logging.error(f"❌ Failed to load model: {e}")
        raise
    # Get prioritized dataset list (NEW!)
    ALL_DATASETS = get_prioritized_dataset_list()
    # Check what's already done
    already_processed = get_processed_datasets()
    logging.info(f"\n✅ Found {len(already_processed)} completed datasets")
    if already_processed:
        logging.info(f" {', '.join(already_processed)}")
    # Filter remaining datasets (now includes SKIP_DATASETS filter!)
    datasets_to_process = [d for d in ALL_DATASETS if d not in SKIP_DATASETS and d not in already_processed]
    logging.info(f"\n{'='*80}")
    logging.info(f"PROCESSING PLAN")
    logging.info(f"{'='*80}")
    logging.info(f"📊 Will process {len(datasets_to_process)} datasets")
    logging.info(f"⏭️ Skipping {len(SKIP_DATASETS)} known problematic datasets")
    logging.info(f"✅ Already completed: {len(already_processed)}")
    logging.info(f"📏 Min series length: {MIN_SERIES_LENGTH} points")
    if datasets_to_process:
        logging.info(f"\n🎯 Next datasets to process:")
        for i, d in enumerate(datasets_to_process[:10], 1):
            tier_name, _ = get_dataset_tier(d)
            logging.info(f" {i}. {d} [{tier_name}]")
        if len(datasets_to_process) > 10:
            logging.info(f" ... and {len(datasets_to_process) - 10} more")
    if len(datasets_to_process) == 0:
        logging.info("✅ All datasets already processed!")
    else:
        # Create smaller batches
        batches = [datasets_to_process[i:i + BATCH_SIZE] for i in range(0, len(datasets_to_process), BATCH_SIZE)]
    
        logging.info(f"\n{'='*80}")
        logging.info(f"BATCH CONFIGURATION")
        logging.info(f"{'='*80}")
        logging.info(f"📦 Total datasets: {len(datasets_to_process)}")
        logging.info(f"📦 Batch size: {BATCH_SIZE}")
        logging.info(f"📦 Number of batches: {len(batches)}")
        logging.info(f"{'='*80}\n")
    
        all_successful = []
        all_failed = []
        all_skipped = []
    
        for batch_num, batch in enumerate(batches, 1):
            logging.info(f"\n{'#'*80}")
            logging.info(f"🎯 BATCH {batch_num}/{len(batches)} - Processing {len(batch)} datasets")
            logging.info(f"{'#'*80}\n")
        
            batch_successful = []
            batch_failed = []
            batch_skipped = []
        
            for i, config_name in enumerate(batch, 1):
                logging.info(f"\n{'='*80}")
                logging.info(f"BATCH {batch_num}/{len(batches)} | Dataset {i}/{len(batch)}: {config_name}")
                logging.info(f"{'='*80}")
            
                # Check disk space before processing
                if not check_disk_space():
                    logging.warning(f"⚠️ Low disk space, clearing cache...")
                    clear_all_cache()
                    check_disk_space()
            
                # Process the dataset
                result = process_dataset(config_name, pipeline)
            
                if result == "skipped":
                    batch_skipped.append(config_name)
                    all_skipped.append(config_name)
                elif result is not None:
                    batch_successful.append(config_name)
                    all_successful.append(config_name)
                    logging.info(f"✅ SUCCESS: {config_name}")
                
                    # CRITICAL: Clear cache after successful processing
                    logging.info(f"\n🧹 Clearing cache after {config_name}...")
                    clear_all_cache()
                    check_disk_space()
                else:
                    batch_failed.append(config_name)
                    all_failed.append(config_name)
                    logging.error(f"❌ FAILED: {config_name}")
                    # Also clear cache after failures
                    clear_all_cache()
            
                # Small delay to let system catch up
                time.sleep(2)
        
            # Batch summary
            logging.info(f"\n{'#'*80}")
            logging.info(f"✅ BATCH {batch_num}/{len(batches)} COMPLETE")
            logging.info(f" Successful: {len(batch_successful)}/{len(batch)}")
            logging.info(f" Skipped: {len(batch_skipped)}/{len(batch)}")
            logging.info(f" Failed: {len(batch_failed)}/{len(batch)}")
            if batch_failed:
                logging.warning(f" Failed: {', '.join(batch_failed)}")
            logging.info(f"{'#'*80}\n")
        
            # Extra cache clear between batches
            if batch_num < len(batches):
                logging.info("🧹 Extra cache clear between batches...")
                clear_all_cache()
                check_disk_space()
    
        # Final summary
        total_processed = len(all_successful) + len(all_skipped)
        logging.info("\n" + "="*80)
        logging.info("🎉 ALL BATCHES COMPLETE!")
        logging.info("="*80)
        logging.info(f"✅ Newly processed: {len(all_successful)}")
        logging.info(f"⏭️ Skipped (already done): {len(all_skipped)}")
        logging.info(f"❌ Failed: {len(all_failed)}")
        logging.info(f"📊 Total completed: {total_processed}/{len(datasets_to_process)}")
    
        if all_successful:
            logging.info(f"\n✅ Newly processed:\n {', '.join(all_successful)}")
        if all_failed:
            logging.warning(f"\n❌ Failed:\n {', '.join(all_failed)}")
        logging.info("="*80)
    # Aggregation
    def aggregate_results(results_dir):
        """Aggregate all results into summary files."""
        logging.info(f"\n📊 Aggregating results...")
    
        all_pkl_files = glob.glob(os.path.join(results_dir, "*_large_results.pkl"))
        all_results_data = []
    
        for pkl_file in all_pkl_files:
            try:
                with open(pkl_file, 'rb') as f:
                    data = pickle.load(f)
                    all_results_data.append(data)
            except Exception as e:
                logging.error(f"Failed to read {pkl_file}: {e}")
    
        if not all_results_data:
            logging.warning("⚠️ No results found to aggregate")
            return
    
        df = pd.DataFrame(all_results_data)
    
        # Include tier info and new target_column info
        cols_order = ['dataset_name', 'tier', 'wql_geometric_mean', 'mase_geometric_mean', 'num_processed_series', 'processing_time_seconds']
    
        other_cols = [col for col in df.columns if col not in cols_order]
        df = df[cols_order + other_cols]
        df = df.sort_values(by="dataset_name")
    
        csv_path = os.path.join(results_dir, "zz_summary_report.csv")
        df.to_csv(csv_path, index=False)
        logging.info(f"✅ Summary CSV: {csv_path}")
    
        all_txt_files = glob.glob(os.path.join(results_dir, "*_large_results.txt"))
        summary_txt_path = os.path.join(results_dir, "zz_summary_report.txt")
    
        with open(summary_txt_path, 'w') as outfile:
            outfile.write("=== MASTER SUMMARY REPORT ===\n")
            outfile.write(df.to_string())
            outfile.write("\n" + "="*80 + "\n")
        
            for fname in sorted(all_txt_files):
                if fname.endswith("zz_summary_report.txt"):
                    continue
                try:
                    with open(fname) as infile:
                        outfile.write(infile.read())
                        outfile.write("\n" + "="*80 + "\n")
                except Exception as e:
                    logging.error(f"Failed to read {fname}: {e}")
    
        logging.info(f"✅ Summary TXT: {summary_txt_path}")
        logging.info(f"\n📊 FINAL STATS:")
        logging.info(f" Total datasets: {len(df)}")
        if len(df) > 0:
            logging.info(f" Avg WQL: {df['wql_geometric_mean'].mean():.4f}")
            logging.info(f" Avg MASE: {df['mase_geometric_mean'].mean():.4f}")
    aggregate_results(RESULTS_DIR)
    logging.info("\n✅ ALL COMPLETE!")
    # Final cache clear
    logging.info("\n🧹 Final cache cleanup...")
    clear_all_cache()
if __name__ == "__main__":
    main()