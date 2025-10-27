import os
import pickle
import glob
import shutil
import time
import signal
from contextlib import contextmanager
from multiprocessing import Pool, cpu_count
from functools import partial
# Set environment variables BEFORE importing
os.environ["DATASETS_VERBOSITY"] = "warning"
import datasets
from datasets import DownloadConfig
from datasets.features import Value as DatasetValue
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
NUM_SAMPLES_PER_SERIES = 100
RESULTS_DIR = "results1"
QUANTILE_LEVELS = [0.1, 0.5, 0.9]
BATCH_SIZE = 4
CACHE_DIR = os.path.expanduser("~/.cache/huggingface")
MIN_FREE_SPACE_GB = 10
MAX_CACHE_SIZE_GB = 20
STREAMING_FALLBACK = True
MIN_SERIES_LENGTH = PREDICTION_LENGTH + 1

# --- NEW: PARALLEL PROCESSING CONFIGURATION ---
# GPU batch size for parallel inference (process multiple series at once)
GPU_BATCH_SIZE = 16  # Process 16 series simultaneously on GPU
# Number of CPU workers for data preprocessing (Map phase)
NUM_CPU_WORKERS = min(cpu_count() - 2, 16)  # Leave 2 cores for system
# Enable/disable parallel processing
ENABLE_PARALLEL = True

# --- DATASET SIZE TIERS ---
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

# Skip problematic datasets
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
def find_target_column(dataset_features):
    """Inspects dataset features to find the first valid numeric Sequence column."""
    numeric_types = (
        DatasetValue(dtype='float32', id=None),
        DatasetValue(dtype='float64', id=None),
        DatasetValue(dtype='int32', id=None),
        DatasetValue(dtype='int64', id=None),
        DatasetValue(dtype='int16', id=None),
        DatasetValue(dtype='int8', id=None)
    )
    ignore_cols = ['id', 'timestamp', 'category', 'item_id']
    for col_name, feature in dataset_features.items():
        if col_name in ignore_cols:
            continue
        if isinstance(feature, datasets.Sequence):
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
    """Context manager for timeout operations."""
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
        
        datasets_cache = os.path.join(CACHE_DIR, "datasets")
        if os.path.exists(datasets_cache):
            shutil.rmtree(datasets_cache, ignore_errors=True)
            os.makedirs(datasets_cache, exist_ok=True)
        
        hub_cache = os.path.join(CACHE_DIR, "hub")
        hub_datasets = os.path.join(hub_cache, "datasets--autogluon--chronos_datasets")
        if os.path.exists(hub_datasets):
            shutil.rmtree(hub_datasets, ignore_errors=True)
        
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
    """Check cache size and clear if needed before download."""
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
    """Check if dataset has already been processed."""
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
    if len(quantiles) != num_quantiles:
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

def split_dataset(ds, prediction_length, target_column_name):
    """Splits each time series in a dataset into training and testing sets."""
    train_data = []
    test_data = []
    for i in range(len(ds)):
        series = ds[i][target_column_name]
        train_data.append(series[:-prediction_length])
        test_data.append(series[-prediction_length:])
    return train_data, test_data

# ═══════════════════════════════════════════════════════════════════════════
# MAPREDUCE PARALLEL PROCESSING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def map_validate_series(args):
    """
    MAP PHASE: Validate and prepare a single series for processing.
    This runs in parallel on CPU workers.
    
    Returns: tuple (series_index, series_id, series_train, series_test, status)
    status can be: 'valid', 'too_short', 'invalid_data'
    """
    series_index, series_id, series_data_train, actual_values_test = args
    
    # Validation checks
    if len(series_data_train) < 1:
        return (series_index, series_id, None, None, 'too_short')
    
    if np.any(np.isnan(series_data_train)) or np.any(np.isinf(series_data_train)):
        return (series_index, series_id, None, None, 'invalid_data')
    
    if np.any(np.isnan(actual_values_test)) or np.any(np.isinf(actual_values_test)):
        return (series_index, series_id, None, None, 'invalid_data')
    
    return (series_index, series_id, series_data_train, actual_values_test, 'valid')

def batch_gpu_inference(pipeline, batch_series_train, prediction_length, num_samples=50):
    """
    Perform batched GPU inference on multiple series at once.
    
    Args:
        pipeline: Chronos model pipeline
        batch_series_train: List of training series
        prediction_length: Number of steps to forecast
        num_samples: Number of samples to generate per series
    
    Returns:
        List of forecast samples for each series in the batch
    """
    # Pad series to same length for batching
    max_len = max(len(s) for s in batch_series_train)
    
    padded_batch = []
    original_lengths = []
    
    for series in batch_series_train:
        original_lengths.append(len(series))
        if len(series) < max_len:
            # Pad with the first value (or could use mean)
            padding = np.full(max_len - len(series), series[0])
            padded_series = np.concatenate([padding, series])
        else:
            padded_series = series
        padded_batch.append(padded_series)
    
    # Convert to tensor batch
    batch_tensor = torch.tensor(np.array(padded_batch), dtype=torch.float32)
    
    # Generate forecasts for entire batch
    with torch.no_grad():
        forecast_samples = pipeline.predict(
            batch_tensor, 
            prediction_length, 
            num_samples=num_samples
        )
    
    # Convert to numpy
    forecast_samples_np = np.asarray(forecast_samples)
    
    # Split back into individual series forecasts
    individual_forecasts = []
    for i in range(len(batch_series_train)):
        individual_forecasts.append(forecast_samples_np[i])
    
    return individual_forecasts

def reduce_compute_metrics(args):
    """
    REDUCE PHASE: Compute metrics for a single series forecast.
    This can run in parallel on CPU workers after GPU inference.
    
    Returns: dict with results or None if failed
    """
    series_id, forecast_samples_np, actual_values_test, series_data_train = args
    
    try:
        # Validate predictions
        if np.any(np.isnan(forecast_samples_np)) or np.any(np.isinf(forecast_samples_np)):
            return None
        
        # Compute quantiles
        quantile_forecasts = np.quantile(forecast_samples_np, QUANTILE_LEVELS, axis=0).T
        
        # Compute WQL
        wql = weighted_quantile_loss(actual_values_test, quantile_forecasts, QUANTILE_LEVELS)
        
        if np.isnan(wql) or np.isinf(wql):
            return None
        
        # Compute MASE
        median_forecast = quantile_forecasts[:, 1]
        
        seasonality = 1
        if len(series_data_train) > seasonality:
            seasonal_diff = series_data_train[seasonality:] - series_data_train[:-seasonality]
        else:
            seasonal_diff = np.diff(series_data_train)
        
        if len(seasonal_diff) == 0 or np.all(seasonal_diff == 0):
            seasonal_error = np.array([1.0])
        else:
            seasonal_error = seasonal_diff
        
        mase = mean_absolute_scaled_error(actual_values_test, median_forecast, seasonal_error)
        
        if np.isnan(mase) or np.isinf(mase):
            return None
        
        return {
            'series_id': str(series_id),
            'wql': wql,
            'mase': mase
        }
    
    except Exception as e:
        logging.debug(f"Error in reduce phase for series {series_id}: {e}")
        return None

def process_series_parallel(pipeline, all_series_data_train, all_series_data_test, unique_ids):
    """
    Process all series using MapReduce-style parallel processing.
    
    Pipeline:
    1. MAP PHASE (CPU parallel): Validate all series
    2. GPU BATCH INFERENCE: Process valid series in batches on GPU
    3. REDUCE PHASE (CPU parallel): Compute metrics for all forecasts
    
    Returns: wql_scores, mase_scores, processed_ids, too_short_count, invalid_count
    """
    logging.info(f"🚀 Starting parallel processing with {NUM_CPU_WORKERS} CPU workers and GPU batch size {GPU_BATCH_SIZE}")
    
    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 1: MAP - Parallel validation on CPU
    # ═══════════════════════════════════════════════════════════════════════
    logging.info("📊 Phase 1: Parallel validation (MAP)")
    
    map_args = [
        (i, unique_ids[i], all_series_data_train[i], all_series_data_test[i])
        for i in range(len(unique_ids))
    ]
    
    with Pool(NUM_CPU_WORKERS) as pool:
        validation_results = pool.map(map_validate_series, map_args)
    
    # Separate valid series from invalid
    valid_series = []
    too_short_count = 0
    invalid_count = 0
    
    for result in validation_results:
        series_index, series_id, series_train, series_test, status = result
        if status == 'valid':
            valid_series.append((series_index, series_id, series_train, series_test))
        elif status == 'too_short':
            too_short_count += 1
        elif status == 'invalid_data':
            invalid_count += 1
    
    logging.info(f"✅ Validation complete: {len(valid_series)} valid, {too_short_count} too short, {invalid_count} invalid")
    
    if len(valid_series) == 0:
        return [], [], [], too_short_count, invalid_count
    
    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 2: GPU Batch Inference
    # ═══════════════════════════════════════════════════════════════════════
    logging.info("🔥 Phase 2: GPU batch inference")
    
    all_forecasts = []
    num_batches = (len(valid_series) + GPU_BATCH_SIZE - 1) // GPU_BATCH_SIZE
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * GPU_BATCH_SIZE
        end_idx = min((batch_idx + 1) * GPU_BATCH_SIZE, len(valid_series))
        
        batch_data = valid_series[start_idx:end_idx]
        batch_series_train = [item[2] for item in batch_data]
        
        # Process entire batch on GPU at once
        batch_forecasts = batch_gpu_inference(
            pipeline, 
            batch_series_train, 
            PREDICTION_LENGTH, 
            num_samples=50
        )
        
        all_forecasts.extend(batch_forecasts)
        
        if (batch_idx + 1) % 10 == 0:
            logging.info(f"  Processed GPU batch {batch_idx + 1}/{num_batches}")
    
    logging.info(f"✅ GPU inference complete: {len(all_forecasts)} forecasts generated")
    
    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 3: REDUCE - Parallel metric computation on CPU
    # ═══════════════════════════════════════════════════════════════════════
    logging.info("📊 Phase 3: Parallel metric computation (REDUCE)")
    
    reduce_args = [
        (valid_series[i][1], all_forecasts[i], valid_series[i][3], valid_series[i][2])
        for i in range(len(valid_series))
    ]
    
    with Pool(NUM_CPU_WORKERS) as pool:
        metric_results = pool.map(reduce_compute_metrics, reduce_args)
    
    # Collect results
    wql_scores = []
    mase_scores = []
    processed_series_ids = []
    
    for result in metric_results:
        if result is not None:
            wql_scores.append(result['wql'])
            mase_scores.append(result['mase'])
            processed_series_ids.append(result['series_id'])
        else:
            invalid_count += 1
    
    logging.info(f"✅ Metric computation complete: {len(processed_series_ids)} successful")
    
    return wql_scores, mase_scores, processed_series_ids, too_short_count, invalid_count

def process_series_sequential(pipeline, all_series_data_train, all_series_data_test, unique_ids):
    """
    Original sequential processing (fallback if parallel is disabled).
    """
    logging.info("🐌 Using sequential processing (parallel disabled)")
    
    wql_scores = []
    mase_scores = []
    processed_series_ids = []
    too_short_count = 0
    invalid_count = 0
    
    for series_index, series_id in enumerate(unique_ids):
        try:
            series_data_train = all_series_data_train[series_index]
            actual_values_test = all_series_data_test[series_index]
            
            if len(series_data_train) < 1:
                too_short_count += 1
                continue
            
            if np.any(np.isnan(series_data_train)) or np.any(np.isinf(series_data_train)):
                invalid_count += 1
                continue
            
            if np.any(np.isnan(actual_values_test)) or np.any(np.isinf(actual_values_test)):
                invalid_count += 1
                continue
            
            context = torch.tensor(series_data_train).unsqueeze(0).to(dtype=torch.float32)
            forecast_samples = pipeline.predict(context, PREDICTION_LENGTH, num_samples=50)
            forecast_samples_np = np.asarray(forecast_samples)
            
            if np.any(np.isnan(forecast_samples_np)) or np.any(np.isinf(forecast_samples_np)):
                invalid_count += 1
                continue
            
            quantile_forecasts = np.quantile(forecast_samples_np[0], QUANTILE_LEVELS, axis=0).T
            wql = weighted_quantile_loss(actual_values_test, quantile_forecasts, QUANTILE_LEVELS)
            
            if np.isnan(wql) or np.isinf(wql):
                invalid_count += 1
                continue
            
            wql_scores.append(wql)
            median_forecast = quantile_forecasts[:, 1]
            
            seasonality = 1
            if len(series_data_train) > seasonality:
                seasonal_diff = series_data_train[seasonality:] - series_data_train[:-seasonality]
            else:
                seasonal_diff = np.diff(series_data_train)
            
            if len(seasonal_diff) == 0 or np.all(seasonal_diff == 0):
                seasonal_error = np.array([1.0])
            else:
                seasonal_error = seasonal_diff
            
            mase = mean_absolute_scaled_error(actual_values_test, median_forecast, seasonal_error)
            
            if np.isnan(mase) or np.isinf(mase):
                invalid_count += 1
                continue
            
            mase_scores.append(mase)
            processed_series_ids.append(str(series_id))
        
        except Exception as e_series:
            logging.error(f"⚠️ Failed on series {series_id}: {str(e_series)}")
            invalid_count += 1
    
    return wql_scores, mase_scores, processed_series_ids, too_short_count, invalid_count

# ═══════════════════════════════════════════════════════════════════════════
# MAIN PROCESSING FUNCTION (with MapReduce integration)
# ═══════════════════════════════════════════════════════════════════════════

def process_dataset(config_name, pipeline):
    """Process a single dataset with MapReduce parallel processing."""
    if dataset_exists(config_name):
        return "skipped"
    
    tier_name, tier_config = get_dataset_tier(config_name)
    timeout_seconds = tier_config["timeout"]
    workers = tier_config["workers"]
    
    logging.info(f"\n{'='*80}")
    logging.info(f"📊 PROCESSING: {config_name}")
    logging.info(f" Tier: {tier_name} | Timeout: {timeout_seconds}s | Workers: {workers}")
    if ENABLE_PARALLEL:
        logging.info(f" 🚀 Parallel Mode: ON | CPU Workers: {NUM_CPU_WORKERS} | GPU Batch: {GPU_BATCH_SIZE}")
    else:
        logging.info(f" 🐌 Parallel Mode: OFF (Sequential)")
    logging.info(f"{'='*80}")
    
    if not check_disk_space():
        logging.error(f"❌ Insufficient disk space to process {config_name}")
        return None
    
    start_time = time.time()
    
    try:
        with timeout(timeout_seconds, f"{config_name} exceeded {timeout_seconds}s timeout"):
            check_and_manage_cache(tier_config.get("max_cache_gb", 10))
            logging.info(f"⬇️ Loading dataset: {config_name}...")
            
            is_streaming = False
            ds_train = None
            
            try:
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
            
            # Dynamic Target Column Discovery
            target_column_name = None
            try:
                target_column_name = tier_config.get("target_override")
                if target_column_name:
                    logging.info(f" Found manual target override: '{target_column_name}'")
                else:
                    if is_streaming:
                        ds_head = ds_train.take(1)
                        features = next(iter(ds_head)).keys()
                        mock_features = {k: ds_train.info.features[k] for k in features}
                        target_column_name = find_target_column(mock_features)
                    else:
                        target_column_name = find_target_column(ds_train.features)
                
                if not target_column_name:
                    raise ValueError("Target column discovery failed.")
            
            except Exception as e_schema:
                logging.error(f" ❌ SCHEMA ERROR: {config_name} - {str(e_schema)}")
                return None
            
            # Convert to pandas DataFrame for sampling
            if is_streaming:
                logging.info("📊 Converting streaming dataset to list...")
                ds_train = datasets.load_dataset(
                    "autogluon/chronos_datasets",
                    config_name,
                    split="train",
                    streaming=True
                )
                ds_list = list(ds_train)
                df_ds = pd.DataFrame(ds_list)
            else:
                ds_train.set_format("numpy")
                df_ds = ds_train.to_pandas()
            
            # Perform sampling
            if 'category' in df_ds.columns:
                sampled_df = df_ds.groupby('category', group_keys=False).apply(
                    lambda x: x.sample(min(len(x), NUM_SAMPLES_PER_SERIES), random_state=42)
                )
            else:
                sampled_df = df_ds.sample(min(len(df_ds), NUM_SAMPLES_PER_SERIES), random_state=42)
            
            # Filter by length
            logging.info(f"🔍 Filtering series by length (min: {MIN_SERIES_LENGTH} points)...")
            original_count = len(sampled_df)
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
            all_series_data_train, all_series_data_test = split_dataset(ds_sampled, PREDICTION_LENGTH, target_column_name)
            logging.info(f"✅ Dataset loaded and split into train/test")
            
            unique_ids = [ds_sampled[i]['id'] for i in range(len(ds_sampled))]
            
            # ═══════════════════════════════════════════════════════════════
            # CHOOSE PROCESSING METHOD: PARALLEL or SEQUENTIAL
            # ═══════════════════════════════════════════════════════════════
            
            if ENABLE_PARALLEL:
                wql_scores, mase_scores, processed_series_ids, too_short_count, invalid_count = \
                    process_series_parallel(pipeline, all_series_data_train, all_series_data_test, unique_ids)
            else:
                wql_scores, mase_scores, processed_series_ids, too_short_count, invalid_count = \
                    process_series_sequential(pipeline, all_series_data_train, all_series_data_test, unique_ids)
            
            # Log statistics
            logging.info(f" 📊 Processed: {len(processed_series_ids)} series")
            logging.info(f" 📊 Too short: {too_short_count}, Invalid: {invalid_count}")
            
            # Calculate metrics
            if len(wql_scores) == 0 or len(mase_scores) == 0:
                logging.warning(f" ⚠️ No valid predictions, setting metrics to NaN")
                wql_geometric_mean = np.nan
                mase_geometric_mean = np.nan
            else:
                wql_geometric_mean = gmean(np.maximum(wql_scores, 1e-9))
                mase_geometric_mean = gmean(np.maximum(mase_scores, 1e-9))
            
            elapsed_time = time.time() - start_time
            
            # Save results
            results = {
                "dataset_name": config_name,
                "tier": tier_name,
                "target_column": target_column_name,
                "workers_used": workers,
                "parallel_processing": ENABLE_PARALLEL,
                "cpu_workers": NUM_CPU_WORKERS if ENABLE_PARALLEL else 0,
                "gpu_batch_size": GPU_BATCH_SIZE if ENABLE_PARALLEL else 1,
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
                f.write(f"Target Column: {target_column_name}\n")
                f.write(f"Workers: {workers}\n")
                f.write(f"Parallel Processing: {ENABLE_PARALLEL}\n")
                if ENABLE_PARALLEL:
                    f.write(f"CPU Workers: {NUM_CPU_WORKERS}\n")
                    f.write(f"GPU Batch Size: {GPU_BATCH_SIZE}\n")
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
            
            # Cleanup
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
    logging.info("CHRONOS PROCESSING WITH MAPREDUCE PARALLEL ACCELERATION")
    logging.info("🚀"*40 + "\n")
    
    if ENABLE_PARALLEL:
        logging.info(f"⚡ Parallel Processing: ENABLED")
        logging.info(f"   CPU Workers: {NUM_CPU_WORKERS}")
        logging.info(f"   GPU Batch Size: {GPU_BATCH_SIZE}")
    else:
        logging.info(f"🐌 Parallel Processing: DISABLED (Sequential mode)")
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    logging.info(f"✅ Results directory: {RESULTS_DIR}")
    
    logging.info("\n🧹 Initial cache cleanup...")
    clear_all_cache()
    check_disk_space()
    
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
    
    ALL_DATASETS = get_prioritized_dataset_list()
    already_processed = get_processed_datasets()
    
    logging.info(f"\n✅ Found {len(already_processed)} completed datasets")
    if already_processed:
        logging.info(f" {', '.join(already_processed)}")
    
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
                
                if not check_disk_space():
                    logging.warning(f"⚠️ Low disk space, clearing cache...")
                    clear_all_cache()
                    check_disk_space()
                
                result = process_dataset(config_name, pipeline)
                
                if result == "skipped":
                    batch_skipped.append(config_name)
                    all_skipped.append(config_name)
                elif result is not None:
                    batch_successful.append(config_name)
                    all_successful.append(config_name)
                    logging.info(f"✅ SUCCESS: {config_name}")
                    
                    logging.info(f"\n🧹 Clearing cache after {config_name}...")
                    clear_all_cache()
                    check_disk_space()
                else:
                    batch_failed.append(config_name)
                    all_failed.append(config_name)
                    logging.error(f"❌ FAILED: {config_name}")
                    clear_all_cache()
                
                time.sleep(2)
            
            logging.info(f"\n{'#'*80}")
            logging.info(f"✅ BATCH {batch_num}/{len(batches)} COMPLETE")
            logging.info(f" Successful: {len(batch_successful)}/{len(batch)}")
            logging.info(f" Skipped: {len(batch_skipped)}/{len(batch)}")
            logging.info(f" Failed: {len(batch_failed)}/{len(batch)}")
            if batch_failed:
                logging.warning(f" Failed: {', '.join(batch_failed)}")
            logging.info(f"{'#'*80}\n")
            
            if batch_num < len(batches):
                logging.info("🧹 Extra cache clear between batches...")
                clear_all_cache()
                check_disk_space()
        
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
        
        cols_order = ['dataset_name', 'tier', 'wql_geometric_mean', 'mase_geometric_mean', 
                      'num_processed_series', 'processing_time_seconds', 'parallel_processing']
        
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
            if ENABLE_PARALLEL:
                avg_time_parallel = df[df['parallel_processing'] == True]['processing_time_seconds'].mean()
                logging.info(f" Avg Processing Time (Parallel): {avg_time_parallel:.2f}s")
    
    aggregate_results(RESULTS_DIR)
    logging.info("\n✅ ALL COMPLETE!")
    logging.info("\n🧹 Final cache cleanup...")
    clear_all_cache()

if __name__ == "__main__":
    main()