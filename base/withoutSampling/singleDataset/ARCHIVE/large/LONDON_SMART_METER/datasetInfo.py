from datasets import load_from_disk

# Load the dataset
dataset = load_from_disk('/home/h20250169/study/modelTraining/dm_eval_2/base/withoutSampling/ALLDATASET/chronos_datasets_monash_london_smart_meters')

print(f"Dataset type: {type(dataset)}")
print(f"Dataset keys: {dataset.keys()}")
print(f"Dataset structure: {dataset}")

# Check each split
for split_name in dataset.keys():
    split_data = dataset[split_name]
    print(f"\n--- {split_name} split ---")
    print(f"Length: {len(split_data)}")
    print(f"Features: {split_data.features}")
    if len(split_data) > 0:
        print(f"First item keys: {split_data[0].keys()}")