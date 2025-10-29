import requests
from datasets import load_dataset

# Get all configurations from the splits endpoint
url = "https://datasets-server.huggingface.co/splits?dataset=autogluon%2Fchronos_datasets"
response = requests.get(url)
data = response.json()

configs = set(item['config'] for item in data['splits'])

# Download each configuration
for config in configs:
    print(f"Downloading configuration: {config}")
    dataset = load_dataset("autogluon/chronos_datasets", config)
    dataset.save_to_disk(f"chronos_datasets_{config}")