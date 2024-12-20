import os
import re
import json
import numpy as np
import shutil
from datetime import timedelta

# Configuration
BASE_PATH = "/home/jingchen/promtsrc/output/imagenet/PromptSRC/vit_b16_c2_ep20_batch4_4+4ctx_cross_datasets-B128_16shots/"
OUTPUT_DIR = "./output_txt"
DATASETS = ['ImageNet']
SEEDS = [1, 2, 3]

# Helper function to parse log file
def parse_log(file_path):
    with open(file_path, "r") as file:
        log_content = file.read()

    start_pos = log_content.find("Evaluate on the *test* set")
    if start_pos == -1:
        return None, None

    # Extract accuracy
    accuracy_match = re.search(r'accuracy:\s+(\d+\.\d+%)', log_content[start_pos:])
    accuracy = float(accuracy_match.group(1)[:-1]) if accuracy_match else None

    # Extract elapsed time
    elapsed_match = re.search(r'Elapsed:\s+([\d:]+)', log_content[start_pos:])
    elapsed_time = None
    if elapsed_match:
        time_parts = map(int, elapsed_match.group(1).split(':'))
        elapsed_time = str(timedelta(hours=next(time_parts), minutes=next(time_parts), seconds=next(time_parts)))

    return accuracy, elapsed_time

# Main processing loop
dataset_info = []
for dataset in DATASETS:
    accuracies = []
    data_save = []

    for seed in SEEDS:
        seed_folder = f"seed{seed}"
        log_file = os.path.join(BASE_PATH, seed_folder, "log.txt")

        if not os.path.exists(log_file):
            continue

        # Copy log file to output directory
        dest_folder = os.path.join(OUTPUT_DIR, dataset, "mpt-base", seed_folder)
        os.makedirs(dest_folder, exist_ok=True)
        shutil.copy2(log_file, dest_folder)

        # Parse log file
        accuracy, elapsed_time = parse_log(log_file)
        if accuracy is not None:
            accuracies.append(accuracy)

        # Save individual seed info
        data_save.append({
            "seed": seed_folder,
            "accuracy": accuracy if accuracy is not None else "N/A",
            "time": elapsed_time if elapsed_time is not None else "N/A"
        })

    # Compute mean and variance
    mean_accuracy = round(np.mean(accuracies), 3) if accuracies else "N/A"
    variance_accuracy = round(np.var(accuracies), 3) if accuracies else "N/A"

    # Save dataset info
    dataset_info.append({
        dataset: {
            "data": data_save,
            "mean": mean_accuracy,
            "var": variance_accuracy
        }
    })

# Save results to JSON
with open("accuracy.json", "w") as json_file:
    json.dump(dataset_info, json_file, indent=4)

print("Accuracy information saved to accuracy.json.")