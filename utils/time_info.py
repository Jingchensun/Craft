# 1 save the dataset time to time_info_jason

import re
import json
from datetime import timedelta

# Create an empty dictionary to store time information for each dataset
time_dict = {}

for datasetname in ['caltech101', 'dtd', 'eurosat', 'fgvc_aircraft', 'food101', 
'oxford_flowers', 'oxford_pets', 'stanford_cars', 'ucf101']: #'sun397', 'imagenet'
    dataset_info = {"times": [], "seed_times": []}
    for seed in range(1, 4):  # Iterate over three seeds
        seed_times = []
        folder = f"/home/jingchen/promtsrc/output/base2new/train_base/{datasetname}/shots_16/PromptSRC/vit_b16_c2_ep20_batch4_4+4ctx/"
        print(folder)

        file = f"{folder}seed{seed}/log.txt"  # Use seed1, seed2, seed3 for each dataset
        
        # Open the log.txt file to read its content
        with open(file, "r") as file:
            log_content = file.read()

        elapsed_match = re.findall(r'Elapsed:\s+([\d:]+)', log_content)

        if elapsed_match:
            for elapsed_time_str in elapsed_match:
                # Parse the elapsed time in the format "1:58:54" using timedelta
                time_parts = elapsed_time_str.split(':')
                elapsed_time = timedelta(hours=int(time_parts[0]), minutes=int(time_parts[1]), seconds=int(time_parts[2]))
                seed_times.append(elapsed_time.total_seconds() / 60)  # Convert to minutes
            dataset_info["seed_times"].append(seed_times)
            dataset_info["times"].extend(seed_times)
        else:
            dataset_info["seed_times"].append("N/A")  # Set to "N/A" if no match is found
            dataset_info["times"].append("N/A")

    # Calculate average time across three seeds
    dataset_avg_time = sum(dataset_info["times"]) / len(dataset_info["times"])
    time_dict[datasetname] = {"average_time": dataset_avg_time, "seed_times": dataset_info["seed_times"]}

# Save the dictionary containing time information for all datasets as a JSON file
with open("time_info.json", "w") as json_file:
    json.dump(time_dict, json_file, indent=4)

print("Time information saved to time_info.json.")