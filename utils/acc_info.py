# 2 evaulate on base2new dataset

import re
import json
import numpy as np
import os
import shutil
from datetime import datetime, timedelta

dataset_info = []

empty = 'sun397', 'imagenet'
dataset1 = ['oxford_pets','oxford_flowers','fgvc_aircraft', 'dtd', 'eurosat', 
                    'stanford_cars', 'food101', 'caltech101', 
                    'ucf101', 'sun397', 'imagenet']
dataset2 = ['oxford_pets','oxford_flowers','fgvc_aircraft', 'dtd', 'eurosat', 
                    'stanford_cars', 'food101', 'caltech101', 
                    'ucf101', 'sun397', 'imagenet',
                    'imagenetv2', 'imagenet_sketch', 'imagenet_a', 'imagenet_r'] 

for datasetname in dataset2:
    data_save = []  # Create a new list to store information for each dataset

    # Promptscr Train
    # folder = "/home/jingchen/promtsrc/output/base2new/train_base/" + datasetname + "/shots_16/PromptSRC/vit_b16_c2_ep20_batch4_4+4ctx/"
    # Promptscr novel test
    # folder = "/home/jingchen/promtsrc/output/base2new/test_new/" + datasetname + "/shots_16/PromptSRC/vit_b16_c2_ep20_batch4_4+4ctx/"
    # Promptscr cross dataset train
    folder = "/home/jingchen/promtsrc/output/imagenet/PromptSRC/vit_b16_c2_ep20_batch4_4+4ctx_cross_datasets-B128_16shots/"
    # Promptscr cross dataset test
    # folder = "/home/jingchen/promtsrc/output/evaluation/PromptSRC/vit_b16_c2_ep20_batch4_4+4ctx_cross_datasets-B128_16shots/" + datasetname + '/' 
    # Maple Train
    # folder = "/home/jingchen/promtsrc/output/base2new/train_base/" + datasetname + "/shots_16/MaPLe/vit_b16_c2_ep5_batch4_2ctx/"
    # Maple Test
    # folder = "/home/jingchen/promtsrc/output/base2new/test_new/" + datasetname + "/shots_16/MaPLe/vit_b16_c2_ep5_batch4_2ctx/"

    accuracies = []  # Used to store accuracies for three seed loops

    for i in range(1, 4):
        accuracy = {}  # Create a new dictionary to store dataset information
        seed_folder = 'seed' + str(i)
        file = os.path.join(folder, seed_folder, "log.txt")
        if os.path.exists(file) == False:
            break
        # Copy the log.txt file to the destination folder
        # destination_folder = os.path.join(os.path.expanduser("./"), "output_txt", datasetname, "mpt_cyclip_12", seed_folder)
        destination_folder = os.path.join(os.path.expanduser("./"), "output_txt", datasetname, "mpt-base", seed_folder)
        os.makedirs(destination_folder, exist_ok=True)  # 创建目标文件夹，如果不存在的话
        shutil.copy2(file, destination_folder)  # 复制文件

        # Open the log.txt file to read its content
        with open(file, "r") as file:
            log_content = file.read()

        # Use regular expressions to find the starting position of "Evaluate on the *test* set"
        start_pos = log_content.find("Evaluate on the *test* set")

        if start_pos != -1:
            # Find "accuracy: xxx.xxx%" after "Evaluate on the *test* set"
            accuracy_match = re.search(r'accuracy:\s+(\d+\.\d+%)', log_content[start_pos:])
            
            if accuracy_match:
                accuracy_value = float(accuracy_match.group(1)[:-1])  # Remove the percentage sign and convert to float
                accuracies.append(accuracy_value)
            else:
                accuracy_value = "N/A"  # Set to "N/A" if no match is found

            elapsed_match = re.search(r'Elapsed:\s+([\d:]+)', log_content[start_pos:])

            if elapsed_match:
                elapsed_time_str = elapsed_match.group(1)
                # Parse the elapsed time in the format "1:58:54" using datetime
                time_parts = elapsed_time_str.split(':')
                elapsed_time = timedelta(hours=int(time_parts[0]), minutes=int(time_parts[1]), seconds=int(time_parts[2]))
                accuracy['time'] = str(elapsed_time)
            else:
                accuracy['time'] = "N/A"  # Set to "N/A" if no match is found

            # Create a dictionary containing accuracy and time information
            accuracy['seed'] = seed_folder
            accuracy['accuracy'] = accuracy_value

            # Add the current dataset's information to the list
            data_save.append(accuracy)

    # Calculate mean and variance
    mean_accuracy = round(np.mean(accuracies), 3)
    variance_accuracy = round(np.var(accuracies), 3)

    # Add mean and variance to dataset information
    dataset_info.append({
        datasetname: {
            "data": data_save,
            "mean": mean_accuracy,
            "var": variance_accuracy
        }
    })

# Save the list containing information for two datasets as a JSON file
with open("accuracy.json", "w") as json_file:
    json.dump(dataset_info, json_file, indent=4)

print("Accuracy information saved to accuracy.json.")
