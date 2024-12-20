import numpy as np
from sklearn.cluster import KMeans
import json

# Base path to the feature files
path = "/home/jingchen/promtsrc/lpclip/clip_feat/"
datasets = ['ImageNet']  # List of datasets to process

for dataset in datasets:
    # Load features and labels
    file_path = path + dataset + '/train.npz'
    data = np.load(file_path)
    features = data["feature_list"]
    labels = data["label_list"]

    # Load unique class labels from JSON file
    classnames_file = 'txt/' + dataset + '.json'
    with open(classnames_file, 'r') as f:
        unique_labels = json.load(f)
    print(f"Number of unique classes: {len(unique_labels)}")
    
    # Compute cluster centers for each class
    class_means = {}
    for label in unique_labels:
        indices = np.where(labels == label)[0]
        class_features = features[indices]

        kmeans = KMeans(n_clusters=1, random_state=0, n_init=10).fit(class_features)
        class_means[label] = kmeans.cluster_centers_[0]

    # Stack cluster centers into a single array
    cluster_centers = np.stack(list(class_means.values()), axis=0)
    print(f"Cluster centers shape: {cluster_centers.shape}")

    # Save the cluster centers
    np.save(path + dataset + '/train_k_means.npy', cluster_centers)