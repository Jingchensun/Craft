import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def load_data(file_path):
    """Load data and return features and labels."""
    data = np.load(file_path)
    return data["feature_list"], data["label_list"]

def extract_class_features(array_data, array_label, unique_labels):
    """Extract features for each class."""
    class_features = {}
    for label in unique_labels:
        label_indices = np.where(array_label == label)[0]
        class_features[label] = array_data[label_indices]
    return class_features

def concatenate_selected_features(class_features, selected_labels):
    """Concatenate features for selected classes."""
    return np.concatenate([class_features[label] for label in selected_labels], axis=0)

def plot_tsne(embedded_features, array_label, selected_labels, title, subplot_position):
    """Plot t-SNE visualization."""
    plt.subplot(subplot_position)
    for label in selected_labels:
        label_indices = np.where(array_label == label)[0]
        label_embedded_features = embedded_features[label_indices]
        plt.scatter(label_embedded_features[:, 0], label_embedded_features[:, 1], label=label, alpha=0.5)
    plt.title(title, fontsize=24)
    # Remove plot borders and ticks
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

# Set paths
base_path = "/home/jingchen/promtsrc/lpclip/prompt_feat_test/"
dataset = 'EuroSAT/'
file_path_baseline = base_path + 'prompt_base_euro/' + 'test.npz'
file_path_ours = base_path + 'our_base_euro/' + 'test.npz'

# Load data for baseline and our method
features_baseline, labels_baseline = load_data(file_path_baseline)
features_ours, labels_ours = load_data(file_path_ours)

# Get unique class labels
unique_labels = np.unique(labels_baseline)

# Extract features for each class
class_features_baseline = extract_class_features(features_baseline, labels_baseline, unique_labels)
class_features_ours = extract_class_features(features_ours, labels_ours, unique_labels)

# Select the first 10 classes
selected_labels = unique_labels[:10]

# Concatenate features for the selected classes
selected_features_baseline = concatenate_selected_features(class_features_baseline, selected_labels)
selected_features_ours = concatenate_selected_features(class_features_ours, selected_labels)

# Create a shared TSNE object and apply dimensionality reduction
tsne = TSNE(n_components=2, random_state=42)
embedded_features_baseline = tsne.fit_transform(selected_features_baseline)
embedded_features_ours = tsne.fit_transform(selected_features_ours)

# Plot t-SNE visualizations
plt.figure(figsize=(20, 8))

# t-SNE plot for baseline method
plot_tsne(embedded_features_baseline, labels_baseline, selected_labels, 'Maple w/o $\\mathcal{L}_{ID}$ (Acc=94.1 on EuroSAT)', 121)

# t-SNE plot for our method
plot_tsne(embedded_features_ours, labels_ours, selected_labels, 'Maple w/ $\\mathcal{L}_{ID}$ (Acc=96.3 on EuroSAT)', 122)

# Show and save the plot
plt.show()
plt.savefig("tsne_comparison.pdf", format='pdf', dpi=300)