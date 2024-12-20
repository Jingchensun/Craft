# 获取每个数据集的 Image cache feature, 求Kmeans 均值

import numpy as np
from sklearn.cluster import KMeans

path = "/home/jingchen/promtsrc/lpclip/clip_feat/"
# datasets=['UCF101', 'Caltech101', 'SUN397', 'StanfordCars', 
# 'Food101', 'OxfordFlowers', 'FGVCAircraft', 'EuroSAT', 'DescribableTextures', 'OxfordPets']
datasets = ['ImageNet']

for dataset in datasets:
    split = '/train.npz'
    file_path = path + dataset + split
    data = np.load(file_path)

    # 获取特征和标签
    array_data = data["feature_list"]
    array_label = data["label_list"]

    # 获取不同类别的标签
    # 获取不同类别的标签
    classnames_file = 'txt/' + dataset + '.json'
    with open(classnames_file, 'r') as f:
        unique_labels = json.load(f)
    print('unique_labels:', len(unique_labels))
    print(unique_labels)
    # 初始化一个字典来存储每个类别的特征均值
    class_means = {}

    # 遍历每个类别
    for label in unique_labels:
        # 筛选出当前类别的特征
        label_indices = np.where(array_label == label)[0]
        label_features = array_data[label_indices]
        # print(label_indices)
        
        # 使用 K-means 聚类算法获取当前类别的聚类中心
        # kmeans = KMeans(n_clusters=1, random_state=0).fit(label_features)
        kmeans = KMeans(n_clusters=1, random_state=0, n_init=10).fit(label_features)
        class_mean = kmeans.cluster_centers_[0]
        # print('class_mean:', class_mean.shape)
        
        # 将类别标签及对应的特征均值存储到字典中
        class_means[label] = class_mean

    # 将字典转换为数组，其中每一行代表一个类别的特征均值
    print('before:', len(class_means))
    cluster_centers_concatenated = np.stack(list(class_means.values()), axis=0)
    print(cluster_centers_concatenated.shape)

    # 保存聚类中心向量
    np.save(path + dataset + split + '_k_means.npy', cluster_centers_concatenated)