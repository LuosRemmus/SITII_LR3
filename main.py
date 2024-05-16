import pandas as pd
from sklearn.cluster import MeanShift, AgglomerativeClustering, AffinityPropagation
from sklearn.metrics import silhouette_score

# загрузка данных
data = pd.read_parquet("data18.parquet")

# Меняем тип данных в массиве на float64 (необходимо для работы с sklearn)
data = data.astype('float64')

# Обучение модели MeanShift
meanshift = MeanShift()
meanshift_clusters = meanshift.fit_predict(data)
print("MeanShift Silhouette Score:", silhouette_score(data, meanshift_clusters))

# Обучение модели AgglomerativeClustering
agglomerative = AgglomerativeClustering(n_clusters=3)
agglomerative_clusters = agglomerative.fit_predict(data)
print("Agglomerative Clustering Silhouette Score:", silhouette_score(data, agglomerative_clusters))

# Обучение модели AffinityPropagation
affinity_propagation = AffinityPropagation()
affinity_clusters = affinity_propagation.fit_predict(data)
print("Affinity Propagation Silhouette Score:", silhouette_score(data, affinity_clusters))
