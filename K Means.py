import numpy as np
from time import time
from sklearn.preprocessing import scale
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn import metrics

np.random.seed(42)

digits = load_digits()
data = scale(digits.data)

n_samples, n_features = data.shape
labels = digits.target
n_digits = len(np.unique(labels))

sample_size = 300

print(82 * '_')

print(f'''
      Digits number: {n_digits},
      Samples number: {n_samples},
      Features number: {n_features}
''')

def bench_k_means(estimator, name, data):
    t0 = time()
    estimator.fit(data)
    print(f'''
          init: {name:10},
          time: {time() - t0:.2f},
          inertia: {int(estimator.inertia_)},
          homo: {metrics.homogeneity_score(labels, estimator.labels_):.3f},
          compl: {metrics.completeness_score(labels, estimator.labels_):.3f},
          v-meas: {metrics.v_measure_score(labels, estimator.labels_):.3f},
          ARI: {metrics.adjusted_rand_score(labels, estimator.labels_):.3f},
          AMI: {metrics.adjusted_mutual_info_score(labels, estimator.labels_):.3f},
          silhouette: {metrics.silhouette_score(data, estimator.labels_):.3f}
    ''')
    
    
clf = KMeans(init='k-means++', n_clusters=n_digits, n_init=10)
bench_k_means(clf, 'k-means++', data)

print(82 * '_')
