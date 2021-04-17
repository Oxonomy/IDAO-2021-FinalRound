from sklearn.datasets import make_blobs

import config as c
from utils.cuda import turn_off_gpu


turn_off_gpu()

x, y = make_blobs(n_samples=10000, centers=2, n_features=2, cluster_std=3)

ensemble_models = []
models = []