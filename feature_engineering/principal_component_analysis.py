import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid")

from mlxtend.preprocessing import standardize
from mlxtend.feature_extraction import PrincipalComponentAnalysis
from sklearn.datasets import make_regression


X, y = make_regression(n_samples=1000, n_targets=1, n_features=5, n_informative=3, noise=0.1)
X = standardize(X)

pca = PrincipalComponentAnalysis(n_components=2)
pca.fit(X)
X_pca = pca.transform(X)

sns.relplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y)
sns.relplot(x=X[:, 0], y=X[:, 1], hue=y)
plt.show()
