import matplotlib.pyplot as plt
from mlxtend.data import iris_data
from mlxtend.preprocessing import standardize
from mlxtend.feature_extraction import LinearDiscriminantAnalysis
from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=1000, centers=3, n_features=3, cluster_std=1)
X = standardize(X)


lda = LinearDiscriminantAnalysis(n_discriminants=2)
lda.fit(X, y)
X_lda = lda.transform(X)

plt.style.use('seaborn-whitegrid')
plt.figure(figsize=(10, 8))

for lab, col in zip((0, 1, 2),
                    ('blue', 'red', 'green')):
    plt.scatter(X_lda[y == lab, 0],
                X_lda[y == lab, 1],
                label=lab,
                c=col)
plt.xlabel('Linear Discriminant 1')
plt.ylabel('Linear Discriminant 2')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()


