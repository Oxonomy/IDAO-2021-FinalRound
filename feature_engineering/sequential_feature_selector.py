import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from mlxtend.feature_selection import SequentialFeatureSelector
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs

X, y = make_regression(n_samples=1000, n_targets=1, n_features=10, n_informative=3, noise=0.1)


sfs = SequentialFeatureSelector(LinearRegression(),
                                k_features=3,
                                forward=True,
                                floating=False,
                                verbose=1,
                                scoring='neg_mean_squared_error',
                                cv=2)

sfs.fit(X, y)
fig = plot_sfs(sfs.get_metric_dict(), kind='std_err')
print(sfs.k_feature_idx_)
plt.title('Sequential Forward Selection (w. StdErr)')
plt.grid()
plt.show()
