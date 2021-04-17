from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def plot_correlation_matrix(corr: np.array, size=(10, 8)):
    sns.set_theme()

    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(240, 20, as_cmap=True)
    f, ax = plt.subplots(figsize=size)
    sns.heatmap(corr, annot=True, mask=mask, cmap=cmap, linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)
    plt.show()


def plot_scatterplot_matrix(data: pd.DataFrame, hist_bins=50, kde_levels=4):
    sns.set()

    g = sns.pairplot(data, kind="scatter", diag_kind="kde", plot_kws=dict(marker=".", linewidth=1, color="b"), size=10)
    g.map_lower(sns.histplot, **{'bins': hist_bins, 'pthresh': .1, 'color': "b"})
    g.map_lower(sns.kdeplot, **{'levels': kde_levels, 'color': "gray", 'linewidth': 1})
    plt.show()
