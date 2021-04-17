from sklearn.datasets import fetch_california_housing
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
corr = df.corr()

sns.set_theme()
mask = np.triu(np.ones_like(corr, dtype=bool))
cmap = sns.diverging_palette(240, 20, as_cmap=True)
f, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, annot=True, mask=mask, cmap=cmap, linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)
plt.show()
