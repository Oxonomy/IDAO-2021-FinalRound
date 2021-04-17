import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

warnings.filterwarnings('ignore')
sns.set()

data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df = df[['MedInc', 'HouseAge', 'AveRooms', 'AveOccup', 'Latitude', 'Longitude']][:1000]

g = sns.pairplot(df, kind="scatter", diag_kind="kde", plot_kws=dict(marker=".", linewidth=1, color="b"), height=2)
g.map_lower(sns.histplot, **{'bins': 50, 'pthresh': .1, 'color': "b"})
g.map_lower(sns.kdeplot, **{'levels': 4, 'color': "gray", 'linewidth': 1})
plt.show()
