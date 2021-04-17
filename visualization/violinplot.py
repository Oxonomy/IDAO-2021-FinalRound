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
numerical = ['MedInc', 'HouseAge', 'AveRooms', 'AveOccup', 'Latitude', 'Longitude']
df['big_bedrm'] = df['AveBedrms'] > 1
df = df[numerical + ['big_bedrm']][:1000]


fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(10, 7))
for idx, feat in enumerate(numerical):
    ax = axes[int(idx / 4), idx % 4]
    sns.violinplot(x='big_bedrm', y=feat, data=df, ax=ax, scale='count')
    ax.set_xlabel('')
    ax.set_ylabel(feat)
fig.tight_layout()
plt.show()
