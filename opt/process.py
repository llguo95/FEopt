import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_csv('wirebond_history.csv', index_col=0)
plt.plot(np.minimum.accumulate(df['2'].values))
plt.scatter(np.arange(len(df)), df['2'].values)
plt.show()