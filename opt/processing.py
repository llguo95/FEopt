import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_csv('wirebond_history_3.csv', index_col=0)
print(df['0'][np.argmin(df['2'].values)])
print(df['1'][np.argmin(df['2'].values)])
print(np.amin(df['2'].values))

fig, axs = plt.subplots(nrows=3, ncols=1, num='History', sharex='all')
its = np.arange(1, len(df) + 1)

axs[0].plot(its, np.minimum.accumulate(df['2'].values), c='r')
axs[0].scatter(its, df['2'].values)
axs[0].scatter(np.argmin(df['2'].values), np.amin(df['2'].values), c='r', marker='*', s=150)
axs[0].set_yscale('log')
axs[0].set_ylabel('peak EPS')

axs[1].scatter(its, df['0'].values)
axs[1].scatter(np.argmin(df['2'].values), df['0'].values[np.argmin(df['2'].values)], c='r', marker='*', s=150)
axs[1].set_ylabel('WThk')

axs[2].scatter(its, df['1'].values)
axs[2].scatter(np.argmin(df['2'].values), df['1'].values[np.argmin(df['2'].values)], c='r', marker='*', s=150)
axs[2].set_ylabel('FL')
axs[2].set_xlabel('Evaluation no.')

for ax in axs:
    ax.axvline(x=16, c='k', linestyle='--')
    ax.grid()

plt.suptitle('Wire-bond optimization history')
plt.tight_layout()
plt.show()