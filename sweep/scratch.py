# content = open('wirebond_sweep/01_Geometric_Inputs.txt', 'r+').read()
# new_content = 'Wthk = 0.3\n\nFL = 1\n\n' + content[content.index("! Total number of layers in an IGBT"):]
# text_file = open('wirebond_sweep/01_Geometric_Inputs.txt', 'r+')
# text_file.write(new_content)
# text_file.close()

# content = open('wirebond_sweep/Max_Strain_0.3_1.0.txt', 'r+').read()
# # i = content.index('EPS_max')
# print(float(content))

# import pandas as pd
# df1 = pd.read_csv('input_output_part1.csv')
# df2 = pd.read_csv('input_output_part2.csv')
#
# df = pd.concat((df1, df2), ignore_index=True)
#
# df[['WThk', 'FL', 'EPS_max']].to_csv('input_output.csv', index=False)
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats.qmc import Sobol


def scale_to_orig(x, bds):
    res = np.zeros(x.shape)
    el_c = 0
    for el in x:
        for d in range(len(el)):
            res[el_c, d] = (bds[d][1] - bds[d][0]) * el[d] + bds[d][0]
        el_c += 1
    return res

sobolsamp = Sobol(d=2, scramble=False)

bds = [(0.1, 0.49), (0.5, 1.2)] # [WThk, FL]

n_DoE = 128
X = np.round(scale_to_orig(sobolsamp.random(n_DoE), bds), 8)

plt.scatter(X[:, 0], X[:, 1])
plt.show()