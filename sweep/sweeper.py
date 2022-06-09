import os

import pandas as pd

import numpy as np

from scipy.stats.qmc import Sobol

def scale_to_orig(x, bds):
    res = np.zeros(x.shape)
    el_c = 0
    for el in x:
        for d in range(len(el)):
            res[el_c, d] = (bds[d][1] - bds[d][0]) * el[d] + bds[d][0]
        el_c += 1
    return res

def objective_function(x: np.ndarray) -> np.ndarray:
    in_content = open('wirebond_sweep/01_Geometric_Inputs_original.txt', 'r').read()
    new_in_content = 'WThk = ' + str(x[0]) + '\n\nFL = ' + str(x[1]) + '\n\n' + in_content[in_content.index("! Total number of layers in an IGBT"):]
    text_file = open('wirebond_sweep/01_Geometric_Inputs.txt', 'w')
    text_file.write(new_in_content)
    text_file.close()

    cmdl = '"ansys2019r3" -p ansys -dis -mpi INTELMPI -np 2 -lch -dir "wirebond_sweep" -j "wirebond_tutorial" -s read -l en-us -b -i "wirebond_sweep/01_Geometric_Inputs.txt"'
    os.system(cmdl)

    out_content = open('wirebond_sweep/Max_Strain_' + str(x[0]) + '_' + str(x[1]) + '.txt', 'rb').read()
    res = np.array(float(out_content))
    return res

sobolsamp = Sobol(d=2, scramble=False)

bds = [(0.1, 0.49), (0.5, 1.2)] # [WThk, FL]

n_DoE = 256
X = np.round(scale_to_orig(sobolsamp.random(n_DoE), bds), 8)[47:]
# print(X)

df = pd.DataFrame(X)
# df.to_csv('input.csv')

input_output_lines = []
for i, x in enumerate(X):
    y = objective_function(x)
    input_output_line = np.hstack((x, y))
    input_output_lines.append(input_output_line)
    df_new = pd.DataFrame(input_output_lines, columns=['WThk', 'FL', 'EPS_max'])
    df_new.to_csv('input_output.csv')