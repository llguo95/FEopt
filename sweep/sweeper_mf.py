import os
import sys

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

def objective_function(x: np.ndarray, MeshSize_XY: float) -> np.ndarray:
    in_content = open('wirebond_mf_sweep/01_Geometric_Inputs_original.txt', 'r').read()
    new_in_content = 'WThk = ' + str(x[0]) + '\n\nFL = ' + str(x[1]) + '\n\n' + \
                     in_content[in_content.index("!Loop Heights: (Limit 0.6 to 2.3)"):]
    text_file = open('wirebond_mf_sweep/01_Geometric_Inputs.txt', 'w')
    text_file.write(new_in_content)
    text_file.close()

    in_content_lf = open('wirebond_mf_sweep/02_Mesh_Parameters_original.txt', 'r').read()
    new_in_content_lf = 'MeshSize_XY = ' + str(MeshSize_XY) + '\n\n' + \
                        in_content_lf[in_content_lf.index("!Mesh Divisions Along Thickness of Each Layer: Z-direction"):]
    text_file_lf = open('wirebond_mf_sweep/02_Mesh_Parameters.txt', 'w')
    text_file_lf.write(new_in_content_lf)
    text_file_lf.close()

    cmdl = '"ansys2019r3" -p ansys -dis -mpi INTELMPI -np 2 -lch -dir "wirebond_mf_sweep" -j "wirebond_ms_' \
           + str(MeshSize_XY) + '" -s read -l en-us -b -i "wirebond_mf_sweep/01_Geometric_Inputs.txt"'
    os.system(cmdl)

    out_path = 'wirebond_mf_sweep/Max_Strain_' + str(MeshSize_XY) + '_' + str(x[0]) + '_' + str(x[1]) + '.txt'
    if os.path.exists(out_path):
        out_content = open(out_path, 'rb').read()
        res = np.array(float(out_content))
    else:
        res = np.nan
    return res

sobolsamp = Sobol(d=2, scramble=False)

bds = [(0.1, 0.49), (0.5, 1.2)] # [WThk, FL]

n_DoE = 64
X = np.round(scale_to_orig(sobolsamp.random(n_DoE), bds), 8)
# print(X)

df = pd.DataFrame(X)
df.to_csv('input.csv')

input_output_lines = []

meshsizes = [0.5, 1.0]
MeshSize_XY = meshsizes[int(sys.argv[1])]
for i, x in enumerate(X):
    y = objective_function(x, MeshSize_XY)
    input_output_line = np.hstack((x, y))
    input_output_lines.append(input_output_line)
    df_new = pd.DataFrame(input_output_lines, columns=['WThk', 'FL', 'EPS_max'])
    df_new.to_csv('input_output_ms_' + str(MeshSize_XY) + '.csv')