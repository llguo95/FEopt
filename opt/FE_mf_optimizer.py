import sys

import pandas as pd

custom_lib_rel_path = '../../'

sys.path.insert(0, custom_lib_rel_path + 'Python_Benchmark_Test_Optimization_Function_Single_Objective')

sys.path.insert(0, custom_lib_rel_path + 'gpytorch')
import gpytorch

sys.path.insert(0, custom_lib_rel_path + 'GPy')
import GPy

sys.path.insert(0, custom_lib_rel_path + 'MFB')

import time
import pickle
import torch
from matplotlib import pyplot as plt

from MFproblem import MFProblem
from main_new import bo_main, bo_main_unit
from objective_formatter import botorch_TestFunction, AugmentedTestFunction

tkwargs = {
    "dtype": torch.double,
    # "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "device": torch.device("cpu"),
}

dim = 1

problem = [
    MFProblem(
        objective_function=AugmentedTestFunction(
            botorch_TestFunction(
                f(d=dim),
            ), negate=False,
        ).to(**tkwargs)
    )
    for f in fs
]

model_type = ['stmf']
lf = [.9]
noise_types = ['b']
acq_types = ['VFUCB']
cost_ratios = [10]
n_reg = [5]
scramble = 1
noise_fix = 0
budget = 10 # 5 * 3 ** (dim - 1)
tol = 0.02

print()
print('dim = ', dim)

dev = 1
DoE_no = 10
exp_name = 'run_1'
vis_opt = 0

max_vals = pd.read_csv('../max_vals.csv', index_col=0)

benchmark_dict = dict(
    dim=dim,
    problem=[p.objective_function.name for p in problem],
    model_type=model_type,
    lf=lf,
    noise_types=noise_types,
    acq_types=acq_types,
    cost_ratios=cost_ratios,
    n_reg=n_reg,
    scramble=scramble,
    noise_fix=noise_fix,
    budget=budget,
    dev=dev,
    DoE_no=DoE_no,
    exp_name=exp_name,
    vis_opt=vis_opt,
    tol=tol,
)

start = time.time()

for cost_ratio in cost_ratios:
    print('cost_ratio = ', cost_ratio)
    n_reg_lf = [10] # [cost_ratio * 5 ** (dim - 1)]
    iter_thresh = cost_ratio * budget # 5 * cost_ratio * 3 ** (dim - 1)

    for noise_type in noise_types:

        for acq_type in acq_types:

            for problem_i, problem_el in enumerate(problem):
                if problem_i not in [0, 2, 5, 7, 12, 15, 16, 25, 26, 27]: continue
                y_max = max_vals.loc[problem_el.objective_function.name[9:], str(dim)]

                for model_type_el in model_type:
                    if model_type_el == 'sogpr':
                        exp_name_el = exp_name + '_sogpr'
                    else:
                        exp_name_el = exp_name

                    for lf_el in lf:

                        opt_problem_name = str(dim) + '_' + noise_type + '_' + acq_type + '_cr' + str(
                            cost_ratio) + '_lf' + str(lf_el) + '_nh' + str(n_reg[0]) + '_nl' + str(n_reg_lf[0])\
                                           + '_acq' + acq_type

                        for n_DoE in range(DoE_no):
                            start_unit = time.time()

                            bo_main_unit(problem_el=problem_el, model_type_el=model_type_el, lf_el=lf_el,
                                         cost_ratio=cost_ratio, y_max=y_max, tol=tol,
                                         n_reg_init_el=n_reg[0], n_reg_lf_init_el=n_reg_lf[0], scramble=scramble,
                                         vis_opt=vis_opt,
                                         noise_fix=noise_fix, noise_type=noise_type, max_budget=budget,
                                         acq_type=acq_type,
                                         iter_thresh=iter_thresh, dev=dev, opt_problem_name=opt_problem_name,
                                         n_DoE=n_DoE, exp_name=exp_name_el)

                            end_unit = time.time()
                            # print('Unit run time', end_unit - start_unit)

stop = time.time()
print('Total run time', stop - start)

dict_file = open('opt_data/' + exp_name + '/' + exp_name + '.pkl', 'wb')
pickle.dump(benchmark_dict, dict_file)

# plt.show()
