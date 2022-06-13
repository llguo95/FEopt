### 1 Setup
### 1.1 Import packages, define helper functions

import numpy as np
import matplotlib.pyplot as plt
import GPy
import pandas as pd

# from torch.quasirandom import SobolEngine

from emukit.model_wrappers.gpy_model_wrappers import GPyModelWrapper
from emukit.bayesian_optimization.acquisitions import ExpectedImprovement
from emukit.core.optimization import GradientAcquisitionOptimizer
from emukit.core import ParameterSpace, ContinuousParameter

import joblib

import os

import pickle

from scipy.stats.qmc import Sobol

from sklearn.preprocessing import StandardScaler

np.random.seed(123)

def scale_to_orig(x, bds):
    res = np.zeros(x.shape)
    el_c = 0
    for el in x:
        for d in range(len(el)):
            res[el_c, d] = (bds[d][1] - bds[d][0]) * el[d] + bds[d][0]
        el_c += 1
    return res

### 1.2 Define objective function

# def objective_function(x: np.ndarray) -> np.ndarray:
#     input_file = open('input.pkl', 'wb')
#     pickle.dump(x, input_file)
#     input_file.close()
#
#     cmdl = 'python call_FEA.py'
#     os.system(cmdl)
#
#     output_file = open('output.pkl', 'rb')
#     res = pickle.load(output_file)
#     output_file.close()
#
#     return res  # np.sum(x ** 2, axis=1)[:, None]

def objective_function(x: np.ndarray) -> np.ndarray:
    in_content = open('wirebond_opt/01_Geometric_Inputs_original.txt', 'r').read()
    new_in_content = 'WThk = ' + str(x[0]) + '\n\nFL = ' + str(x[1]) + '\n\n' + in_content[in_content.index("! Total number of layers in an IGBT"):]
    text_file = open('wirebond_opt/01_Geometric_Inputs.txt', 'w')
    text_file.write(new_in_content)
    text_file.close()

    out_path = 'wirebond_sweep/Max_Strain_' + str(x[0]) + '_' + str(x[1]) + '.txt'
    if os.path.exists(out_path):
        out_content = open(out_path, 'rb').read()
        res = np.array(float(out_content))
    else:
        print('Simulation with WThk = ' + str(x[0]) + ', FL = ' + str(x[1]) + ' failed. Calling surrogate.')
        m_load = GPy.models.GPRegression(X_scaled, Y_scaled)
        m_load.update_model(False)  # do not call the underlying expensive algebra on load
        m_load.initialize_parameter()  # Initialize the parameters (connect the parameters up)
        m_load[:] = np.load('../sweep/gpy_model.npy', allow_pickle=True)  # Load the parameters
        m_load.update_model(True)  # Call the algebra only once

        scaler_X_load = joblib.load('../sweep/scaler_X.gz')
        scaler_Y_load = joblib.load('../sweep/scaler_Y.gz')
        mean_scaled, _ = m_load.predict(scaler_X_load.transform(x))

        res = scaler_Y_load.inverse_transform(mean_scaled)
    return res

dim = 2
bds = [(0.1, 0.49), (0.5, 1.2)] # [WThk, FL]

### 1.3 Initial data

sobolsamp = Sobol(d=dim, scramble=False)

n_DoE = 4
X = np.round(scale_to_orig(sobolsamp.random(n_DoE), bds), 8)
Y = []
for i, x in enumerate(X):
    Y.append(objective_function(x))

    ### Save data per iteration
    total_history = np.hstack((X[:i + 1], np.array(Y)[:, None]))
    df = pd.DataFrame(total_history)
    df.to_csv('wirebond_history.csv')

Y = np.array(Y)[:, None]

test_x_axes = [np.linspace(bds[k][0], bds[k][1], 50) for k in range(dim)]
test_x = np.meshgrid(*test_x_axes)
test_x_list = np.hstack([layer.reshape(-1, 1) for layer in test_x])

# exact_y_list = objective_function(test_x_list)

scaler_X = StandardScaler()
scaler_X.fit(X)

scaler_Y = StandardScaler()
scaler_Y.fit(Y)

X_scaled = scaler_X.transform(X)
Y_scaled = scaler_Y.transform(Y)

test_x_list_scaled = scaler_X.transform(test_x_list)

### 1.4 Setup acquisition optimizer

space = ParameterSpace(
    # [ContinuousParameter('x_%d' % i, bds[i][0], bds[i][1]) for i in range(dim)]
    [ContinuousParameter('x_%d' % i, min(test_x_list_scaled[:, i]), max(test_x_list_scaled[:, i])) for i in range(dim)]
)

optimizer = GradientAcquisitionOptimizer(space)

### 2 Optimization
### 2.1 Optimization loop definition

def BO_loop(X_scaled, Y_scaled):
    gpy_model = GPy.models.GPRegression(X_scaled, Y_scaled, GPy.kern.RBF(dim))

    # gpy_model.likelihood.constrain_bounded(1e-10, 1)
    # gpy_model.likelihood.fix(1e-10)
    # gpy_model.optimize_restarts(
    #     num_restarts=5,
    #     verbose=False,
    # )

    gpy_model.optimize()

    emukit_model = GPyModelWrapper(gpy_model)

    ei_acquisition = ExpectedImprovement(emukit_model, jitter=0)
    # ei_list = ei_acquisition.evaluate(test_x_list_scaled)

    x_new_scaled, _ = optimizer.optimize(ei_acquisition)
    y_new = np.array(
        [objective_function(
            np.round(scaler_X.inverse_transform(x_new_scaled)[0], 8)
        )]
    )[:, None]

    X_scaled = np.append(X_scaled, x_new_scaled, axis=0)
    Y_scaled = np.append(Y_scaled, scaler_Y.transform(y_new), axis=0)
    # Y = scaler_Y.inverse_transform(Y_scaled)
    # scaler_Y.fit(Y)
    # Y_scaled = scaler_Y.transform(Y)

    emukit_model.set_data(X_scaled, Y_scaled)

    return X_scaled, Y_scaled, emukit_model

### 2.2 Running the optimization loop

### Save data per iteration
input_history = scaler_X.inverse_transform(X_scaled)
output_history = scaler_Y.inverse_transform(Y_scaled)
total_history = np.hstack((input_history, output_history))

n_iter = 100
for _ in range(n_iter):
    X_scaled, Y_scaled, emukit_model = BO_loop(X_scaled, Y_scaled)

    ### Save data per iteration
    input_history = scaler_X.inverse_transform(X_scaled)
    output_history = scaler_Y.inverse_transform(Y_scaled)
    total_history = np.hstack((input_history, output_history))
    df = pd.DataFrame(total_history)
    df.to_csv('wirebond_history.csv')

### 3 Post-processing
### 3.1 History and response surface collection

# input_history = scaler_X.inverse_transform(X_scaled)
# output_history = scaler_Y.inverse_transform(Y_scaled)
# total_history = np.hstack((input_history, output_history))
# print(total_history)

# mu_plot, var_plot = emukit_model.predict(test_x_list_scaled)
# mu_plot, var_plot = scaler_Y.inverse_transform(mu_plot), scaler_Y.inverse_transform(var_plot)
#
# print('Initial DoE \n', input_history[:n_DoE]); print()
# print('Input iteration history \n', input_history[n_DoE:]); print()
# print('Output history \n', output_history)

### 3.2 Visualzization

vis = 0

if vis:
    plt.figure()
    plt.plot(range(1, n_iter + 1), np.minimum.accumulate(output_history[n_DoE:]), '-o')
    plt.plot(range(1, n_iter + 1), output_history[n_DoE:], 'o', alpha=.5)
    plt.xlabel('Iteration no.')
    plt.ylabel('Cumulative minimum')
    plt.title('Optimization output history')
    plt.tight_layout()
    plt.grid()

    if dim == 1:
        plt.figure(figsize=(6, 4))
        plt.plot(input_history, output_history, "ro", markersize=10, label="Previous observations")
        plt.plot(input_history[-1], output_history[-1], "bo", markersize=10, label="Latest observation")
        # plt.plot(test_x_list, exact_y_list, "k", label="Objective Function")
        # plt.plot(test_x_list, mu_plot, "C0", label="Model")
        # plt.fill_between(test_x_list[:, 0],
        #                  mu_plot[:, 0] + np.sqrt(var_plot)[:, 0],
        #                  mu_plot[:, 0] - np.sqrt(var_plot)[:, 0], color="C0", alpha=0.6)
        # plt.fill_between(test_x_list[:, 0],
        #                  mu_plot[:, 0] + 2 * np.sqrt(var_plot)[:, 0],
        #                  mu_plot[:, 0] - 2 * np.sqrt(var_plot)[:, 0], color="C0", alpha=0.4)
        # plt.fill_between(test_x_list[:, 0],
        #                  mu_plot[:, 0] + 3 * np.sqrt(var_plot)[:, 0],
        #                  mu_plot[:, 0] - 3 * np.sqrt(var_plot)[:, 0], color="C0", alpha=0.2)
        plt.legend(prop={'size': 10})
        plt.xlabel(r"$x$")
        plt.ylabel(r"$f(x)$")
        plt.title('Objective function and response')
        plt.grid(True)
        # c = (max(exact_y_list) - min(exact_y_list)) / 10
        # plt.ylim([min(exact_y_list) - c, max(exact_y_list) + c])
        plt.tight_layout()

    plt.show()