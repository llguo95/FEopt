import GPy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import joblib

from emukit.model_wrappers import GPyModelWrapper

def uniform_grid(bl, tr, n, mesh=False):
    coord_axes = [np.linspace(bl_c, tr_c, n_c) for (bl_c, tr_c, n_c) in zip(bl, tr, n)]
    coord_mesh = np.array(np.meshgrid(*coord_axes))
    s = coord_mesh.shape
    coord_list = coord_mesh.reshape((s[0], np.prod(s[1:]))).T
    if mesh:
        res = coord_mesh, coord_list
    else:
        res = coord_list
    return res

df = pd.read_csv('input_output.csv')
df_clean = df.dropna()

WThk = df_clean['WThk'].values[:, None]
FL = df_clean['FL'].values[:, None]
EPS_max = df_clean['EPS_max'].values[:, None]

x_train = df_clean[['WThk', 'FL']].values

scaler_X = StandardScaler()
scaler_X.fit(x_train)
joblib.dump(scaler_X, 'scaler_X.gz')

scaler_Y = StandardScaler()
scaler_Y.fit(EPS_max)
joblib.dump(scaler_Y, 'scaler_Y.gz')

X_scaled = scaler_X.transform(x_train)
Y_scaled = scaler_Y.transform(EPS_max)

fig = plt.figure(num='Scatter plot')
ax = fig.add_subplot(projection='3d')
ax.scatter(WThk, FL, EPS_max)

gpy_model = GPy.models.GPRegression(X_scaled, Y_scaled, GPy.kern.RBF(2))

# gpy_model.likelihood.constrain_bounded(1e-10, 1)
# gpy_model.likelihood.fix(1e-10)
# gpy_model.optimize_restarts(
#     num_restarts=5,
#     verbose=False,
# )

gpy_model.optimize()
np.save('gpy_model.npy', gpy_model.param_array)

m_load = GPy.models.GPRegression(X_scaled, Y_scaled)
m_load.update_model(False) # do not call the underlying expensive algebra on load
m_load.initialize_parameter() # Initialize the parameters (connect the parameters up)
m_load[:] = np.load('gpy_model.npy', allow_pickle=True) # Load the parameters
m_load.update_model(True) # Call the algebra only once

emukit_model = GPyModelWrapper(m_load)

test_x_mesh, test_x_list = uniform_grid([0.1, 0.5], [0.49, 1.2], [50, 50], mesh=True)
test_x_list_scaled = scaler_X.transform(test_x_list)

mu_plot_scaled, var_plot_scaled = emukit_model.predict(test_x_list_scaled)
mu_plot = scaler_Y.inverse_transform(mu_plot_scaled)
ucb_plot = scaler_Y.inverse_transform(mu_plot_scaled + 2 * np.sqrt(np.abs(var_plot_scaled)))
lcb_plot = scaler_Y.inverse_transform(mu_plot_scaled - 2 * np.sqrt(np.abs(var_plot_scaled)))

ax.plot_surface(test_x_mesh[0], test_x_mesh[1], mu_plot.reshape(test_x_mesh[0].shape), alpha=.5, cmap='viridis')
ax.plot_surface(test_x_mesh[0], test_x_mesh[1], ucb_plot.reshape(test_x_mesh[0].shape), alpha=.1, color='k')
ax.plot_surface(test_x_mesh[0], test_x_mesh[1], lcb_plot.reshape(test_x_mesh[0].shape), alpha=.1, color='k')

plt.figure(num='Contour plot')
plt.contourf(test_x_mesh[0], test_x_mesh[1], mu_plot.reshape(test_x_mesh[0].shape), levels=100)

plt.figure(num='cb contour plot')
plt.contourf(test_x_mesh[0], test_x_mesh[1], (ucb_plot - mu_plot).reshape(test_x_mesh[0].shape), levels=100)

plt.show()