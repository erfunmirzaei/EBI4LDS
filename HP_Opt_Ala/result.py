import os
import json
import pickle
import importlib
import numpy as np
from tqdm import tqdm
from einops import einsum
from sklearn.model_selection import ParameterGrid
from sklearn.gaussian_process.kernels import RBF
from kooplearn.models import Linear, Nonlinear, Kernel
from kooplearn.data import traj_to_contexts
from risk_bound import risk_bound_Ala

from io_typing import Trajectory
from io_data import ala2_dataset, lagged_sampler

import matplotlib.pyplot as plt

current_file_path = os.path.dirname(__file__)
results_path = os.path.join(current_file_path, 'results/results.pkl')
results = pickle.load(open(results_path, 'rb'))
best_estimator_id = np.argmin(results['avg_bias'])
rmse = results['forecast_rmse']

time_horizon = int(rmse.shape[1]*0.6)
fig, ax = plt.subplots(figsize = (4, 3))
ax.plot(np.arange(1, time_horizon + 1)*0.05, rmse[:,:time_horizon].T, 'k--', lw=0.75, alpha=0.75)
ax.plot(np.arange(1, time_horizon + 1)*0.05, rmse[best_estimator_id, :time_horizon], 'r-', lw=1.5, label='Minimizer of Emp. Risk Bound Est.')
ax.set_xlabel(r'Forecast horizon [ns]')
ax.set_yscale('log')
ax.set_ylabel('RMSE')
ax.set_ylim(3e-2, 0.3e1)
ax.set_title('Model selection for Alanine Dipeptide')
ax.margins(x=0)
plt.legend(loc = 'upper left', frameon=False)
plt_path = os.path.join(current_file_path, 'results/ala_model_selection.pdf')
plt.savefig(plt_path, bbox_inches="tight", dpi = 600)
plt.show()
