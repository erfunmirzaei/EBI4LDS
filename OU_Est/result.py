import importlib
import subprocess
import sys
for module in ['kooplearn', 'datasets', 'matplotlib', 'ml-confs']: # !! Add here any additional module that you need to install on top of kooplearn
    try:
        importlib.import_module(module)
    except ImportError:
        if module == 'kooplearn':
            module = 'kooplearn[full]'
        # pip install -q {module}
        subprocess.check_call([sys.executable, "-m", "pip", "install", module])

from pathlib import Path
import ml_confs
import numpy as np
from utils import plot_OU_tau, plot_OU_N

# Load configs
main_path = Path(__file__).parent
data_path = main_path / "__data__"
configs = ml_confs.from_file(main_path / "configs.yaml")

# First Experiment: Plot the bounds for different values of tau
Pinelis_bound = []
Pinelis_emp_bound_biased_cov_est = []
Pinelis_emp_bound_unbiased_cov_est = []
M_bound = []
M_emp_bound_biased_cov_est = []
M_emp_bound_unbiased_cov_est = []
taus = []

Pinelis_bound.append(np.load(str(main_path) + f'/results/Pinelis_bound_n_{configs.n_plot_tau}_delta_{configs.delta}.npy'))
Pinelis_emp_bound_biased_cov_est.append(np.load(str(main_path) + f'/results/Pinelis_emp_bound_biased_cov_est_n_{configs.n_plot_tau}_delta_{configs.delta}.npy'))
Pinelis_emp_bound_unbiased_cov_est.append(np.load(str(main_path) + f'/results/Pinelis_emp_bound_unbiased_cov_est_n_{configs.n_plot_tau}_delta_{configs.delta}.npy'))
M_bound.append(np.load(str(main_path) + f'/results/M_bound_n_{configs.n_plot_tau}_delta_{configs.delta}.npy'))
M_emp_bound_biased_cov_est.append(np.load(str(main_path) + f'/results/M_emp_bound_biased_cov_est_n_{configs.n_plot_tau}_delta_{configs.delta}.npy'))
M_emp_bound_unbiased_cov_est.append(np.load(str(main_path) + f'/results/M_emp_bound_unbiased_cov_est_n_{configs.n_plot_tau}_delta_{configs.delta}.npy'))
taus.append(np.load(str(main_path) + f'/results/taus_n_{configs.n_plot_tau}_delta_{configs.delta}.npy'))

# Plot the bounds for different values of tau
print('Plotting bounds for different values of tau')
labels = ["Pinelis bound", "New Pessimistic bound", "EBI(biased)", "EBI(unbiased)", "Estimated True value"]
plot_OU_tau(configs, Pinelis_bound, Pinelis_emp_bound_biased_cov_est, Pinelis_emp_bound_unbiased_cov_est, M_bound, M_emp_bound_biased_cov_est, M_emp_bound_unbiased_cov_est, taus, labels)


# # Second Experiment: Plot the bounds for different values of N
# Ns = np.arange(configs.n_train_first, configs.n_sample_est_tr, configs.n_train_step) 
Ns = [500,1000,2000, 5000, 10000, 20000,40000]

Pinelis_bound = []
Pinelis_emp_bound_biased_cov_est = []
Pinelis_emp_bound_unbiased_cov_est = []
M_bound = []
M_emp_bound_biased_cov_est = []
M_emp_bound_unbiased_cov_est = []
True_value = []

Pinelis_bound.append(np.load(str(main_path) + f'/results/Pinelis_bound_delta_{configs.delta}.npy'))
Pinelis_emp_bound_biased_cov_est.append(np.load(str(main_path) + f'/results/Pinelis_emp_bound_biased_cov_est_delta_{configs.delta}.npy'))
Pinelis_emp_bound_unbiased_cov_est.append(np.load(str(main_path) + f'/results/Pinelis_emp_bound_unbiased_cov_est_delta_{configs.delta}.npy'))
M_bound.append(np.load(str(main_path) + f'/results/M_bound_delta_{configs.delta}.npy'))
M_emp_bound_biased_cov_est.append(np.load(str(main_path) + f'/results/M_emp_bound_biased_cov_est_delta_{configs.delta}.npy'))
M_emp_bound_unbiased_cov_est.append(np.load(str(main_path) + f'/results/M_emp_bound_unbiased_cov_est_delta_{configs.delta}.npy'))
True_value.append(np.load(str(main_path) + f'/results/True_value_delta_{configs.delta}.npy'))

# Plot the bounds for different values of N
print('Plotting bounds for different values of N')
labels = ["Pinelis bound", "New Pessimistic bound", "EBI(biased)", "EBI(unbiased)", "Estimated True value"]
plot_OU_N(configs, Pinelis_bound, Pinelis_emp_bound_biased_cov_est, Pinelis_emp_bound_unbiased_cov_est, M_bound, M_emp_bound_biased_cov_est, M_emp_bound_unbiased_cov_est, True_value, Ns, [1], labels)
          