import os
import json
import pickle
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import ParameterGrid
from sklearn.gaussian_process.kernels import RBF
from kooplearn.models import Kernel
from kooplearn.data import traj_to_contexts
from risk_bound import risk_bound_Ala

from data_pipeline import Trajectory, ala2_dataset, lagged_sampler


def load_data(current_file_path:os.PathLike, split:str = 'train') -> Trajectory:
    """
    Load the Ala2 dataset
    Args:
        current_file_path: os.PathLike, path to the current file
        split: str, split of the dataset
    Returns:
        data: Trajectory, Ala2 dataset
    """
    path = os.path.join(current_file_path, 'data/')
    return ala2_dataset(path, split = split) 

def hp_fit(configs:dict, current_file_path: os.PathLike):
    """
    Hyperparameter optimization for the Ala2 dataset
    Args:
        configs: dict, configurations for the experiment
        current_file_path: os.PathLike, path to the current file
    """
    #Check if 'results' path exists
    results_path = os.path.join(current_file_path, 'results')
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    
    #Loading the distances data
    train_dist = load_data(current_file_path, split = 'train').dist
    train_pos = load_data(current_file_path, split = 'train').pos
    val_dist = load_data(current_file_path, split = 'validation').dist
    test_dist = load_data(current_file_path, split = 'test').dist
    test_pos = load_data(current_file_path, split = 'test').pos

    n_snapshots = train_dist.shape[0]
    lagtime = configs['data']['lag']
    shuffle = configs['data']['shuffle']
    num_steps = configs['data']['n_forecast']
    num_points = configs['data']['n_train']

    train_data = traj_to_contexts(train_dist[:configs['data']['n_train']+lagtime,:], time_lag=lagtime)
    val_data = traj_to_contexts(test_dist[:configs['data']['n_val']+lagtime,:], time_lag=lagtime)
    test_data = traj_to_contexts(test_dist[:configs['data']['n_test']+lagtime,:], time_lag=lagtime)

    train_pos_data = traj_to_contexts(train_pos[:configs['data']['n_train']+lagtime,:], time_lag=lagtime)
    val_pos_data = traj_to_contexts(test_pos[:configs['data']['n_val']+lagtime,:], time_lag=lagtime)
    test_pos_data = traj_to_contexts(test_pos[:configs['data']['n_test']+lagtime*num_steps,:], context_window_len= num_steps +1, time_lag=lagtime)

    train_data.observables = {'pos': train_pos_data.data}

    estimator_args = configs['estimator']

    # Define the parameter grid
    tikhonov_regs = np.geomspace(1e-8, 1e-4, 4)
    length_scales = np.geomspace(1e-2, 1e0, 4)
    params = list(
        ParameterGrid(
            {
                'tikhonov_reg': tikhonov_regs,
                'length_scale': length_scales,
            }
        )
    )


    val_scores = np.empty((len(params), 1))
    rmse = np.empty((len(params), num_steps))
    for iter_idx, iterate in tqdm(enumerate(params), total=len(params)):
        # try :
        model = Kernel(RBF(length_scale= iterate['length_scale']), reduced_rank=True, rank = estimator_args['rank'], tikhonov_reg = iterate['tikhonov_reg']).fit(train_data)
        risk_bound = risk_bound_Ala(model, n = train_data.shape[0], r = estimator_args['rank'], delta = 0.05)
        val_scores[iter_idx] = risk_bound
        # except:
        #     val_scores[iter_idx] = np.inf

        test_data.observables = {'pos': train_pos_data.data}
        pos_pred = np.array([model.predict(test_data, t= t, predict_observables= True)['pos'] for t in range(1, num_steps + 1)])
        
        pos_test = test_pos_data.lookforward(1)
        pos_pred = pos_pred[:,:,1,:]
        pos_pred = pos_pred.swapaxes(0,1)
        # print(pos_pred.shape, pos_test.shape)
        dY = (pos_pred - pos_test)**2
        dY = np.mean(dY, axis = (0, 2)) #Averaging on observations and features
        rmse[iter_idx] = np.sqrt(dY)

    result = {
        'avg_bias': val_scores,
        'forecast_rmse': rmse,
        'kernels': params,
        'configs': configs
    }
    #Check if 'results' path exists
    
    #Pickling the results
    with open(os.path.join(results_path, 'results.pkl'), 'wb') as f:
        pickle.dump(result, f)

if __name__ == "__main__":
    current_file_path = os.path.dirname(__file__)
    #Loading HP configs
    cfg_path = os.path.join(current_file_path, 'configs/hp.json')
    with open(cfg_path, "r") as cfg_file:
        configs = json.load(cfg_file)
    hp_fit(configs, current_file_path)
