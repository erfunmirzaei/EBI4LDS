import numpy as np
from numpy.typing import NDArray

def sample(num_points:int, num_trajectories: int = 1, seed:int = 0 ) -> NDArray[np.float64]:
    rng = np.random.default_rng(seed)
    drift = np.exp(-1)
    diffusion = np.sqrt(1 - np.exp(-2))

    noise = rng.normal(scale = diffusion, size = (num_points - 1, num_trajectories))

    X = np.zeros((num_points, num_trajectories), dtype = np.float64)
    X[0] = rng.normal(size=num_trajectories)
    for i in range(1, num_points):
        X[i] = drift*X[i-1] + noise[i-1]
    # if num_trajectories == 1:
    return X.astype('float16')
    # else:
    #     return X[:, None, :]

def make_dataset(configs):
    """
    Take samples from the OU process
    Args:
        configs: object, configurations for the experiment
    Returns:
        data_points: np.array, shape (n, n_repits), samples from the OU process
    """
    data_points = data_points = sample(configs.n_train_points + configs.n_val_points + configs.n_test_points, num_trajectories= configs.n_repits)
    return data_points
