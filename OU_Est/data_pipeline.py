from src import OU_process

def make_dataset(configs):
    """
    Take samples from the OU process
    Args:
        configs: object, configurations for the experiment
    Returns:
        data_points: np.array, shape (n, n_repits), samples from the OU process
    """
    data_points = data_points = OU_process.sample(configs.n_train_points + configs.n_val_points + configs.n_test_points, num_trajectories= configs.n_repits)
    return data_points
