# An Empirical Bernstein Inequality for Dependent Data in Hilbert Spaces and Applications

Welcome to the official repository for the experiments presented in the paper ["An Empirical Bernstein Inequality for Dependent Data in Hilbert Spaces and Applications"](). This paper introduces novel empirical Bernstein inequalities for dependent random variables in Hilbert spaces and demonstrates their applications in various scenarios. For any questions or collaboration suggestions, please reach out to [erfunmirzaei@gmail.com](mailto:erfunmirzaei@gmail.com).

The paper includes three numerical experiments: Covariance Estimation Using Samples from Ornstein–Uhlenbeck Process, Noisy Ordered MNIST, and EBI-based Model Selection. The code for each experiment is organized into separate folders. To reproduce the results, follow these general steps: run `simulation.py`, then run `result.py` to analyze the results in the corresponding folder.

## Covariance Estimation Using Samples from Ornstein–Uhlenbeck Process

The `OU_Est` folder contains the code for this experiment.

- `cov_estimation.py`: Functions for calculating bounds for covariance estimation
- `corr_est_cov_est.py`: Functions for estimating the proxies of covariance introduced in the paper
- `configs.yaml`: Hyperparameter selection
- `utils.py`: Plot functions and divisors
- `data_pipeline.py`: Functions for generating samples from the OU process
- `simulation.py`: Running simulations and saving results
- `result.py`: Plotting the final results

## Noisy Ordered MNIST

The `Noisy_Ordered_MNIST` folder contains the code for this experiment.

- `oracle.py`: Oracle CNN for encoding MNIST images
- `transfer_op.py`: Fitting function for the transfer operator models
- `hparams_tuning.py`: Tuning the hyperparameters of the Gaussian kernel and Tikhonov regularization
- `normalized_corr_est_cov_est.py`: Functions for estimating the normalized proxies of covariance introduced in the paper
- `configs.yaml`: Hyperparameter selection
- `utils.py`: Plot functions
- `data_pipeline.py`: Functions for generating noisy and noise-free ordered MNIST images
- `simulation.py`: Running simulations and saving results
- `result.py`: Plotting the final results

## EBI-based Model Selection

The `EBI_Model_Selection` folder contains the code for this experiment.

- `get_dataset.py`: Functions for downloading the alanine dipeptide dataset
- `corr_est_cov_est.py`: Functions for estimating the proxies of covariance introduced in the paper
- `risk_bound.py`: Functions for computing the risk bound
- `configs.yaml`: Hyperparameter selection
- `utils.py`: Plot functions
- `data_pipeline.py`: Functions for data processing
- `simulation.py`: Running simulations and saving results
- `result.py`: Plotting the final results

## Installation

To set up the environment for running the experiments, install the required dependencies by running:

```sh
pip install -r requirements.txt
```

Alternatively, manually install the dependencies listed below:

```sh
pip install Flask==2.0.2
pip install requests==2.26.0
pip install pandas==1.3.3
pip install numpy==1.21.2
pip install scikit-learn==0.24.2
pip install matplotlib==3.4.3
pip install ml_confs== 0.0.1
```

## Running the Experiments

Each experiment is organized into its own folder. Follow the instructions below to reproduce the results for each experiment.

### Reproducing the Results

Navigate to the `OU_Est` or  `Noisy_Ordered_MNIST` or `EBI_Model_Selection` folder and run the following commands:

```sh
python simulation.py
python result.py
```

## Contact

For any questions or suggestions for collaboration, please reach out to [erfunmirzaei@gmail.com](mailto:erfunmirzaei@gmail.com).

# Contributing to EBI4LDS

Thank you for considering contributing to EBI4LDS! Here are some guidelines to help you get started:

## How to Contribute

1. **Fork the Repository**: Create a personal fork of the project on GitHub.
2. **Clone Your Fork**: Clone your forked repository to your local machine.
    ```sh
    git clone https://github.com/your-username/EBI4LDS.git
    ```
3. **Create a Branch**: Create a new branch for your changes.
    ```sh
    git checkout -b my-feature-branch
    ```
4. **Make Changes**: Make your changes to the codebase.
5. **Commit Changes**: Commit your changes with a clear and concise commit message.
    ```sh
    git commit -m "Add new feature"
    ```
6. **Push Changes**: Push your changes to your forked repository.
    ```sh
    git push origin my-feature-branch
    ```
7. **Create a Pull Request**: Open a pull request to the main repository with a description of your changes.

## Reporting Issues

If you find any bugs or have suggestions, please open an issue on GitHub.

Thank you for your contributions!
