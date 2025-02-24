import numpy as np
from corr_est_cov_est import biased_covariance_estimator, unbiased_covariance_estimator

def risk_bound_Ala(kernel_model, n, r, delta):
    
    """
    Compute the risk bounds for different values of N
    Args:
        kernel_model: object, kernel model
        n: int, number of data points
        r: float, noise level
        delta: float, confidence level
    Returns:
        risk_bound: float, risk bound
    """
    c_h = 1
    K = kernel_model.kernel_X / n
    L = kernel_model.kernel_Y / n
    U = kernel_model.U
    V = kernel_model.V
    norm_est = np.trace(U.T @ K @ U @ V.T @ L @ V)
    for tau in range(1,n):
        # TODO: This is for this process only. You can change this to any other process
        if (n / tau) % 2 == 0: 
            if delta/(2*norm_est) >= 2*(n/(2*tau) - 1)*(0.97**tau):
                min_tau = tau
                break
    
    tau = min_tau 
    beta_coeff = 0.97**tau
    beta_coeff_prime = 0.97**(tau-1)
    m = n / (2*tau)
    l_tau = np.log(12/((delta/(2*norm_est))- 2*(m-1)*beta_coeff))
    L_tau = np.log(12/((delta/(2*norm_est)) - 2*(m-1)*beta_coeff_prime))


    # Compute the kernel matrix and the biased and unbiased covariance estimator
    kernel_matrix = kernel_model.kernel_X
    biased_cov_est = biased_covariance_estimator(kernel_matrix, tau)
    unbiased_cov_est = unbiased_covariance_estimator(kernel_matrix, tau)

    T_hat = kernel_model.kernel_YX
    biased_cross_cov_est = biased_covariance_estimator(T_hat, tau)
    unbiased_cross_cov_est = unbiased_covariance_estimator(T_hat, tau)
    # Compute the Variance trace(D- D_hat) 
    diag_elss = np.diagonal(kernel_matrix)
    diag_elss = diag_elss.reshape(int(2*m), tau)
    Ys = diag_elss.mean(axis = 1)
    V_D = 2*np.var(Ys, ddof=1)

    First_term = ((128*(norm_est**2)*c_h*tau)/(3*n))*l_tau 
    Second_term = ((128*np.sqrt(r)*norm_est*c_h*tau)/(3*n))*L_tau 
    Third_term = ((7*c_h)/(3*(m-1)))*l_tau
    Fourth_term = np.sqrt(((2*l_tau + 1)*32*(norm_est**4)*tau*biased_cov_est)/n)
    Fifth_term = np.sqrt(((2*L_tau + 1)*8*r*(norm_est**2)*tau*biased_cross_cov_est)/n)
    Sixth_term = np.sqrt((2*V_D*tau*l_tau)/n)
    risk_bound = First_term + Second_term + Third_term + Fourth_term + Fifth_term + Sixth_term 

    return risk_bound