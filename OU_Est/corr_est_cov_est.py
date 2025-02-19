import torch
import numpy as np

def make_block_matrices(matrix, tau):
    """ 
    Reshape a matrix into block matrices of size tau x tau.
    Args:
        matrix: numpy array of shape (n, n)
        tau: block size
    Returns:
        block_matrix_result: numpy array of shape (n_blocks, n_blocks, tau, tau)
    """
    n = matrix.shape[0]
    assert n % tau == 0, "Matrix dimensions must be divisible by tau"

    n_blocks = n // tau
    blocks = matrix.reshape(n_blocks, tau, n_blocks, tau)
    block_matrix_result = blocks.transpose(0, 2, 1, 3)
    
    return block_matrix_result

def biased_covariance_estimator(cov_mtx, tau):
    """
    Estimates the covariance from the squared kernel matrix using the biased estimator in the paper
    Args:
        cov_mtx: numpy array of shape (n, n)
        tau: block size
    Returns:
        sum: sum of the terms in biased estimator
    """
    n = cov_mtx.shape[0]
    block_cov_matrix = make_block_matrices(cov_mtx,tau) /np.sqrt(n*tau)
    diag_blocks = torch.diagonal(torch.from_numpy(block_cov_matrix), offset=0, dim1=0, dim2=1)
    return torch.sum(diag_blocks)

def unbiased_covariance_estimator(cov_mtx, tau):
    """
    Estimates the covariance from the squared kernel matrix using the unbiased estimator in the paper
    Args:
        cov_mtx: numpy array of shape (n, n)
        tau: block size 
    Returns:
        sum: sum of the terms in unbiased estimator
    """
    n = cov_mtx.shape[0]
    m = n / (2*tau)
    block_cov_matrix = make_block_matrices(cov_mtx,tau)/np.sqrt(n*tau)
    block_cov_matrix = torch.from_numpy(block_cov_matrix)

    diag_blocks = torch.diagonal(block_cov_matrix, offset=0, dim1=0, dim2=1)
    sum = torch.sum(diag_blocks)
    for signed in [1,-1]:
        for i in range(2,n, 2):
            diag_blocks = torch.diagonal(block_cov_matrix, offset=i*signed, dim1=0, dim2=1)
            if m > 1:
                sum -= torch.sum(diag_blocks) / (m-1)
    
    # print(plus_term, minus_term)
    if sum < 0:
        sum = 0.0

    return sum 

def sum_diagonals(cov_mtx, tau):
    """
    Sums the diagonals of the block matrices
    Args:
        cov_mtx: numpy array of shape (n, n)
        tau: block size
    Returns:
        sum: sum of the diagonals of the block matrices
    """
    n = cov_mtx.shape[0]
    m = n / (2*tau)

    block_cov_matrix = make_block_matrices(cov_mtx,tau) /tau
    block_cov_matrix = torch.from_numpy(block_cov_matrix)
    diag_blocks = torch.diagonal(block_cov_matrix, offset=0, dim1=0, dim2=1)
    sum = torch.sum(diag_blocks)
    for signed in [1,-1]:
        for i in range(2,n, 2):
            diag_blocks = torch.diagonal(block_cov_matrix, offset=i*signed, dim1=0, dim2=1)
            if m > 1:
                sum += torch.sum(diag_blocks) 
    return sum

def sum_off_diagonals(cov_mtx, tau):
    """
    Sums the off-diagonals of the block matrices
    Args:
        cov_mtx: numpy array of shape (n, n)
        tau: block size
    Returns:
        sum: sum of the off-diagonals of the block matrices
    """
    n = cov_mtx.shape[0]
    m = n / (2*tau)
    block_cov_matrix = make_block_matrices(cov_mtx,tau) /tau
    block_cov_matrix = torch.from_numpy(block_cov_matrix)
    sum = 0
    for signed in [1,-1]:
        for i in range(2,n, 2):
            diag_blocks = torch.diagonal(block_cov_matrix, offset=i*signed, dim1=0, dim2=1)
            if m > 1:
                sum += torch.sum(diag_blocks) 
    return sum