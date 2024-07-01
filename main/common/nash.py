""" Solve equilibrium (Nash/Correlated) with Linear Programming in zero-sum games. Adapted from https://github.com/quantumiracle/nash-dqn/blob/master/equilibrium_solver/eq_ECOSsolver.py. """
import ecos
import torch
import numpy as np
from scipy.sparse import csr_matrix
import time
from multiprocessing.pool import ThreadPool


def NashEquilibriumECOSSolver(M):
    """
    https://github.com/embotech/ecos-python
    min  c*x
    s.t. A*x = b
         G*x <= h
    https://github.com/embotech/ecos/wiki/Usage-from-MATLAB
    args: 
        c,b,h: numpy.array
        A, G: Scipy sparse matrix
    """
    row, col = M.shape
    c = np.zeros(row+1)
    # max z
    c[-1] = -1  
    
    # x1+x2+...+xn=1
    A = np.ones(row+1)
    A[-1] = 0.
    A = csr_matrix([A])
    b=np.array([1.])
    
    # M.T*x<=z
    G1 = np.ones((col, row+1))
    G1[:col, :row] = -1. * M.T
    # x>=0
    G2 = np.zeros((row, row+1))
    for i in range(row):
        G2[i, i]=-1. 
    # x<=1.
    G3 = np.zeros((row, row+1))
    for i in range(row):
        G3[i, i]=1. 
    G = csr_matrix(np.concatenate((G1, G2, G3)))
    h = np.concatenate((np.zeros(col), np.zeros(row), np.ones(row)))
    
    # specify number of variables
    dims={'l': col+2*row, 'q': []}
                       
    solution = ecos.solve(c,G,h,dims,A,b, verbose=False)

    p1_dist = solution['x'][:row]
    p2_dist = solution['z'][:col] # z is the dual variable of x
    # There are at least two bad cases with above constrained optimization,
    # where the constraints are not fully satisfied (some numerical issue):
    # 1. the sum of vars is larger than 1.
    # 2. the value of var may be negative.
    abs_p1_dist = np.abs(p1_dist)
    abs_p2_dist = np.abs(p2_dist)
    normalized_p1_dist = abs_p1_dist/np.sum(abs_p1_dist)
    normalized_p2_dist = abs_p2_dist/np.sum(abs_p2_dist)

    nash_value = normalized_p1_dist@M@normalized_p2_dist.T

    return (normalized_p1_dist, normalized_p2_dist), nash_value


def NashEquilibriumECOSParallelSolver(Ms):
    # this is found to be not faster than iterate over non-parallel version
    # ref: https://github.com/embotech/ecos-python/pull/20
    # t0 = time.time()
    pool = ThreadPool(2)
    results = pool.map(NashEquilibriumECOSSolver, Ms)
    # t1 = time.time()
    policies = []
    values = []
    for re in results:
        policies.append(re[0])
        values.append(re[1])
    # t2 = time.time()
    # print('compute time: ', t1-t0, t2-t1)
    return policies, values


def compute_nash(q_values, update=False):
    action_dim = np.sqrt(q_values.shape[-1]).astype(int)
    q_tables = q_values.reshape(-1, action_dim, action_dim)
    if isinstance(q_values, torch.Tensor):
        return_torch = True
        q_tables = q_tables.cpu().numpy()
    else:
        return_torch = False
    all_actions = []
    all_dists = []
    all_ne_values = []

    for q_table in q_tables:
        dist, value = NashEquilibriumECOSSolver(q_table)
        all_dists.append(dist)
        all_ne_values.append(value)

    if update:
        return all_dists, torch.tensor(all_ne_values, dtype=q_values.dtype, device=q_values.device) if return_torch else all_ne_values
    else:
        # Sample actions from Nash strategies
        for ne in all_dists:
            actions = []
            for dist in ne:  # iterate over agents
                try:
                    sample_hist = np.random.multinomial(1, dist)  # return one-hot vectors as sample from multinomial
                except:
                    print('Not a valid distribution from Nash equilibrium solution: ', dist)
                a = np.where(sample_hist>0)
                actions.append(a)
            all_actions.append(np.array(actions).reshape(-1))

        return torch.tensor(np.array(all_actions), dtype=torch.int, device=q_values.device) if return_torch else np.array(all_actions), all_dists, torch.tensor(all_ne_values, dtype=q_values.dtype, device=q_values.device) if return_torch else all_ne_values


if __name__ == "__main__":
    A_list = [
        np.array([[0, -1, 1], 
            [1, 0, -1], 
            [-1, 1, 0]]),
        np.array([[ 0.001,  0.001,  0.00,     0.00,     0.005,  0.01, ],
            [ 0.033,  0.166,  0.086,  0.002, -0.109,  0.3,  ],
            [ 0.001,  0.003,  0.023,  0.019, -0.061, -0.131,],
            [-0.156, -0.039,  0.051,  0.016, -0.028, -0.287,],
            [ 0.007,  0.029,  0.004,  0.005,  0.003, -0.012],
            [ 0.014,  0.018, -0.001,  0.008, -0.009,  0.007]]),
        np.array([[ 0.08333333,  2.08333333 ,-0.91666667],
            [-0.91666667,  0.08333333,  1.08333333],
            [ 1.08333333, -0.91666667,  0.08333333]]),
        np.array([[0, 2, -1],
            [-1, 0, 1],
            [ 1, -1, 0],
            [0, 2, -1]]),
        np.array([[1, 1, 1], 
            [0, 0, 0], 
            [0, 0, 0],
            [0, 0, 0]]),
    ]
    # A = A + 1
    for A in A_list:
        t0 = time.time()
        ne, ne_v = NashEquilibriumECOSSolver(A)
        t1 = time.time()
        print(t1 - t0)
        print(ne)
        print(ne_v)

        ne_, ne_v_ = NashEquilibriumECOSSolver(-A.T)
        print(ne_)
        print(ne_v_)
        assert np.isclose(ne[0], ne_[1], atol=1e-5).all() and np.isclose(ne[1], ne_[0], atol=1e-5).all(), 'error for transpose'

        As = 3 * [A]
        nes, ne_vs = NashEquilibriumECOSParallelSolver(As)
        # print(nes)
        # print(ne_vs)
        assert np.isclose(ne[0], nes[0][0]).all() and np.isclose(ne[1], nes[0][1]).all(), 'error when A is not zero-sum'

    ## comparison test
    # n = 100   # number of matrices
    # size = 10  # matrix dimension
    # A = np.random.random((n, size, size))

    # t0 = time.time()
    # for a in A:
    #     ne, ne_v = NashEquilibriumECOSSolver(a)
    # t1 = time.time()
    # print('t1', t1 - t0)

    # print('-'*100)
    # t0 = time.time()
    # nes, ne_vs = NashEquilibriumECOSParallelSolver(A)
    # t1 = time.time()
    # print('t2', t1 - t0)