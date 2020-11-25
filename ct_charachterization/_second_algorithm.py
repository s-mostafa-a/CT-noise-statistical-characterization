import numpy as np

from ._first_algorithm_using_log import run_first_log_algorithm


def run_second_algorithm(y: np.array, mu: np.array, neighborhood_size: int, delta=-1030, max_iter=5, tol=0.01,
                         non_central=False):
    global_theta, global_gamma = run_first_log_algorithm(y=y, mu=mu, neighborhood_size=0, delta=delta,
                                                         max_iter=max_iter,
                                                         tol=tol, non_central=non_central)
    initial_alpha = list(global_theta[1, ...])
    local_theta, local_gamma = run_first_log_algorithm(y=y, mu=mu, neighborhood_size=neighborhood_size, delta=delta,
                                                       max_iter=max_iter, tol=tol, non_central=non_central,
                                                       initial_alpha=initial_alpha)
    return local_theta, local_gamma
