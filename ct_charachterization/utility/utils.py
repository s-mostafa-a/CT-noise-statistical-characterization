import math

import numpy as np


def non_central_gamma_pdf(x, alpha, beta, delta):
    assert x >= delta, f'''x must be more than or equal to delta. x: {x}, delta: {delta}'''
    y = x - delta
    return central_gamma_pdf(y=y, alpha=alpha, beta=beta)


def central_gamma_pdf(y, alpha, beta):
    assert alpha > 0 and beta > 0, f'''Alpha and Beta must be more than zero. Alpha: {alpha}, Beta: {beta}'''
    form = np.power(y, (alpha - 1)) * np.exp(-y / beta)
    denominator = np.power(beta, alpha) * math.gamma(alpha)
    return form / denominator


def broadcast_3d_tile(matrix, h, w, d):
    m, n, o = matrix.shape[0] * h, matrix.shape[1] * w, matrix.shape[2] * d
    return np.broadcast_to(matrix.reshape(matrix.shape[0], 1, matrix.shape[1], 1, matrix.shape[2], 1),
                           (matrix.shape[0], h, matrix.shape[1], w, matrix.shape[2], d)).reshape(m, n, o)


def broadcast_2d_tile(matrix, h, w):
    m, n = matrix.shape[0] * h, matrix.shape[1] * w
    return np.broadcast_to(matrix.reshape(matrix.shape[0], 1, matrix.shape[1], 1),
                           (matrix.shape[0], h, matrix.shape[1], w)).reshape(m, n)
