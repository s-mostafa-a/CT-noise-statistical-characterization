import math


def non_central_gamma_pdf(x, alpha, beta, delta=-1024):
    assert alpha > 0 and beta > 0
    assert x >= delta
    y = x - delta
    form = math.pow(y, (alpha - 1)) * math.exp(-y / beta)
    denominator = math.pow(beta, alpha) * math.gamma(alpha)
    return form / denominator


def central_gamma_pdf(y, alpha, beta):
    assert alpha > 0 and beta > 0
    form = math.pow(y, (alpha - 1)) * math.exp(-y / beta)
    denominator = math.pow(beta, alpha) * math.gamma(alpha)
    return form / denominator
