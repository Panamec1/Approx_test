import itertools
import math
import time
from decimal import *

import numpy as np
from scipy import special
from scipy.stats import norm
from scipy.special import gamma


def kern(t, s, mu):
    ch = np.minimum(t, s) - (s * t)
    zn = (np.sqrt(t * (1 - t))) * (np.sqrt(s * (1 - s)))
    return np.prod(np.power(ch / zn, mu))


def direct(indSize=2, mu=np.array([0.1, 0.1])):
    # print(indSize ** (2 * len(mu)))
    k = np.empty([indSize ** (len(mu)), indSize ** (len(mu))])
    start_time = time.time()

    ind = np.array(range(0, indSize)) / indSize
    ind[0] = 0.0001
    ind[len(ind) - 1] = 0.9999

    r = (len(ind)) ** len(mu)
    indx = np.zeros(r)
    indexes = np.array(list(itertools.product(ind, repeat=len(mu))))
    k = np.empty([indSize ** (len(mu)), indSize ** (len(mu))])
    f = lambda i, j: kern(indexes[i], indexes[j], mu)
    t = np.fromfunction(np.vectorize(f), (indSize ** (len(mu)),
                                          indSize ** (len(mu))), dtype=int)
    t = np.array(t).reshape(indSize ** (len(mu)), indSize ** (len(mu)))
    evalues, evectors = np.linalg.eigh(t)
    sqrt_matrix = evectors * np.sqrt(np.diag(evalues)) @ np.linalg.inv(evectors)
    vect = np.random.normal(0, 1, indSize ** (len(mu)))
    r = sqrt_matrix.dot(vect)
    print("--- %s seconds ---" % str(time.time() - start_time))
    return r


def evNumbers(mu, j):
    # mu - вектор mu_k
    # j  - параметр
    return np.prod(mu / ((mu + j) * (mu + j - 1)))


def evLeg(n, x):
    sr = Decimal(0)
    for k in range(n):
        arg1 = Decimal((gamma(n + 1) * gamma(k + 1)) / (gamma(n - k + 1)))
        arg2 = Decimal((gamma(n + k + 1) * gamma(k + 1)) / (gamma(n + 1)))
        arg3 = Decimal(((x - 1) / 2) ** k)
        r = arg2 * arg3 * arg1
        if r < ((0.1) ** 5):
            break
        sr = sr + r
    return sr


def evFunc(mu, j, t):
    # mu - вектор mu_k
    # j  - параметр
    # t  - вектор нормальных распределений
    st = Decimal(1)
    for i in range(len(mu)):
        st = st * Decimal(phi(mu[i], j, t[i]))
    return st


def spLegend(mu, k, x):
    h = (Decimal((-1) ** k)) / (Decimal((2 ** (mu + k - 1))) * Decimal((gamma(mu + k))))
    h1 = h * Decimal(1 / ((1 - (x ** 2)) ** (mu / 2)))
    pol = special.eval_legendre(k, x)
    return h1 * Decimal(pol)


def phi(mu, j, t):
    ch = Decimal(2 * mu + 2 * j - 1)
    zn = 1
    mult1 = Decimal(math.sqrt(ch / zn))
    mult2 = spLegend(mu, j, 2 * t - 1)
    return mult1 * mult2


def approximation(mu, t):
    n = int(math.e ** (np.sum(mu) + 2 * num + norm.ppf(1 - eps * eps) * math.sqrt(-np.sum(np.log(mu)) - 4 * num)))
    approx_value = Decimal(0)
    for i in range(1, n + 1):
        slog = (np.random.normal(0, 1, 1))
        slog = Decimal(slog[0]) * Decimal((math.sqrt(evNumbers(mu, i))))
        slog = slog * Decimal(evFunc(mu, i, t))

        if abs(slog) < 0.1 ** 30:
            break

        approx_value = approx_value + slog
    return approx_value


num = 7
indSize = 7
eps = 0.01

t = [0.1, 0.2, 0.3, 0.1, 0.2,
     0.3, 0.1, 0.2, 0.1]
mu = np.array([0.01] * num)

st = time.time()
for i in range(indSize ** (len(mu))):
    print(i, indSize ** (len(mu)), time.time() - st)
    approximation(mu, t)
print(time.time() - st, "--- Approx %s seconds ---")

direct(indSize, mu=mu)
# 5 10 1551.643
