"""GELU variants and derivatives, in float64."""
import math

import numpy as np

SQRT2PI = math.sqrt(2 * math.pi)
KAPPA = 2 * math.sqrt(2 / math.pi)   # tanh GELU written as x / (1 + exp(-KAPPA (x + C x^3)))
C = 0.044715

_erf = np.frompyfunc(math.erf, 1, 1)
_erfc = np.frompyfunc(math.erfc, 1, 1)


def Phi(x):
    """Standard normal CDF, accurate in both tails (erfc avoids cancellation)."""
    x = np.asarray(x, dtype=np.float64)
    return (0.5 * _erfc(-x / math.sqrt(2))).astype(np.float64)


def phi(x):
    return np.exp(-np.asarray(x, dtype=np.float64) ** 2 / 2) / SQRT2PI


def gelu(x):
    """Exact GELU, g(x) = x * Phi(x)."""
    x = np.asarray(x, dtype=np.float64)
    return x * Phi(x)


def dgelu(x):
    """g'(x) = Phi(x) + x phi(x)."""
    return Phi(x) + np.asarray(x, dtype=np.float64) * phi(x)


def d4gelu(x):
    """g''''(x) = (-x^4 + 7x^2 - 4) phi(x); its sup over the reals is 4 / sqrt(2 pi)."""
    x = np.asarray(x, dtype=np.float64)
    return (-x**4 + 7 * x**2 - 4) * phi(x)


def gelu_tanh(x):
    """PyTorch GELU(approximate='tanh'), in the equivalent sigmoid form."""
    x = np.asarray(x, dtype=np.float64)
    return x / (1 + np.exp(-KAPPA * (x + C * x**3)))
