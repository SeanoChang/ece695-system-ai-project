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


def kernel_fp32(x32, erf_shift=0, fma=False):
    """FP32 model of the production kernel y = 0.5 x (1 + erf(x / sqrt 2)), as PyTorch and Triton compute it.

    erf is correctly rounded to FP32 and then moved by erf_shift ulps, to model a library that is not
    correctly rounded. With fma, the product and the sum are one fused operation (emulated in float64).
    """
    x32 = np.asarray(x32, dtype=np.float32)
    z = (x32 * np.float32(1 / math.sqrt(2))).astype(np.float32)
    e = _erf(z.astype(np.float64)).astype(np.float64).astype(np.float32)
    for _ in range(abs(erf_shift)):
        e = np.clip(np.nextafter(e, np.float32(math.copysign(2, erf_shift))), -1, 1).astype(np.float32)
    a = (np.float32(0.5) * x32).astype(np.float32)
    if fma:
        return (a.astype(np.float64) * e.astype(np.float64) + a.astype(np.float64)).astype(np.float32)
    return (a * (np.float32(1) + e).astype(np.float32)).astype(np.float32)
