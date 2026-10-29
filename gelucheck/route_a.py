"""Route A: the batch identity for tanh GELU.

With y = x / (1 + exp(-k z)), z = x + c x^3, exact arithmetic gives for every nonzero x
    x - y = y exp(-k z),  so over a block  prod (x - y) = exp(-k sum z) prod y.
A real checker compares the two sides in log form (products of 256 values overflow):
    e_i = log|x_i - y_i| - (-k z_i + log|y_i|),   block passes when |sum e_i| <= tol.
With 16-bit outputs, rounding of correct y values makes e_i nonzero; this module measures how much.
"""
import numpy as np

from .gelu import C, KAPPA


def mismatch(x, y):
    """Per-element log mismatch e_i; non-finite where x - y or y is 0, or y is inf/nan."""
    with np.errstate(all="ignore"):   # nan/inf outputs (e.g. after a bit flip) give non-finite terms
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        return np.log(np.abs(x - y)) - (-KAPPA * (x + C * x**3) + np.log(np.abs(y)))


def block_stat(e):
    """|sum e| per block (last axis) and whether every term was finite."""
    finite = np.isfinite(e)
    return np.abs(np.where(finite, e, 0).sum(-1)), finite.all(-1)


def fails(e, tol):
    s, finite = block_stat(e)
    return (~finite) | (s > tol)


def rounding_bound(x, y16):
    """Per-block tolerance from FP16 rounding: sum of half an ulp of y, propagated through e_i."""
    x = np.asarray(x, dtype=np.float64)
    with np.errstate(all="ignore"):   # flipped exponent bits can produce inf/nan; those blocks fail anyway
        yf = y16.astype(np.float64)
        ulp = np.spacing(np.abs(y16)).astype(np.float64)
        b = np.abs(x) * ulp / 2 / np.abs(yf * (x - yf))
    return np.where(np.isfinite(b), b, 0).sum(-1)


def flip_one_bit(y16, rng):
    """Flip one uniformly random bit of one random element per block (last axis). Returns copy, bit index."""
    out = y16.copy()
    nb, b = out.shape
    idx = rng.integers(0, b, nb)
    bit = rng.integers(0, 16, nb).astype(np.uint16)
    out.view(np.uint16)[np.arange(nb), idx] ^= (np.uint16(1) << bit)
    return out, bit
