"""Route B: a piecewise-cubic table that certifies the exact 16-bit GELU output.

Offline: on each interval [a, a + h], a cubic Hermite polynomial matches g and g' at both ends.
Its error is at most M h^4 / 384 (M = max |g''''| on the interval). A per-interval bound Delta
adds FP32 coefficient storage, Horner rounding and input-offset rounding, and is stored rounded up.

Runtime: p = P(x). The true value lies in [p - Delta, p + Delta]. If that window sits strictly
inside one rounding cell, the cell's value is provably the correctly rounded output. Otherwise
the input is not certified (recompute it).
"""
from dataclasses import dataclass

import numpy as np

from .formats import CELLS
from .gelu import d4gelu, dgelu, gelu

U32 = 2.0**-24          # FP32 unit roundoff


@dataclass
class Table:
    lo: float
    hi: float
    h: float
    a: np.ndarray        # left ends (float64, exact in FP32)
    coef32: np.ndarray   # (n, 4) FP32 coefficients c0..c3 in t = x - a
    delta32: np.ndarray  # (n,) FP32 error bounds, rounded up
    interp: np.ndarray   # (n,) interpolation part of the bound, for reporting

    @property
    def n(self):
        return len(self.a)

    @property
    def nbytes(self):
        return self.n * 5 * 4   # 4 coefficients + 1 bound, FP32


def build(lo=-5.0, hi=5.0, h=1 / 16, samples=401):
    n = int(round((hi - lo) / h))
    a = lo + h * np.arange(n)
    y0, y1, d0, d1 = gelu(a), gelu(a + h), dgelu(a), dgelu(a + h)
    c2 = (3 * (y1 - y0) / h - 2 * d0 - d1) / h
    c3 = (d0 + d1 - 2 * (y1 - y0) / h) / h**2
    c64 = np.stack([y0, d0, c2, c3], 1)
    c32 = c64.astype(np.float32)
    t = np.linspace(0, h, samples)
    # Sampled maxima carry a 1% margin; the sampling step is far finer than the curvature scale.
    M = np.abs(d4gelu(a[:, None] + t)).max(1) * 1.01
    dmax = np.abs(dgelu(a[:, None] + t[:: max(1, samples // 100)])).max(1) * 1.01
    hp = np.array([1, h, h**2, h**3])
    interp = M * h**4 / 384
    coef_err = (np.abs(c64 - c32.astype(np.float64)) * hp).sum(1)
    gamma6 = 6 * U32 / (1 - 6 * U32)                    # Horner: 3 multiplies + 3 adds
    horner = gamma6 * (np.abs(c32.astype(np.float64)) * hp).sum(1)
    t_err = dmax * U32 * h                              # rounding of t = x - a in FP32
    delta = interp + coef_err + horner + t_err
    delta32 = np.nextafter(delta.astype(np.float32), np.float32(np.inf))
    return Table(lo, hi, h, a, c32, delta32, interp)


def evaluate(tab, x32):
    """FP32 Horner evaluation, as a GPU kernel would do it. Returns p (float32) and interval j."""
    x32 = np.asarray(x32, dtype=np.float32)
    j = np.clip(((x32.astype(np.float64) - tab.lo) / tab.h).astype(np.int64), 0, tab.n - 1)
    t = (x32 - tab.a[j].astype(np.float32)).astype(np.float32)
    c = tab.coef32[j]
    p = c[:, 3]
    for k in (2, 1, 0):
        p = (p * t).astype(np.float32)
        p = (p + c[:, k]).astype(np.float32)
    return p, j


def certify(tab, x32, fmt):
    """Returns (certified mask, predicted 16-bit encodings). Zero and sign-ambiguous cases are not certified."""
    p, j = evaluate(tab, x32)
    p64 = p.astype(np.float64)
    d = tab.delta32[j].astype(np.float64)
    bits, mlo, mhi = CELLS[fmt](p64)
    inside = np.asarray(x32, dtype=np.float64)
    inside = (inside >= tab.lo) & (inside < tab.hi)
    ok = inside & (p64 - d > mlo) & (p64 + d < mhi) & (bits != 0) & (bits != 0x8000)
    return ok, bits


P = 65537


def checksum(written_bits, predicted_bits, r):
    """D = sum r_i (written_i - predicted_i) mod P over the last axis. D == 0 when all outputs are right."""
    diff = written_bits.astype(np.int64) - predicted_bits.astype(np.int64)
    return (r.astype(np.int64) * diff).sum(-1) % P
