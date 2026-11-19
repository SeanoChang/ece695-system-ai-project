"""Route B: a piecewise-cubic table that certifies the 16-bit GELU output.

Offline: on each interval [a, a + h], a cubic Hermite polynomial matches g and g' at both ends.
Its error is at most M h^4 / 384 (M = max |g''''| on the interval). A per-interval bound Delta
adds FP32 coefficient storage, Horner rounding and input-offset rounding, and is stored rounded up.

Runtime: p = P(x). The true value lies in [p - Delta, p + Delta]. Two targets:

  "exact"   If that window sits strictly inside one rounding cell, the cell's value is the correctly
            rounded output.
  "kernel"  A real kernel computes 0.5 x (1 + erf(x / sqrt 2)) in FP32 and is not correctly rounded.
            Its output lies within Kern of the true value (see kernel_bound). If the window widened by
            Kern sits strictly inside one cell, every output such a kernel can produce rounds to that
            cell's value. This holds for any kernel inside the bound, whatever its erf code.

Otherwise the input is not certified (recompute it).
"""
import math
from dataclasses import dataclass

import numpy as np

from .formats import CELLS
from .gelu import d4gelu, dgelu, gelu

U32 = 2.0**-24          # FP32 unit roundoff
SLACK = 1 + 2.0**-20    # covers the FP32 rounding of the cell test itself (relative error at most 2^-24)
TARGETS = ("kernel", "exact")


@dataclass
class Table:
    lo: float
    hi: float
    h: float
    a: np.ndarray        # left ends (float64, exact in FP32)
    coef32: np.ndarray   # (n, 4) FP32 coefficients c0..c3 in t = x - a
    delta32: np.ndarray  # (n,) FP32 bounds on |g - p|, rounded up
    interp: np.ndarray   # (n,) interpolation part of the bound, for reporting
    kern32: np.ndarray   # (n,) FP32 bounds on |kernel output - g| for the FP32 erf-form kernel, rounded up
    margin32: dict       # target -> (n,) FP32 margin of the cell test: (Delta [+ Kern]) * SLACK, rounded up

    @property
    def n(self):
        return len(self.a)

    @property
    def nbytes(self):
        return self.n * 5 * 4   # 4 coefficients + 1 bound, FP32


def kernel_bound(xmax, gmax, erf_ulps=2.0):
    """Bound on |y - g(x)| for y = fl(fl(0.5 x) * fl(1 + erf_lib(fl(x * c32)))) in FP32, |x| <= xmax, |g| <= gmax.

    Error of the computed (1 + erf(x / sqrt 2)), in units of U32:
      1.0             input: fl(x * c32) has relative error at most 2 U32, and |w erf'(w)| <= 0.484;
      erf_ulps + 0.5  erf library: within erf_ulps ulp of the correctly rounded result (CUDA states 2 ulp
                      for erff, from tests), so within erf_ulps + 0.5 ulp of erf, and ulp(erf) <= U32;
      1.0             the addition 1 + e (|1 + e| <= 2).
    The product with x / 2 (exact) adds one rounding, U32 |y|. A fused multiply-add only removes roundings.
    """
    k = 1.0 + (erf_ulps + 0.5) + 1.0
    return (0.5 * k * U32 * xmax + U32 * gmax) / (1 - U32)


def build(lo=-5.0, hi=5.0, h=1 / 16, samples=401, erf_ulps=2.0):
    n = int(round((hi - lo) / h))
    a = lo + h * np.arange(n)
    # The FP32 kernel finds the interval as floor((x - lo) / h) and t = x - a. Both need exact knots.
    if math.frexp(h)[0] != 0.5 or n * h != hi - lo or not (a.astype(np.float32) == a).all():
        raise ValueError("h must be a power of two, and lo and hi multiples of h that are exact in FP32")
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
    kern = kernel_bound(np.maximum(np.abs(a), np.abs(a + h)), np.abs(gelu(a[:, None] + t)).max(1) * 1.01, erf_ulps)
    d = delta32.astype(np.float64)
    margin32 = {"exact": _up32(d * SLACK), "kernel": _up32((d + kern) * SLACK)}
    return Table(lo, hi, h, a, c32, delta32, interp, _up32(kern), margin32)


def _up32(v):
    """Round float64 values up to FP32."""
    f = v.astype(np.float32)
    return np.where(f.astype(np.float64) < v, np.nextafter(f, np.float32(np.inf)), f)


def evaluate(tab, x32):
    """FP32 Horner evaluation, as a GPU kernel does it. Returns p (float32) and interval j, in the shape of x32.

    The interval index is exact (float64). The GPU kernel computes it in FP32 and steps back one interval
    when t < 0, which gives the same index.
    """
    x32 = np.asarray(x32, dtype=np.float32)
    with np.errstate(invalid="ignore"):
        j = np.floor((x32.astype(np.float64) - tab.lo) / tab.h)
    j = np.clip(np.nan_to_num(j, nan=0.0), 0, tab.n - 1).astype(np.int64)
    t = (x32 - tab.a[j].astype(np.float32)).astype(np.float32)
    c = tab.coef32[j]
    p = c[..., 3]
    for k in (2, 1, 0):
        p = (p * t).astype(np.float32)
        p = (p + c[..., k]).astype(np.float32)
    return p, j


def certify(tab, x32, fmt, target):
    """Returns (certified mask, predicted 16-bit encodings). Zero and sign-ambiguous cases are not certified.

    target is "kernel" or "exact" (see the module docstring). It has no default: the two give different
    predictions of what is safe, and a caller must say which reference it checks against.

    The cell test runs in FP32, operation for operation as in the GPU kernel. The cell midpoints are exact
    in FP32, and SLACK in the margin covers the rounding of the two subtractions.
    """
    x32 = np.asarray(x32, dtype=np.float32)
    p, j = evaluate(tab, x32)
    bits, mlo, mhi = CELLS[fmt](p.astype(np.float64))
    km = tab.margin32[target][j]
    with np.errstate(invalid="ignore", over="ignore"):
        inside = (x32 >= np.float32(tab.lo)) & (x32 < np.float32(tab.hi))
        ok = inside & ((p - mlo.astype(np.float32)) > km) & ((mhi.astype(np.float32) - p) > km)
    return ok & ((bits & 0x7FFF) != 0), bits


P = 65537


def checksum(written_bits, predicted_bits, r, mask=None):
    """D = sum r_i (written_i - predicted_i) mod P over the last axis. D == 0 when all outputs are right.

    With a mask, only the outputs where it is true (the certified ones) take part.
    """
    diff = written_bits.astype(np.int64) - predicted_bits.astype(np.int64)
    if mask is not None:
        diff = np.where(mask, diff, 0)
    return (r.astype(np.int64) * diff).sum(-1) % P
