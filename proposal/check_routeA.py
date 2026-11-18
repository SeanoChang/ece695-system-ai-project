"""Route A on correct outputs: how far apart are the two sides of the batch identity per block?

Work in logs (as a real checker must, to avoid overflow): per element, the identity says
log|x - y| = -k*(x + c x^3) + log|y|. With 16-bit outputs y, the per-element mismatch is
e_i = log|x - y_q| - (-k(x+cx^3) + log|y_q|); a block compares sum(e_i) with 0.
A block fails on CORRECT outputs when |sum e_i| exceeds the tolerance, or a term is log(0).
"""
import math
import numpy as np

k, c = 2 * math.sqrt(2 / math.pi), 0.044715
rng = np.random.default_rng(0)
B, NB = 256, 20_000


def bf16(v):
    f = v.astype(np.float32).view(np.uint32).astype(np.uint64)
    f = (f + 0x7FFF + ((f >> 16) & 1)) & 0xFFFF0000
    return f.astype(np.uint32).view(np.float32).astype(np.float64)


q = {"FP16": lambda v: v.astype(np.float16).astype(np.float64), "BF16": bf16}
for in_name in ("FP32 inputs", "16-bit inputs"):
    for out in ("FP16", "BF16"):
        x = rng.standard_normal(B * NB)
        x = x.astype(np.float32).astype(np.float64) if in_name == "FP32 inputs" else q[out](x)
        x[x == 0] = 2.0**-8
        y = q[out](x / (1 + np.exp(-k * (x + c * x**3))))
        with np.errstate(divide="ignore", invalid="ignore"):
            e = np.log(np.abs(x - y)) - (-k * (x + c * x**3) + np.log(np.abs(y)))
        e = e.reshape(NB, B)
        nonfinite = ~np.isfinite(e).all(1)
        s = np.abs(np.where(np.isfinite(e), e, 0).sum(1))
        res = [f"{100 * (nonfinite | (s > tol)).mean():6.2f}%" for tol in (1e-1, 1.0, 10.0)]
        print(f"{in_name:14s} {out}: blocks failing on correct outputs at tol 0.1 / 1 / 10 (log units): "
              f"{' / '.join(res)}; median |sum e| = {np.median(s):.2f}")

# detection: flip one random bit of one FP16 output per block, FP32 inputs, tolerance 10 (no false alarms above)
x = rng.standard_normal(B * NB).astype(np.float32).astype(np.float64)
y16 = (x / (1 + np.exp(-k * (x + c * x**3)))).astype(np.float16)
xb, yb = x.reshape(NB, B), y16.reshape(NB, B).copy()
idx = rng.integers(0, B, NB)
bit = rng.integers(0, 16, NB).astype(np.uint16)
v = yb.view(np.uint16)
v[np.arange(NB), idx] ^= (np.uint16(1) << bit)
yf = yb.astype(np.float64)
with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
    e = np.log(np.abs(xb - yf)) - (-k * (xb + c * xb**3) + np.log(np.abs(yf)))
s = np.abs(np.where(np.isfinite(e), e, 0).sum(1))
caught = (~np.isfinite(e).all(1)) | (s > 10.0)
for lo, hi, nm in ((0, 10, "mantissa bits 0-9"), (10, 15, "exponent bits 10-14"), (15, 16, "sign bit 15")):
    m = (bit >= lo) & (bit < hi)
    print(f"tol 10, one flipped FP16 output bit, {nm}: caught {100 * caught[m].mean():.1f}%")

# trade-off at several tolerances (FP32 inputs, FP16 outputs): false alarms on clean blocks vs. detection
xc = rng.standard_normal(B * NB).astype(np.float32).astype(np.float64).reshape(NB, B)
yc = (xc / (1 + np.exp(-k * (xc + c * xc**3)))).astype(np.float16).astype(np.float64)
with np.errstate(divide="ignore", invalid="ignore"):
    ec = np.log(np.abs(xc - yc)) - (-k * (xc + c * xc**3) + np.log(np.abs(yc)))
sc = np.abs(np.where(np.isfinite(ec), ec, 0).sum(1)); fc = ~np.isfinite(ec).all(1)
print("tolerance | false alarms (clean) | caught (one bit flip, any bit)")
for tol in (0.01, 0.1, 0.5, 1, 3, 10):
    print(f"{tol:9} | {100 * (fc | (sc > tol)).mean():19.2f}% | {100 * ((~np.isfinite(e).all(1)) | (s > tol)).mean():6.1f}%")
