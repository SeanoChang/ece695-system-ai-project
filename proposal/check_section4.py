"""Quick checks of the claims in Section 4 of the GELU checksum proposal.

1. Error bound and table size (h = 1/16 on [-5, 5]).
2. Rebuild the FP32 cubic Hermite table, certify FP16/BF16 outputs, compare with a double GELU.
3. Checksum mod 65537: single errors always caught; multi-error escape rate about 1/(p-1).
"""
import math
import numpy as np

rng = np.random.default_rng(0)
SQ2PI = math.sqrt(2 * math.pi)
erf = np.vectorize(math.erf)


def g(x):  # GELU in double
    return x * 0.5 * (1 + erf(x / math.sqrt(2)))


def dg(x):  # derivative: Phi(x) + x phi(x)
    return 0.5 * (1 + erf(x / math.sqrt(2))) + x * np.exp(-x * x / 2) / SQ2PI


def g4(x):
    return (-x**4 + 7 * x**2 - 4) * np.exp(-x * x / 2) / SQ2PI


# ── 1. bound and table size ─────────────────────────────────────────────
h, lo, hi = 1 / 16, -5.0, 5.0
n_int = int(round((hi - lo) / h))
xs = np.linspace(-12, 12, 2_400_001)
sup_g4 = np.abs(g4(xs)).max()
print(f"[1] intervals = {n_int}  (claim 160)")
print(f"    sup|g''''| = {sup_g4:.10f}  vs 4/sqrt(2pi) = {4 / SQ2PI:.10f}")
print(f"    interp bound = {sup_g4 * h**4 / 384:.4e}  (claim < 6.35e-8)")
print(f"    table bytes = {n_int * 5 * 4}  (claim 3.2 KB: 4 coeffs + 1 bound, FP32)")

# ── 2. build the table ─────────────────────────────────────────────────
a = lo + h * np.arange(n_int)                      # left ends (exact in FP32)
y0, y1, d0, d1 = g(a), g(a + h), dg(a), dg(a + h)
# Hermite cubic in t = x - a: c0 + c1 t + c2 t^2 + c3 t^3
c0, c1 = y0, d0
c2 = (3 * (y1 - y0) / h - 2 * d0 - d1) / h
c3 = (d0 + d1 - 2 * (y1 - y0) / h) / h**2
C64 = np.stack([c0, c1, c2, c3], 1)
C32 = C64.astype(np.float32)
u = 2.0**-24
# per-interval bound: interpolation + coefficient storage + Horner rounding + t rounding
M = np.array([np.abs(g4(np.linspace(aj, aj + h, 401))).max() * 1.01 for aj in a])
interp = M * h**4 / 384
hp = np.array([1, h, h**2, h**3])
coef_err = (np.abs(C64 - C32.astype(np.float64)) * hp).sum(1)
gam6 = 6 * u / (1 - 6 * u)
horner = gam6 * (np.abs(C32.astype(np.float64)) * hp).sum(1)
dmax = np.array([np.abs(dg(np.linspace(aj, aj + h, 101))).max() * 1.01 for aj in a])
t_err = dmax * u * h                                # |g'| on the interval times |t rounding| <= u*h
Delta = interp + coef_err + horner + t_err
Delta32 = np.nextafter(Delta.astype(np.float32), np.float32(np.inf))   # stored, rounded up
print(f"[2] Delta range {Delta.min():.2e} .. {Delta.max():.2e}; interp part max {interp.max():.2e}")

# empirical check that |g - P| <= Delta on dense points
xt = np.linspace(lo, hi - 1e-9, 2_000_001).astype(np.float32)


def poly32(x32):
    j = np.clip(((x32.astype(np.float64) - lo) / h).astype(np.int64), 0, n_int - 1)
    t = (x32 - a[j].astype(np.float32)).astype(np.float32)
    c = C32[j]
    p = c[:, 3]
    for k in (2, 1, 0):
        p = (p * t).astype(np.float32)
        p = (p + c[:, k]).astype(np.float32)
    return p, j


p, j = poly32(xt)
err = np.abs(g(xt.astype(np.float64)) - p.astype(np.float64))
print(f"    max |g - P_fp32| = {err.max():.3e}; max err/Delta = {(err / Delta[j]).max():.3f} (must be < 1)")


# rounding to 16-bit formats, with the two neighbours that bound the cell
def fp16_cell(v64):
    v = v64.astype(np.float16)
    dn = np.nextafter(v, np.float16(-np.inf)).astype(np.float64)
    up = np.nextafter(v, np.float16(np.inf)).astype(np.float64)
    vv = v.astype(np.float64)
    return v.view(np.uint16), (vv + dn) / 2, (vv + up) / 2


def bf16_round(v64):
    """Round double to bf16 (nearest, ties to even) via exact candidates."""
    f = v64.astype(np.float32).view(np.uint32)
    lo_b = (f & 0xFFFF0000).astype(np.uint32)
    hi_b = (lo_b + 0x10000).astype(np.uint32)
    lo_v = lo_b.view(np.float32).astype(np.float64)
    hi_v = hi_b.view(np.float32).astype(np.float64)
    # for negative numbers lo_b has smaller magnitude; compare distances directly
    dlo, dhi = np.abs(v64 - lo_v), np.abs(v64 - hi_v)
    even_lo = ((lo_b >> 16) & 1) == 0
    pick_lo = (dlo < dhi) | ((dlo == dhi) & even_lo)
    b = np.where(pick_lo, lo_b, hi_b)
    return (b >> 16).astype(np.uint16)


def bf16_cell(v64):
    b = bf16_round(v64).astype(np.uint32)
    vv = (b << 16).view(np.float32).astype(np.float64)
    dn = ((b - 1) << 16).view(np.float32).astype(np.float64)
    up = ((b + 1) << 16).view(np.float32).astype(np.float64)
    # for negative values bit order is reversed; take sorted neighbours
    lo_n, hi_n = np.minimum(dn, up), np.maximum(dn, up)
    return b.astype(np.uint16), (vv + lo_n) / 2, (vv + hi_n) / 2


def run(name, x32):
    p, j = poly32(x32)
    p64 = p.astype(np.float64)
    D = Delta32[j].astype(np.float64)
    gref = g(x32.astype(np.float64))
    out = {}
    for fmt, cell in (("FP16", fp16_cell), ("BF16", bf16_cell)):
        v, mlo, mhi = cell(p64)
        ok = (p64 - D > mlo) & (p64 + D < mhi) & (v != 0)   # exclude zero/sign cases -> fallback
        ref, _, _ = cell(gref)
        wrong = int((ok & (v != ref)).sum())
        out[fmt] = (ok.mean() * 100, wrong)
        if name.startswith("Exp 2"):
            bins = np.floor(x32).astype(int)
            print(f"      {fmt} certified by unit bin:", " ".join(f"[{b},{b+1}):{ok[bins==b].mean()*100:.1f}" for b in range(-5, 5)))
    print(f"    {name:22s} FP16 certified {out['FP16'][0]:.2f}% (wrong {out['FP16'][1]}), "
          f"BF16 certified {out['BF16'][0]:.2f}% (wrong {out['BF16'][1]})")


N = 400_000
xn = rng.standard_normal(4 * N)
xn = xn[(xn >= lo) & (xn < hi)][:N].astype(np.float32)
xu = rng.uniform(lo, hi, N).astype(np.float32)
print("    proposal: normal FP16 99.57 / BF16 99.94; uniform FP16 99.62 / BF16 99.91; 0 wrong")
run("Exp 1 standard normal", xn)
run("Exp 2 uniform [-5,5]", xu)

# ── 3. checksum mod 65537 ──────────────────────────────────────────────
P = 65537
B = 256
trials = 200_000
r = rng.integers(1, P, size=(trials, B))
# single wrong output: any nonzero difference in (-65535, 65535)
idx = rng.integers(0, B, trials)
diff = rng.integers(1, 65536, trials) * rng.choice([-1, 1], trials)
D1 = (r[np.arange(trials), idx] * diff) % P
print(f"[3] single-error trials {trials}: missed {int((D1 == 0).sum())} (claim: always caught)")
# fixed two-error pattern, fresh random r each trial: escape rate should be about 1/65536
T2 = 20_000_000
r1 = rng.integers(1, P, T2)
r2 = rng.integers(1, P, T2)
d1, d2 = 12345, -777
esc = int(((r1 * d1 + r2 * d2) % P == 0).sum())
print(f"    two-error trials {T2}: escaped {esc}, rate {esc / T2:.2e} vs bound 1/(p-1) = {1 / (P - 1):.2e}")
print(f"    two independent checksums: bound (1/(p-1))^2 = {(1 / (P - 1))**2:.2e} = 2^-32 is {2.0**-32:.2e}")
