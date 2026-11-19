"""Route B checks: error bound, certification rates, wrong outputs, and the checksum.

Usage: python scripts/run_route_b.py [--n 400000] [--pairs 20000000]
"""
import argparse
import math

import numpy as np

from gelucheck import route_b
from gelucheck.formats import CELLS, round_bits
from gelucheck.gelu import d4gelu, gelu, kernel_fp32


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400_000, help="inputs per experiment")
    ap.add_argument("--pairs", type=int, default=20_000_000, help="trials for the two-error escape rate")
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    tab = route_b.build(-5.0, 5.0, 1 / 16)
    sup = np.abs(d4gelu(np.linspace(-12, 12, 2_400_001))).max()
    print("== Table")
    print(f"intervals {tab.n}; table {tab.nbytes} bytes; [-20, 20] version {route_b.build(-20, 20, 1/16).nbytes} bytes")
    print(f"sup |g''''| = {sup:.10f} (4/sqrt(2 pi) = {4 / math.sqrt(2 * math.pi):.10f}); "
          f"interpolation bound h^4 sup/384 = {sup * tab.h**4 / 384:.3e}")
    print(f"Delta range {tab.delta32.min():.2e} .. {tab.delta32.max():.2e}")
    print(f"kernel error bound range {tab.kern32.min():.2e} .. {tab.kern32.max():.2e}")

    xt = np.linspace(tab.lo, tab.hi - 1e-9, 2_000_001).astype(np.float32)
    p, j = route_b.evaluate(tab, xt)
    err = np.abs(gelu(xt.astype(np.float64)) - p.astype(np.float64))
    print(f"dense check, {xt.size} points: max |g - P| = {err.max():.3e}; max error / Delta = "
          f"{(err / tab.delta32[j]).max():.3f} (must be < 1)")

    print("== Certification (share of outputs settled without recomputation)")
    print("target exact: the correctly rounded GELU. target kernel: the FP32 kernel 0.5 x (1 + erf(x / sqrt 2)),")
    print("modelled with erf 0 and 2 ulp off, with and without a fused multiply-add. 'differs' counts certified")
    print("predictions that are not the reference's output (must be 0 for the target's own reference).")
    xn = rng.standard_normal(4 * args.n)
    inputs = {"standard normal": xn[(xn >= -5) & (xn < 5)][: args.n].astype(np.float32),
              "uniform [-5, 5]": rng.uniform(-5, 5, args.n).astype(np.float32)}
    for name, x32 in inputs.items():
        ref = gelu(x32.astype(np.float64))
        models = [kernel_fp32(x32, shift, fma) for shift in (-2, 0, 2) for fma in (False, True)]
        for target in ("exact", "kernel"):
            row = []
            for fmt in ("FP16", "BF16"):
                ok, bits = route_b.certify(tab, x32, fmt, target)
                d_exact = int((ok & (bits != CELLS[fmt](ref)[0])).sum())
                d_kern = max(int((ok & (bits != round_bits(y, fmt))).sum()) for y in models)
                row.append(f"{fmt} {100 * ok.mean():.2f}% (differs from exact {d_exact}, from kernel {d_kern})")
            print(f"{name:16s} target {target:6s} " + ", ".join(row))

    print("== Checksum mod 65537, blocks of 256")
    trials, B = 200_000, 256
    r = rng.integers(1, route_b.P, size=(trials, B))
    pred = rng.integers(0, 65536, size=(trials, B)).astype(np.uint16)
    written = pred.copy()
    idx = rng.integers(0, B, trials)
    written[np.arange(trials), idx] ^= rng.integers(1, 65536, trials).astype(np.uint16)
    missed = int((route_b.checksum(written, pred, r) == 0).sum())
    print(f"one wrong output per block: {trials} trials, missed {missed} (theory: never)")
    r1, r2 = rng.integers(1, route_b.P, args.pairs), rng.integers(1, route_b.P, args.pairs)
    esc = int(((r1 * 12345 + r2 * -777) % route_b.P == 0).sum())
    print(f"two fixed wrong outputs: {args.pairs} trials, escaped {esc}, rate {esc / args.pairs:.2e} "
          f"(bound 1/(p-1) = {1 / (route_b.P - 1):.2e})")


if __name__ == "__main__":
    main()
