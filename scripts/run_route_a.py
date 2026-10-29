"""Route A checks: false alarms on correct 16-bit outputs vs. detection of single bit flips.

Usage: python scripts/run_route_a.py [--blocks 20000]
"""
import argparse

import numpy as np

from gelucheck import route_a
from gelucheck.formats import ROUND
from gelucheck.gelu import gelu_tanh

TOLS = (0.01, 0.1, 0.5, 1, 3, 10)


def sweep(rng, nb, B=256):
    """Fixed and rounding-aware tolerances; FP32 inputs, FP16 outputs. Returns dict of rows."""
    x = rng.standard_normal((nb, B)).astype(np.float32).astype(np.float64)
    y16 = gelu_tanh(x).astype(np.float16)
    flipped, _ = route_a.flip_one_bit(y16, rng)
    e_clean = route_a.mismatch(x, y16.astype(np.float64))
    e_flip = route_a.mismatch(x, flipped)
    fixed = [(t, 100 * route_a.fails(e_clean, t).mean(), 100 * route_a.fails(e_flip, t).mean()) for t in TOLS]
    s_c, f_c = route_a.block_stat(e_clean)
    s_f, f_f = route_a.block_stat(e_flip)
    b_c, b_f = route_a.rounding_bound(x, y16), route_a.rounding_bound(x, flipped)
    aware = [(a, 100 * ((~f_c) | (s_c > a * b_c)).mean(), 100 * ((~f_f) | (s_f > a * b_f)).mean()) for a in (1, 2, 4)]
    return {"fixed": fixed, "aware": aware}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", type=int, default=20_000)
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    res = sweep(rng, args.blocks)
    print("== FP32 inputs, FP16 outputs, blocks of 256")
    print("fixed tolerance | false alarms (correct blocks) | random single bit flips caught")
    for t, fa, det in res["fixed"]:
        print(f"{t:15g} | {fa:28.2f}% | {det:6.1f}%")
    print("rounding-aware tolerance (alpha x sum of propagated half-ulps)")
    for a, fa, det in res["aware"]:
        print(f"alpha {a:9g} | {fa:28.2f}% | {det:6.1f}%")

    print("== False alarms vs. block size (fixed tolerance 0.1)")
    for B in (32, 256, 1024):
        x = rng.standard_normal((max(2000, args.blocks * 256 // B // 4), B)).astype(np.float32).astype(np.float64)
        e = route_a.mismatch(x, ROUND["FP16"](gelu_tanh(x)))
        print(f"B = {B:5d}: {100 * route_a.fails(e, 0.1).mean():.2f}%")

    print("== Output format and input precision (false alarms at tolerance 0.1 / 1 / 10)")
    for inp in ("FP32 inputs", "16-bit inputs"):
        for fmt in ("FP16", "BF16"):
            x = rng.standard_normal((args.blocks, 256))
            x = x.astype(np.float32).astype(np.float64) if inp == "FP32 inputs" else ROUND[fmt](x)
            x[x == 0] = 2.0**-8
            e = route_a.mismatch(x, ROUND[fmt](gelu_tanh(x)))
            print(f"{inp:14s} {fmt}: " + " / ".join(f"{100 * route_a.fails(e, t).mean():.2f}%" for t in (0.1, 1, 10)))


if __name__ == "__main__":
    main()
