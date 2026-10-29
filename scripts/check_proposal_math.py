"""Numerical checks of the formulas in the proposal (tanh form, batch identity, tail bound, sizes)."""
import math

import numpy as np

from gelucheck.gelu import C, KAPPA, Phi, gelu_tanh, phi


def main():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(64)
    t = 0.5 * x * (1 + np.tanh(math.sqrt(2 / math.pi) * (x + C * x**3)))
    print(f"tanh form vs. sigmoid form: max difference {np.abs(t - gelu_tanh(x)).max():.1e}")
    y = gelu_tanh(x)
    lhs = np.log(np.abs(x - y)).sum()
    rhs = -KAPPA * (x.sum() + C * (x**3).sum()) + np.log(np.abs(y)).sum()
    same_sign = np.prod(np.sign(x - y)) == np.prod(np.sign(y))
    print(f"batch identity in float64, 64 inputs: log sides {lhs:.12f} vs {rhs:.12f}; signs agree {same_sign}")
    xs = np.linspace(-30, 30, 600_001)
    gap = np.abs(xs) * Phi(-np.abs(xs))                  # |g(x) - max(x, 0)|, no cancellation
    m = phi(xs) > 1e-300
    print(f"tail bound |g - max(x,0)| <= phi(x): max ratio {(gap[m] / phi(xs)[m]).max():.6f} (must be <= 1)")
    print(f"table sizes: [-20,20] {40 * 16} intervals, {640 * 5 * 4} bytes; [-5,5] {160 * 5 * 4} bytes")


if __name__ == "__main__":
    main()
