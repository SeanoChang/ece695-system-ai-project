import math

import numpy as np
import pytest

from gelucheck import route_a, route_b
from gelucheck.formats import CELLS, bf16_bits, bf16_value, round_bits
from gelucheck.gelu import C, KAPPA, Phi, gelu, gelu_tanh, kernel_fp32, phi


def test_tanh_forms_agree():
    x = np.linspace(-8, 8, 1001)
    t = 0.5 * x * (1 + np.tanh(math.sqrt(2 / math.pi) * (x + C * x**3)))
    assert np.abs(t - gelu_tanh(x)).max() < 1e-12


def test_batch_identity_exact_in_float64():
    x = np.random.default_rng(0).standard_normal(64)
    e = route_a.mismatch(x, gelu_tanh(x))
    s, finite = route_a.block_stat(e)
    assert finite and s < 1e-9


def test_tail_bound():
    x = np.linspace(-30, 30, 60_001)
    m = phi(x) > 1e-300
    gap = np.abs(x) * Phi(-np.abs(x))
    assert (gap[m] / phi(x)[m]).max() <= 1.0


def test_bf16_rounding_matches_brute_force():
    # every BF16 value between two neighbours: round-to-nearest must pick the closer one (ties to even)
    rng = np.random.default_rng(1)
    v = rng.standard_normal(20_000).astype(np.float32).astype(np.float64) * 10.0 ** rng.integers(-6, 6, 20_000)
    b = bf16_bits(v).astype(np.uint32)
    for nb in ((b - 1) & 0xFFFF, (b + 1) & 0xFFFF):
        assert np.all(np.abs(bf16_value(b) - v) <= np.abs(bf16_value(nb) - v))


def test_route_b_bound_holds_and_certified_outputs_are_exact():
    tab = route_b.build(-5.0, 5.0, 1 / 16)
    x = np.random.default_rng(2).uniform(-5, 5, 100_000).astype(np.float32)
    p, j = route_b.evaluate(tab, x)
    assert (np.abs(gelu(x.astype(np.float64)) - p.astype(np.float64)) < tab.delta32[j]).all()
    for fmt in ("FP16", "BF16"):
        ok, bits = route_b.certify(tab, x, fmt, "exact")
        ref, _, _ = CELLS[fmt](gelu(x.astype(np.float64)))
        assert ok.mean() > 0.99
        assert not np.any(ok & (bits != ref))


def test_kernel_target_matches_every_kernel_inside_the_bound():
    # The real kernel is not correctly rounded. Model it with erf up to 2 ulp off, with and without a fused
    # multiply-add. A certified prediction must equal the output of every such kernel.
    tab = route_b.build(-5.0, 5.0, 1 / 16)
    rng = np.random.default_rng(4)
    edges = (tab.a[rng.integers(0, tab.n, 20_000)] + rng.integers(-3, 4, 20_000) * 2.0**-22).astype(np.float32)
    x = np.concatenate([rng.uniform(-5, 5, 150_000), rng.standard_normal(150_000), edges]).astype(np.float32)
    g = gelu(x.astype(np.float64))
    _, j = route_b.evaluate(tab, x)
    inside = (x >= -5) & (x < 5)
    exact_misses = 0
    for shift in (-2, -1, 0, 1, 2):
        for fma in (False, True):
            y = kernel_fp32(x, shift, fma)
            assert (np.abs(y.astype(np.float64) - g)[inside] <= tab.kern32[j][inside]).all()
            for fmt in ("FP16", "BF16"):
                ok, bits = route_b.certify(tab, x, fmt, "kernel")
                assert not np.any(ok & (bits != round_bits(y, fmt))), (shift, fma, fmt)
            ok, bits = route_b.certify(tab, x, "FP16", "exact")
            exact_misses += int((ok & (bits != round_bits(y, "FP16"))).sum())
    assert exact_misses > 1000          # the "exact" target is not a safe reference for this kernel
    ok, _ = route_b.certify(tab, rng.standard_normal(200_000).astype(np.float32), "FP16", "kernel")
    assert ok.mean() > 0.99


def test_evaluate_keeps_the_input_shape():
    tab = route_b.build(-5.0, 5.0, 1 / 16)
    x = np.random.default_rng(5).uniform(-6, 6, (7, 33, 5)).astype(np.float32)
    x[0, 0, :3] = np.nan, np.inf, -np.inf
    p, j = route_b.evaluate(tab, x)
    pf, jf = route_b.evaluate(tab, x.ravel())
    assert p.shape == x.shape and np.array_equal(p.ravel(), pf, equal_nan=True) and np.array_equal(j.ravel(), jf)
    for target in route_b.TARGETS:
        ok, bits = route_b.certify(tab, x, "FP16", target)
        okf, bitsf = route_b.certify(tab, x.ravel(), "FP16", target)
        assert ok.shape == x.shape and np.array_equal(ok.ravel(), okf) and np.array_equal(bits.ravel(), bitsf)
        assert not ok[0, 0, :3].any() and not ok[(x < -5) | (x >= 5)].any()


def test_build_rejects_knots_that_are_not_exact():
    for lo, hi, h in ((-5.0, 5.0, 0.1), (-5.0, 5.03, 1 / 16), (-5.01, 5.0, 1 / 16)):
        with pytest.raises(ValueError):
            route_b.build(lo, hi, h)


def test_checksum_catches_any_single_error():
    rng = np.random.default_rng(3)
    pred = rng.integers(0, 65536, (5000, 256)).astype(np.uint16)
    r = rng.integers(1, route_b.P, (5000, 256))
    assert np.all(route_b.checksum(pred, pred, r) == 0)
    bad = pred.copy()
    bad[np.arange(5000), rng.integers(0, 256, 5000)] ^= rng.integers(1, 65536, 5000).astype(np.uint16)
    assert np.all(route_b.checksum(bad, pred, r) != 0)
    # with a mask, an error on an output that is not certified is outside the checksum
    hit = np.zeros(bad.shape, bool)
    hit[bad != pred] = True
    assert np.all(route_b.checksum(bad, pred, r, ~hit) == 0)
    assert np.all(route_b.checksum(bad, pred, r, hit) != 0)
