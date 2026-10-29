import math

import numpy as np

from gelucheck import route_a, route_b
from gelucheck.formats import CELLS, bf16_bits, bf16_value
from gelucheck.gelu import C, KAPPA, Phi, gelu, gelu_tanh, phi


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
        ok, bits = route_b.certify(tab, x, fmt)
        ref, _, _ = CELLS[fmt](gelu(x.astype(np.float64)))
        assert ok.mean() > 0.99
        assert not np.any(ok & (bits != ref))


def test_checksum_catches_any_single_error():
    rng = np.random.default_rng(3)
    pred = rng.integers(0, 65536, (5000, 256)).astype(np.uint16)
    r = rng.integers(1, route_b.P, (5000, 256))
    assert np.all(route_b.checksum(pred, pred, r) == 0)
    bad = pred.copy()
    bad[np.arange(5000), rng.integers(0, 256, 5000)] ^= rng.integers(1, 65536, 5000).astype(np.uint16)
    assert np.all(route_b.checksum(bad, pred, r) != 0)
