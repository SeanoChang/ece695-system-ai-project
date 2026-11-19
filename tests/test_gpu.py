"""Tests of the Triton kernels against the NumPy reference.

Run on a CUDA GPU:                  PYTHONPATH=. python -m pytest tests/test_gpu.py
Run on a CPU (Linux, logic only):   TRITON_INTERPRET=1 PYTHONPATH=. python -m pytest tests/test_gpu.py
Skipped when Triton is missing, or when there is neither a GPU nor the interpreter.
"""
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

INTERP = os.environ.get("TRITON_INTERPRET") == "1"
if not (INTERP or torch.cuda.is_available()):
    pytest.skip("needs a CUDA GPU or TRITON_INTERPRET=1", allow_module_level=True)

from gelucheck import gpu, route_b  # noqa: E402
from gelucheck.gelu import gelu  # noqa: E402

DEV = gpu.device()
N = (1 << 13) + 37 if INTERP else (1 << 21) + 37        # not a multiple of the block: the last block is partial
CHECKED = ("dup", "range", "rb_block", "rb_list", "rb_sum")


def bits(y):
    return y.cpu().numpy().view(np.uint16).astype(np.int64)


def inputs(n, seed=0):
    """Standard normal, uniform over and beyond the table, and values next to the knots."""
    rng = np.random.default_rng(seed)
    k = n // 4
    knots = rng.integers(-80, 81, k) / 16 + rng.integers(-4, 5, k) * 2.0**-22
    x = np.concatenate([rng.standard_normal(n - 3 * k), rng.uniform(-5, 5, k), rng.uniform(-6.5, 6.5, k), knots])
    return torch.tensor(rng.permutation(x).astype(np.float32), device=DEV)


@pytest.fixture(scope="module")
def clean():
    x = inputs(N)
    b = gpu.Bench(N)
    y = torch.empty(N, dtype=torch.float16, device=DEV)
    b.run("none", x, y)
    return b, x, y


def run(b, variant, x, fused=True, faults=None, y=None):
    y = torch.empty(x.numel(), dtype=torch.float16, device=DEV) if y is None else y
    b.run(variant, x, y, fused=fused, faults=faults)
    return y, b.counts()


def test_kernel_output_is_inside_the_error_bound(clean):
    b, x, y = clean
    xn = x.cpu().numpy()
    ins = (xn >= b.tab.lo) & (xn < b.tab.hi)
    _, j = route_b.evaluate(b.tab, xn)
    g = gelu(xn.astype(np.float64))
    # y is FP16: compare with the cell of the true value widened by the bound
    lo = (g - b.tab.kern32[j]).astype(np.float16).astype(np.float64)
    hi = (g + b.tab.kern32[j]).astype(np.float16).astype(np.float64)
    yv = y.cpu().numpy().astype(np.float64)
    assert ((yv >= lo) & (yv <= hi))[ins].all()


def test_interval_index_is_exact_next_to_the_knots():
    # x just below a knot: (x - LO) rounds up to the knot in FP32. The kernel must still pick the interval
    # that holds x, with an offset t inside [0, H).
    tab = route_b.build()
    rng = np.random.default_rng(6)
    n = 4096 if INTERP else 1 << 20
    xn = (rng.integers(-80, 80, n) / 16 + rng.integers(-6, 7, n) * 2.0 ** rng.integers(-24, -19, n)).astype(np.float32)
    xn = xn[(xn >= -5) & (xn < 5)]
    x = torch.tensor(xn, device=DEV)
    j = torch.zeros(x.numel(), dtype=torch.int32, device=DEV)
    t = torch.zeros(x.numel(), dtype=torch.float32, device=DEV)
    gpu.k_index[(-(-x.numel() // 1024),)](x, j, t, x.numel(), LO=-5.0, H=1 / 16, NT=160, BLOCK=1024)
    p_ref, j_ref = route_b.evaluate(tab, xn)
    rounds_up = ((xn + np.float32(5)) * np.float32(16)).astype(np.float32) >= (j_ref + 1)
    assert rounds_up.sum() > n // 100               # the case under test occurs
    assert np.array_equal(j.cpu().numpy(), j_ref)
    tv = t.cpu().numpy()
    assert (tv >= 0).all() and (tv < 1 / 16).all()
    assert np.array_equal(tv, (xn - tab.a[j_ref].astype(np.float32)).astype(np.float32))


def test_certify_matches_numpy(clean):
    b, x, y = clean
    f = b.faults(N)
    run(b, "rb_list", x, faults=f)
    ok, vb = f.ok.cpu().numpy().astype(bool), f.vb.cpu().numpy()
    ok_ref, vb_ref = route_b.certify(b.tab, x.cpu().numpy(), "FP16", "kernel")
    if INTERP:                                      # no fused multiply-add: the same bits
        assert np.array_equal(ok, ok_ref) and np.array_equal(vb[ok], vb_ref[ok].astype(np.int64))
    else:                                           # a fused multiply-add can move p by one ulp
        assert (ok != ok_ref).mean() < 1e-3
        both = ok & ok_ref
        assert np.array_equal(vb[both], vb_ref[both].astype(np.int64))
    assert ok.mean() > 0.6 and not ok[(x.cpu().numpy() < -5) | (x.cpu().numpy() >= 5)].any()
    assert np.array_equal(vb[ok], bits(y)[ok])      # every certified prediction is the kernel's output


@pytest.mark.parametrize("fused", (True, False))
@pytest.mark.parametrize("variant", CHECKED)
def test_clean_run_has_no_false_alarm(clean, variant, fused):
    b, x, y = clean
    y2, c = run(b, variant, x, fused=fused, y=None if fused else y.clone())
    assert c["bad"] == 0
    assert np.array_equal(bits(y2), bits(y))        # every variant stores the same outputs
    f = b.faults(N)                                 # the injection build with no fault: same outputs, no flag
    y3, c3 = run(b, variant, x, fused=fused, faults=f, y=None if fused else y.clone())
    assert c3 == c and np.array_equal(bits(y3), bits(y)) and int(f.flag.sum()) == 0 and int(f.d.sum()) == 0
    if variant in ("rb_list", "rb_sum"):
        assert c["uncertified"] == N - int(f.ok.sum())
    if variant == "rb_block":
        blocks = np.pad(f.ok.cpu().numpy(), (0, -N % b.block), constant_values=1).reshape(-1, b.block)
        assert c["uncertified"] == int((blocks == 0).any(1).sum())


def one_fault_per_block(b, seed, site, persist=False):
    """A random single-bit fault in one element of every block."""
    rng = np.random.default_rng(seed)
    nb = -(-N // b.block)
    pos = np.minimum(np.arange(nb) * b.block + rng.integers(0, b.block, nb), N - 1)
    f = b.faults(N, persist)
    mask = torch.tensor((1 << rng.integers(0, 32, nb)).astype(np.uint32).view(np.int32), device=DEV)
    (f.fe if site == "erf" else f.fo)[torch.tensor(pos, device=DEV)] = mask
    return f, pos


@pytest.mark.parametrize("site", ("erf", "out"))
def test_every_effective_transient_fault_is_caught(clean, site):
    b, x, y = clean
    for variant in ("dup", "rb_block", "rb_list", "rb_sum", "range"):
        f, pos = one_fault_per_block(b, 1, site)
        yf, c = run(b, variant, x, faults=f)
        changed = bits(yf) != bits(y)
        assert changed.sum() > len(pos) // 4 and not changed[np.setdiff1d(np.arange(N), pos)].any()
        flag = f.flag.cpu().numpy() > 0
        if variant == "range":                      # a range check sees only large errors
            assert not (flag & ~changed).any()
        elif variant == "rb_sum":
            blk = np.zeros(len(pos), bool)
            np.logical_or.at(blk, np.flatnonzero(flag) // b.block, True)
            blk |= f.d.cpu().numpy() != 0
            assert np.array_equal(blk, changed[pos])
            assert c["bad"] == int(blk.sum())
        else:
            assert np.array_equal(flag, changed)
            assert c["bad"] == int(changed.sum())


def test_checksum_matches_numpy(clean):
    b, x, y = clean
    f, _ = one_fault_per_block(b, 2, "out")
    rng = np.random.default_rng(3)                  # and several faults in some blocks
    extra = torch.tensor(rng.integers(0, N, 64), device=DEV)
    f.fo[extra] = torch.tensor(rng.integers(1, 1 << 31, 64).astype(np.int32), device=DEV)
    yf, _ = run(b, "rb_sum", x, faults=f)
    pad = lambda v: np.pad(v, (0, -N % b.block)).reshape(-1, b.block)
    d_ref = route_b.checksum(pad(bits(yf)), pad(f.vb.cpu().numpy()), b.r_np, pad(f.ok.cpu().numpy()) > 0)
    assert np.array_equal(f.d.cpu().numpy(), d_ref) and (d_ref != 0).sum() > 0


def test_fault_in_both_copies_escapes_duplication_but_not_route_b(clean):
    b, x, y = clean
    f, pos = one_fault_per_block(b, 4, "erf", persist=True)
    yf, c = run(b, "dup", x, faults=f)
    changed = bits(yf) != bits(y)
    assert changed.sum() > len(pos) // 4 and c["bad"] == 0
    for variant in ("rb_list", "rb_sum"):
        f, pos = one_fault_per_block(b, 4, "erf", persist=True)
        yf, c = run(b, variant, x, faults=f)
        ok = f.ok.cpu().numpy() > 0
        if variant == "rb_list":
            assert np.array_equal(f.flag.cpu().numpy() > 0, changed & ok)   # caught exactly where certified
        else:
            assert np.array_equal(f.d.cpu().numpy() != 0, (changed & ok)[pos])
            assert int(f.flag.sum()) == 0
        assert (changed & ok).sum() > 0


@pytest.mark.parametrize("variant", ("dup", "rb_list", "rb_sum", "rb_block"))
def test_separate_checker_sees_a_fault_in_the_stored_output(clean, variant):
    b, x, y = clean
    rng = np.random.default_rng(5)
    nb = -(-N // b.block)
    pos = np.minimum(np.arange(nb) * b.block + rng.integers(0, b.block, nb), N - 1)
    yb = y.cpu().numpy().view(np.uint16).copy()
    yb[pos] ^= (1 << rng.integers(0, 16, nb)).astype(np.uint16)
    ybad = torch.tensor(yb.view(np.float16), device=DEV)
    f = b.faults(N)
    _, c = run(b, variant, x, fused=False, faults=f, y=ybad)
    flag = f.flag.cpu().numpy() > 0
    if variant == "rb_sum":
        blk = np.zeros(nb, bool)
        np.logical_or.at(blk, np.flatnonzero(flag) // b.block, True)
        assert (blk | (f.d.cpu().numpy() != 0)).all() and c["bad"] == nb
    else:
        assert np.array_equal(np.flatnonzero(flag), np.sort(pos)) and c["bad"] == nb


def test_special_inputs_do_not_break_the_checkers():
    v = [np.nan, np.inf, -np.inf, 0.0, -0.0, 1e-30, -1e-30, 5.0, -5.0, np.nextafter(np.float32(5), 0), 1e30, -1e30]
    x = torch.tensor(np.array(v * 7, dtype=np.float32), device=DEV)
    b = gpu.Bench(x.numel())
    y0, _ = run(b, "none", x)
    for variant in ("dup", "rb_block", "rb_list", "rb_sum"):
        y, c = run(b, variant, x)
        assert c["bad"] == 0 and np.array_equal(bits(y), bits(y0))


def test_matmul_epilogue_variants_agree_and_raise_no_false_alarm():
    M, K, N2 = (40, 64, 128) if INTERP else (300, 1024, 4096)
    g = torch.Generator().manual_seed(0)
    a = (torch.randn(M, K, generator=g) / K**0.5).half().to(DEV)
    w = torch.randn(K, N2, generator=g).half().to(DEV)
    b = gpu.Bench(M * N2)
    out = {}
    for variant in gpu.MM_VARIANTS:
        c = torch.zeros(M, N2, dtype=torch.float16, device=DEV)
        b.mm(variant, a, w, c)
        out[variant] = c
        if variant not in ("matmul", "none"):
            assert b.counts()["bad"] == 0, variant
            assert np.array_equal(bits(c), bits(out["none"])), variant
    pre = a.float() @ w.float()
    assert (out["matmul"].float() - pre).abs().max() < 2e-2
    assert (out["none"].float() - torch.nn.functional.gelu(pre)).abs().max() < 2e-2


@pytest.mark.skipif(INTERP, reason="needs compiled code")
def test_duplicate_is_not_merged_by_the_compiler(clean):
    b, x, y = clean
    grid = (-(-N // b.block),)
    size = {}
    for mode in (0, 1):
        k = gpu.k_std[grid](x, y.clone(), b.bad, b.cnt, b.list, b.xval, b.coef, b.marg, b.r,
                            b.i0, b.i0, b.i0, b.i0, b.i0, b.i0, 0, 0, N,
                            MODE=mode, FUSED=True, INJ=False, BLOCK=b.block, **b.const)
        ptx = getattr(k, "asm", {}).get("ptx")
        if ptx is None:
            pytest.skip("this Triton version does not return the compiled kernel")
        size[mode] = sum(1 for line in ptx.splitlines()
                         if line.strip() and not line.strip().startswith(("//", ".", "{", "}", "$")))
    assert size[1] > 1.5 * size[0], size
