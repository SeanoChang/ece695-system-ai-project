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
                            b.i0, b.i0, b.i0, b.i0, b.i0, b.i0, b.i0, 0, 0, -1, 0, N,
                            MODE=mode, FUSED=True, INJ=False, BLOCK=b.block, **b.const)
        ptx = getattr(k, "asm", {}).get("ptx")
        if ptx is None:
            pytest.skip("this Triton version does not return the compiled kernel")
        size[mode] = sum(1 for line in ptx.splitlines()
                         if line.strip() and not line.strip().startswith(("//", ".", "{", "}", "$")))
    assert size[1] > 1.5 * size[0], size


@pytest.mark.parametrize("stage", gpu.COST_STAGES)
def test_cost_stages_store_the_same_outputs_and_count_what_their_check_sees(clean, stage):
    b, x, y = clean
    y2 = torch.empty_like(y)
    b.ctr.zero_()                                   # "none" has no counter: it must leave these at 0
    b.cost(stage, x, y2)
    c = b.counts()
    assert np.array_equal(bits(y2), bits(y))
    if stage == "predict":                          # every in-range output against its prediction, no cell test
        xn = x.cpu().numpy()
        p, _ = route_b.evaluate(b.tab, xn)
        ins = (xn >= b.tab.lo) & (xn < b.tab.hi)
        expect = int((ins & (p.astype(np.float32).astype(np.float16).view(np.uint16).astype(np.int64) != bits(y))).sum())
        assert expect > 0
        if INTERP:                                  # no fused multiply-add: the same predictions
            assert c["bad"] == expect
        else:                                       # a fused multiply-add can move a prediction by one ulp
            assert abs(c["bad"] - expect) <= max(10, expect // 100)
    else:
        assert c["bad"] == 0
    if stage in ("append", "list"):                 # the list holds exactly the inputs that rb_list lists
        _, ref = run(b, "rb_list", x)
        assert c["uncertified"] == ref["uncertified"] > 0
    else:                                           # no stage before "append" writes the list
        assert c["uncertified"] == 0


@pytest.mark.skipif(INTERP, reason="needs compiled code")
def test_cost_stages_are_not_removed_by_the_compiler_and_end_at_rb_list(clean):
    # Each stage must add instructions to the one before it; a stage the compiler could drop would time as 0.
    # The last kernel stage must use the registers of k_std rb_list, so its occupancy is that of the real check.
    b, x, y = clean
    size, regs = [], []
    for s in gpu.COST_STAGES[:-1]:
        k = b.cost(s, x, y.clone())
        ptx = getattr(k, "asm", {}).get("ptx")
        if ptx is None:
            pytest.skip("this Triton version does not return the compiled kernel")
        size.append(sum(1 for line in ptx.splitlines()
                        if line.strip() and not line.strip().startswith(("//", ".", "{", "}", "$"))))
        regs.append(k.n_regs)
    assert all(a < b2 for a, b2 in zip(size, size[1:])), size
    ref = b.run("rb_list", x, y.clone())
    assert regs[-1] == ref.n_regs and k.n_spills == 0, (regs, ref.n_regs)


def sm_faults(b, bad_sm, seed=7):
    """A fault in the FP32 output of every element (bits 13-30), active only where the SM is bad_sm."""
    rng = np.random.default_rng(seed)
    f = b.faults(N, bad_sm=bad_sm)
    f.fo[:] = torch.tensor((1 << rng.integers(13, 31, N)).astype(np.int32), device=DEV)
    return f


def fault_effect(b, x, y, seed=7):
    """Where the masks of sm_faults change the stored output, if they hit the first computation."""
    f = sm_faults(b, -1, seed)
    yf, _ = run(b, "none", x, faults=f)
    return bits(yf) != bits(y)


def per_element(b, per_block):
    return np.repeat(per_block.cpu().numpy(), b.block)[:N]


def a_bad_sm(b, x):
    """The SM that computed block 1 in one launch: an SM that the kernels really use."""
    f = b.faults(N)
    run(b, "none", x, faults=f)
    sm = f.sm.cpu().numpy()
    assert (sm >= 0).all()                          # every block recorded its SM
    return int(sm[1])


def test_sm_fault_changes_only_outputs_computed_on_the_bad_sm(clean):
    b, x, y = clean
    bad = a_bad_sm(b, x)
    f = sm_faults(b, bad)
    yf, _ = run(b, "none", x, faults=f)
    on_bad = per_element(b, f.sm) == bad
    changed = bits(yf) != bits(y)
    assert on_bad.any() and not changed[~on_bad].any()
    assert changed.sum() > on_bad.sum() // 2


def test_fused_dup_misses_an_sm_fault_and_route_b_catches_it_on_certified_outputs(clean):
    b, x, y = clean
    bad = a_bad_sm(b, x)
    eff = fault_effect(b, x, y)
    f = sm_faults(b, bad)
    yf, c = run(b, "dup", x, faults=f)
    assert (bits(yf) != bits(y)).sum() > 100 and c["bad"] == 0     # both copies ran on the bad SM
    for variant in ("rb_list", "rb_sum"):
        f = sm_faults(b, bad)
        yf, _ = run(b, variant, x, faults=f)
        changed, ok, flag = bits(yf) != bits(y), f.ok.cpu().numpy() > 0, f.flag.cpu().numpy() > 0
        assert (changed & ok).sum() > 100
        # The table check runs on no SM's GELU: on certified outputs it flags exactly the changed ones.
        # Uncertified outputs go to the second pass; a second pass on the bad SM recomputes a wrong value
        # and flags a correct output. A flag can only appear where the fault changes the output.
        assert not (flag & ~eff).any()
        if variant == "rb_list":
            assert np.array_equal(flag & ok, changed & ok)
        else:
            assert not (flag & ok).any()            # rb_sum checks certified outputs through d only
            wrong = np.pad(changed & ok, (0, -N % b.block)).reshape(-1, b.block).any(1)
            assert np.array_equal(f.d.cpu().numpy() != 0, wrong)
        if INTERP:
            # The second pass: list entry k is recomputed by program k // BLOCK (fewer than FB_GRID * BLOCK
            # entries), whose SM in the interpreter is that program id mod SIM_SMS.
            count = b.counts()["uncertified"]
            assert 0 < count < gpu.FB_GRID * b.block
            idx = b.list[:count].cpu().numpy()
            pass_bad = np.zeros(N, bool)
            pass_bad[idx] = (np.arange(count) // b.block) % gpu.SIM_SMS == bad
            first_bad = per_element(b, f.sm) == bad
            assert pass_bad.any() and (first_bad & ~ok).any()
            assert np.array_equal(flag & ~ok, eff & ~ok & (first_bad != pass_bad))


@pytest.mark.parametrize("shifted", (False, True))
def test_separate_dup_catches_an_sm_fault_exactly_where_its_check_ran_elsewhere(clean, shifted):
    b, x, y = clean
    bad = a_bad_sm(b, x)
    nb = -(-N // b.block)
    f = sm_faults(b, bad)
    yf, _ = run(b, "none", x, faults=f)             # the faulty kernel; its fault stays in yf
    changed = bits(yf) != bits(y)
    f2 = sm_faults(b, bad)                          # the same masks: the checker's copy is faulty on bad too
    b.run("dup", x, yf, fused=False, faults=f2, shift=nb // 2 if shifted else 0)
    first_bad = per_element(b, f.sm) == bad
    check_bad = per_element(b, f2.sm2) == bad
    eff = fault_effect(b, x, y)
    # A flag exactly where the fault changes the output and one of the two computations ran on the bad SM:
    # a wrong output checked elsewhere (caught), or a correct output checked on the bad SM (a false alarm
    # from the defective checker). Both on the bad SM: the same wrong value twice, missed.
    assert np.array_equal(f2.flag.cpu().numpy() > 0, eff & (first_bad != check_bad))
    caught = changed & ~check_bad
    if INTERP:                                      # the interpreter's SM is the program id mod 8
        assert caught.any() == shifted


@pytest.mark.skipif(INTERP, reason="needs a GPU")
def test_recorded_sm_ids_are_real_sms(clean):
    b, x, y = clean
    f = b.faults(N)
    run(b, "none", x, faults=f)
    sm = f.sm.cpu().numpy()
    assert sm.min() >= 0 and sm.max() < torch.cuda.get_device_properties(0).multi_processor_count
    assert len(np.unique(sm)) > 1
