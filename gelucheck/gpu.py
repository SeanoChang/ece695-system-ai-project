"""Triton kernels: GELU with each checker, fault injection, and GELU fused into a matrix multiply.

The protected kernel is y = 0.5 x (1 + erf(x / sqrt 2)) in FP32, stored as FP16.

Variants (the check that runs with, or after, that kernel):
  none      no check
  dup       a second GELU, compared bit for bit. The second input goes through an XOR with a run-time
            zero, so the compiler cannot merge the two copies. It is the cheapest true duplicate.
  range     max(min(x, 0), -0.1701) - tol <= y <= max(x, 0) + tol
  rb_block  Route B, each output compared with its prediction; a block with an uncertified input
            recomputes the whole block
  rb_list   Route B, each output compared; uncertified inputs go to a list for a second, small kernel
  rb_sum    Route B, block checksum D = sum r_i (written_i - predicted_i) mod 65537 over the certified
            outputs; uncertified inputs go to the list

fused=True checks the value in the register of the kernel that computes GELU. fused=False is a separate
checker kernel that reads y back from memory, so it also sees a fault in the store or in memory.

Fault model (injection builds only): faults(bad_sm=-1) is a fault in the first computation, and also in
the recomputation if persist. faults(bad_sm=k) is a defective SM: every computation that runs on SM k gets
the fault, and no other computation does. Each block records the SM it ran on (sm, and sm2 for a separate
checker), read from the hardware (%smid); the CPU interpreter has no SMs and uses the program id mod 8.
A separate checker can check block (pid + shift) mod nb, so that its copy runs on another SM.

Counters: bad counts wrong outputs (rb_sum: blocks with D != 0, plus wrong uncertified outputs);
uncertified counts list entries (rb_block: blocks that were recomputed).

Runs on a CUDA GPU, or on the CPU with TRITON_INTERPRET=1 (slow; for tests of the logic).
"""
import os
from dataclasses import dataclass

import numpy as np
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import smid as _cuda_smid

from . import route_b

VARIANTS = ("none", "dup", "range", "rb_block", "rb_list", "rb_sum")
MM_VARIANTS = ("matmul", "none", "dup", "rb_block", "rb_list", "rb_sum")
COST_STAGES = ("none", "count", "predict", "cell", "append", "list")   # see k_cost
BLOCK = 1024           # elements per program in the standalone kernels
TILE = 64              # the fused matmul uses TILE x TILE (or 16 x TILE) tiles
FB_GRID = 40           # programs of the second pass; fixed, so the host never reads the list length
SIM_SMS = 8            # the CPU interpreter has no SMs: _sm() is the program id mod SIM_SMS there
_SIM = tl.constexpr(os.environ.get("TRITON_INTERPRET") == "1")


def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


@triton.jit
def _gelu(x):
    return 0.5 * x * (1.0 + tl.erf(x * 0.7071067811865476))


@triton.jit
def _gelu_f(x, fe, fo):
    """GELU with a fault: fe is XORed into the FP32 erf result, fo into the FP32 output."""
    e = tl.erf(x * 0.7071067811865476)
    e = (e.to(tl.int32, bitcast=True) ^ fe).to(tl.float32, bitcast=True)
    y = 0.5 * x * (1.0 + e)
    return (y.to(tl.int32, bitcast=True) ^ fo).to(tl.float32, bitcast=True)


@triton.jit
def _sm():
    """The SM that runs this program: %smid on the GPU; the program id mod SIM_SMS in the interpreter."""
    if _SIM:
        return tl.program_id(0) % SIM_SMS
    else:
        return _cuda_smid()


@triton.jit
def _active(bad_sm, persist):
    """Fault factors (0 or 1) for the first computation and for a recomputation in this program.

    bad_sm < 0: the first computation always, the recomputation if persist (the older model).
    bad_sm >= 0: both exactly when this program runs on SM bad_sm (a defective SM).
    """
    old = (bad_sm < 0).to(tl.int32)
    on = (_sm() == bad_sm).to(tl.int32)
    return old + (1 - old) * on, old * persist + (1 - old) * on


@triton.jit
def _opaque(x, zero):
    """x again, through an XOR with a run-time zero: the compiler must treat it as a new value."""
    return (x.to(tl.int32, bitcast=True) ^ zero).to(tl.float32, bitcast=True)


@triton.jit
def _bits16(v16):
    return v16.to(tl.int16, bitcast=True).to(tl.int32) & 0xFFFF


@triton.jit
def _f16_from_bits(b):
    return (b & 0xFFFF).to(tl.int16).to(tl.float16, bitcast=True).to(tl.float32)


@triton.jit
def _index(x, LO: tl.constexpr, H: tl.constexpr, NT: tl.constexpr):
    """Interval j and offset t = x - a_j in [0, H) for x in [LO, LO + NT H). Other x (and NaN) are not in range."""
    inr = (x >= LO) & (x < LO + NT * H)
    xc = tl.where(inr, x, LO)                       # keeps the table index valid for NaN and out-of-range x
    j = tl.floor((xc - LO) * (1.0 / H)).to(tl.int32)
    j = tl.minimum(tl.maximum(j, 0), NT - 1)
    t = xc - (LO + j.to(tl.float32) * H)
    # (xc - LO) can round up to the next knot. Then t < 0: step back, so that t is in [0, H).
    j = tl.maximum(j - (t < 0.0).to(tl.int32), 0)
    t = xc - (LO + j.to(tl.float32) * H)
    return inr, j, t


@triton.jit
def _lookup(x, coef_ptr, marg_ptr, LO: tl.constexpr, H: tl.constexpr, NT: tl.constexpr):
    """Route B, part 1: the interval of x, the offset t in it, its cubic's coefficients and its margin."""
    inr, j, t = _index(x, LO, H, NT)
    c0 = tl.load(coef_ptr + j * 4 + 0)
    c1 = tl.load(coef_ptr + j * 4 + 1)
    c2 = tl.load(coef_ptr + j * 4 + 2)
    c3 = tl.load(coef_ptr + j * 4 + 3)
    km = tl.load(marg_ptr + j)
    return inr, t, c0, c1, c2, c3, km


@triton.jit
def _poly(t, c0, c1, c2, c3):
    """Route B, part 2: the cubic, in Horner form (the order of route_b.evaluate)."""
    return ((c3 * t + c2) * t + c1) * t + c0


@triton.jit
def _cell(p, km, inr):
    """Route B, part 3: certified if [p - km, p + km] lies strictly inside one FP16 rounding cell."""
    vb = _bits16(p.to(tl.float16))
    vf = _f16_from_bits(vb)
    n1 = _f16_from_bits(vb - 1)
    n2 = _f16_from_bits(vb + 1)
    mlo = (vf + tl.minimum(n1, n2)) * 0.5           # the two midpoints of the rounding cell, exact in FP32
    mhi = (vf + tl.maximum(n1, n2)) * 0.5
    ok = inr & ((p - mlo) > km) & ((mhi - p) > km) & ((vb & 0x7FFF) != 0)
    return ok, vb


@triton.jit
def _certify(x, coef_ptr, marg_ptr, LO: tl.constexpr, H: tl.constexpr, NT: tl.constexpr):
    """Route B. Returns (certified, predicted FP16 bits). Mirrors route_b.certify operation for operation."""
    inr, t, c0, c1, c2, c3, km = _lookup(x, coef_ptr, marg_ptr, LO, H, NT)
    return _cell(_poly(t, c0, c1, c2, c3), km, inr)


@triton.jit
def _append(unc, idx, val, cnt_ptr, list_ptr, xval_ptr):
    """Append (index, GELU input) where unc is true to the global list. One atomic per program."""
    u = unc.to(tl.int32)
    base = tl.atomic_add(cnt_ptr, tl.sum(u, axis=0))
    pos = base + tl.cumsum(u, axis=0) - 1
    tl.store(list_ptr + pos, idx, mask=unc)
    tl.store(xval_ptr + pos, val, mask=unc)


@triton.jit(do_not_specialize=["zero", "persist", "bad_sm", "shift", "n"])
def k_std(x_ptr, y_ptr, bad_ptr, cnt_ptr, list_ptr, xval_ptr, coef_ptr, marg_ptr, r_ptr,
          fe_ptr, fo_ptr, flag_ptr, d_ptr, ok_ptr, vb_ptr, sm_ptr, zero, persist, bad_sm, shift, n,
          MODE: tl.constexpr, FUSED: tl.constexpr, INJ: tl.constexpr,
          LO: tl.constexpr, H: tl.constexpr, NT: tl.constexpr, BLOCK: tl.constexpr):
    """Standalone GELU (FP32 in, FP16 out) with check MODE (index into VARIANTS).

    INJ builds add what the experiments need and the timed builds must not pay for: fault masks, one flag
    per element, the checksum per block, the certified mask and prediction per element, and the SM of
    each block. A separate checker (not FUSED) checks block (pid + shift) mod the grid size.
    """
    pid = tl.program_id(0)
    blk = pid
    if not FUSED:
        blk = (pid + shift) % tl.num_programs(0)
    offs = blk * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m, other=1.0)
    fe1 = tl.zeros((BLOCK,), dtype=tl.int32)        # fault masks of the first computation and of the recomputation
    fo1 = tl.zeros((BLOCK,), dtype=tl.int32)
    fe2 = fe1
    fo2 = fo1
    if INJ:
        fe = tl.load(fe_ptr + offs, mask=m, other=0)
        fo = tl.load(fo_ptr + offs, mask=m, other=0)
        first, again = _active(bad_sm, persist)
        fe1 = fe * first
        fo1 = fo * first
        fe2 = fe * again
        fo2 = fo * again
        tl.store(sm_ptr + blk, _sm())
    if FUSED:
        if INJ:
            y = _gelu_f(x, fe1, fo1).to(tl.float16)
        else:
            y = _gelu(x).to(tl.float16)
        tl.store(y_ptr + offs, y, mask=m)
    else:
        y = tl.load(y_ptr + offs, mask=m, other=0.0)
    yb = _bits16(y)
    bad = offs < 0                                   # all false

    if MODE == 1:
        x2 = _opaque(x, zero)
        if INJ:
            y2 = _gelu_f(x2, fe2, fo2)
        else:
            y2 = _gelu(x2)
        bad = m & (_bits16(y2.to(tl.float16)) != yb)
    if MODE == 2:
        yf = y.to(tl.float32)
        tol = 1e-3 * tl.abs(x) + 1e-4
        inside = (yf >= tl.maximum(tl.minimum(x, 0.0), -0.1701) - tol) & (yf <= tl.maximum(x, 0.0) + tol)
        bad = m & (inside.to(tl.int32) == 0)         # also flags NaN
    if MODE >= 3:
        ok, vb = _certify(x, coef_ptr, marg_ptr, LO, H, NT)
        okm = m & ok
        unc = m & (ok.to(tl.int32) == 0)
        if INJ:
            tl.store(ok_ptr + offs, okm.to(tl.int32), mask=m)
            tl.store(vb_ptr + offs, vb, mask=m)
        if MODE == 3:
            bad = okm & (vb != yb)
            if tl.sum(unc.to(tl.int32), axis=0) > 0:
                x2 = _opaque(x, zero)
                if INJ:
                    y2 = _gelu_f(x2, fe2, fo2)
                else:
                    y2 = _gelu(x2)
                bad = bad | (unc & (_bits16(y2.to(tl.float16)) != yb))
                tl.atomic_add(cnt_ptr, 1)
        if MODE == 4:
            bad = okm & (vb != yb)
            _append(unc, offs, x, cnt_ptr, list_ptr, xval_ptr)
        if MODE == 5:
            r = tl.load(r_ptr + tl.arange(0, BLOCK))
            diff = tl.where(okm, yb - vb, 0)
            # |r diff| < 2^32 and BLOCK terms: the sum needs 64 bits. Only D == 0 matters, so the sign
            # convention of % does not.
            d = tl.sum(r.to(tl.int64) * diff.to(tl.int64), axis=0) % 65537
            tl.atomic_add(bad_ptr, (d != 0).to(tl.int32))
            if INJ:
                tl.store(d_ptr + blk, ((d + 65537) % 65537).to(tl.int32))
            _append(unc, offs, x, cnt_ptr, list_ptr, xval_ptr)
    if MODE >= 1:
        if MODE != 5:
            tl.atomic_add(bad_ptr, tl.sum(bad.to(tl.int32), axis=0))
        if INJ:
            tl.store(flag_ptr + offs, bad.to(tl.int32), mask=m)


@triton.jit(do_not_specialize=["zero", "n"])
def k_cost(x_ptr, y_ptr, bad_ptr, cnt_ptr, list_ptr, xval_ptr, coef_ptr, marg_ptr, zero, n,
           STAGE: tl.constexpr, LO: tl.constexpr, H: tl.constexpr, NT: tl.constexpr, BLOCK: tl.constexpr):
    """Fused GELU plus the first STAGE parts of rb_list, to time each part (COST_STAGES[STAGE]).

    0 none     GELU and store, as k_std "none"
    1 count    + the alarm counter: one atomic per block (the host also resets the counters)
    2 predict  + interval index, table loads and the cubic; every in-range output is compared with its
               prediction (no cell test, so this stage flags the outputs it could not certify)
    3 cell     + the margin and the rounding-cell test; only certified outputs are compared
    4 append   + the append of uncertified inputs to the list: the code of k_std rb_list
               (stage 5, list, adds the second pass on the host)
    Every stage uses its results in a real check, as rb_list does; no stage adds code only to keep work
    alive, so register use and occupancy at each stage are those of a real kernel. The lookup and the
    cubic are one step: the cubic alone has no use without a prediction to compare.
    zero is a run-time 0 added to the counter, so the atomic of stage 1 is not folded away.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m, other=1.0)
    y = _gelu(x).to(tl.float16)
    tl.store(y_ptr + offs, y, mask=m)
    if STAGE >= 1:
        bad = offs < 0                               # all false: stage 1 times the counter alone
        if STAGE >= 2:
            inr, t, c0, c1, c2, c3, km = _lookup(x, coef_ptr, marg_ptr, LO, H, NT)
            p = _poly(t, c0, c1, c2, c3)
            if STAGE == 2:
                bad = m & inr & (_bits16(p.to(tl.float16)) != _bits16(y))
            else:
                ok, vb = _cell(p, km, inr)
                bad = m & ok & (vb != _bits16(y))
                if STAGE >= 4:
                    _append(m & (ok.to(tl.int32) == 0), offs, x, cnt_ptr, list_ptr, xval_ptr)
        tl.atomic_add(bad_ptr, tl.sum(bad.to(tl.int32), axis=0) + zero)


@triton.jit(do_not_specialize=["n"])
def k_index(x_ptr, j_ptr, t_ptr, n, LO: tl.constexpr, H: tl.constexpr, NT: tl.constexpr, BLOCK: tl.constexpr):
    """Test kernel: the interval index and offset that _certify uses."""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    inr, j, t = _index(tl.load(x_ptr + offs, mask=m, other=0.0), LO, H, NT)
    tl.store(j_ptr + offs, j, mask=m)
    tl.store(t_ptr + offs, t, mask=m)


@triton.jit(do_not_specialize=["persist", "bad_sm"])
def k_fallback(xval_ptr, list_ptr, cnt_ptr, y_ptr, bad_ptr, fe_ptr, fo_ptr, flag_ptr, persist, bad_sm,
               INJ: tl.constexpr, BLOCK: tl.constexpr):
    """Second pass: recompute only the listed (uncertified) inputs and compare with the stored outputs.

    Reads the list length on the device and walks the list with a fixed grid: no host round trip.
    """
    count = tl.load(cnt_ptr)
    step = tl.num_programs(0) * BLOCK
    for start in range(tl.program_id(0) * BLOCK, count, step):
        k = start + tl.arange(0, BLOCK)
        m = k < count
        i = tl.load(list_ptr + k, mask=m, other=0)
        x = tl.load(xval_ptr + k, mask=m, other=1.0)
        y = tl.load(y_ptr + i, mask=m, other=0.0)
        if INJ:
            _, again = _active(bad_sm, persist)
            fe = tl.load(fe_ptr + i, mask=m, other=0) * again
            fo = tl.load(fo_ptr + i, mask=m, other=0) * again
            y2 = _gelu_f(x, fe, fo)
        else:
            y2 = _gelu(x)
        bad = m & (_bits16(y2.to(tl.float16)) != _bits16(y))
        tl.atomic_add(bad_ptr, tl.sum(bad.to(tl.int32), axis=0))
        if INJ:
            tl.atomic_add(flag_ptr + i, bad.to(tl.int32), mask=m)


@triton.jit(do_not_specialize=["zero", "M", "N", "K"])
def k_mm(a_ptr, b_ptr, c_ptr, bad_ptr, cnt_ptr, list_ptr, xval_ptr, coef_ptr, marg_ptr, r_ptr, zero, M, N, K,
         MODE: tl.constexpr, LO: tl.constexpr, H: tl.constexpr, NT: tl.constexpr,
         BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """C = GELU(A @ B): A (M x K) and B (K x N) in FP16, FP32 accumulator, C in FP16.

    MODE indexes MM_VARIANTS. The checks are the fused ones of k_std, on the accumulator tile.
    """
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    rn = tl.program_id(1) * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        a = tl.load(a_ptr + rm[:, None] * K + (k0 + rk)[None, :],
                    mask=(rm[:, None] < M) & ((k0 + rk)[None, :] < K), other=0.0)
        b = tl.load(b_ptr + (k0 + rk)[:, None] * N + rn[None, :],
                    mask=((k0 + rk)[:, None] < K) & (rn[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
    m = (rm[:, None] < M) & (rn[None, :] < N)
    cptr = c_ptr + rm[:, None] * N + rn[None, :]
    if MODE == 0:
        tl.store(cptr, acc.to(tl.float16), mask=m)
    else:
        y = _gelu(acc).to(tl.float16)
        tl.store(cptr, y, mask=m)
        yb = _bits16(y)
        if MODE == 2:
            bad = m & (_bits16(_gelu(_opaque(acc, zero)).to(tl.float16)) != yb)
            tl.atomic_add(bad_ptr, tl.sum(tl.sum(bad.to(tl.int32), axis=1), axis=0))
        if MODE >= 3:
            ok, vb = _certify(acc, coef_ptr, marg_ptr, LO, H, NT)
            okm = m & ok
            unc = m & (ok.to(tl.int32) == 0)
            idx = rm[:, None] * N + rn[None, :]
            if MODE == 3:
                bad = okm & (vb != yb)
                if tl.sum(tl.sum(unc.to(tl.int32), axis=1), axis=0) > 0:
                    bad = bad | (unc & (_bits16(_gelu(_opaque(acc, zero)).to(tl.float16)) != yb))
                    tl.atomic_add(cnt_ptr, 1)
                tl.atomic_add(bad_ptr, tl.sum(tl.sum(bad.to(tl.int32), axis=1), axis=0))
            if MODE == 4:
                bad = okm & (vb != yb)
                tl.atomic_add(bad_ptr, tl.sum(tl.sum(bad.to(tl.int32), axis=1), axis=0))
            if MODE == 5:
                r = tl.reshape(tl.load(r_ptr + tl.arange(0, BM * BN)), (BM, BN))
                diff = tl.where(okm, yb - vb, 0)
                d = tl.sum(tl.sum(r.to(tl.int64) * diff.to(tl.int64), axis=1), axis=0) % 65537
                tl.atomic_add(bad_ptr, (d != 0).to(tl.int32))
            if MODE >= 4:
                _append(tl.reshape(unc, (BM * BN,)), tl.reshape(idx, (BM * BN,)), tl.reshape(acc, (BM * BN,)),
                        cnt_ptr, list_ptr, xval_ptr)


@dataclass
class Faults:
    """One launch injects many faults. All tensors are on the device.

    fe, fo   (n,) int32  XOR masks for the FP32 erf result and the FP32 GELU output of each element
    persist  bool        True: the recomputation (duplicate, second pass) gets the same fault, as from a
                         defective unit. False: only the first computation gets it.
    flag     (n,) int32  out: > 0 where an element check fired
    d        (n_blocks,) int32  out: rb_sum checksum of each block
    ok, vb   (n,) int32  out: Route B certified mask and predicted FP16 bits
    sm, sm2  (n_blocks,) int32  out: the SM that computed each block (fused run or the faulty kernel), and
                         the SM that checked it (separate checker); -1 until a run writes it
    bad_sm   int         -1: the model above. k >= 0: a defective SM k (persist is then ignored)
    """
    fe: torch.Tensor
    fo: torch.Tensor
    flag: torch.Tensor
    d: torch.Tensor
    ok: torch.Tensor
    vb: torch.Tensor
    sm: torch.Tensor
    sm2: torch.Tensor
    persist: bool = False
    bad_sm: int = -1


class Bench:
    """Table tensors and scratch buffers for inputs of up to n elements."""

    def __init__(self, n, tab=None, target="kernel", dev=None, block=BLOCK, seed=0):
        self.tab = tab if tab is not None else route_b.build()
        self.dev = dev or device()
        self.n, self.block = n, block
        self.const = dict(LO=float(self.tab.lo), H=float(self.tab.h), NT=int(self.tab.n))
        self.coef = torch.tensor(self.tab.coef32, device=self.dev).contiguous()
        self.marg = torch.tensor(self.tab.margin32[target], device=self.dev).contiguous()
        r = np.random.default_rng(seed).integers(1, route_b.P, max(block, TILE * TILE))
        self.r_np = r[:block]
        self.r = torch.tensor(r, dtype=torch.int32, device=self.dev)
        self.ctr = torch.zeros(2, dtype=torch.int32, device=self.dev)       # [bad, uncertified]
        self.bad, self.cnt = self.ctr[0:], self.ctr[1:]
        self.list = torch.zeros(n, dtype=torch.int32, device=self.dev)
        self.xval = torch.zeros(n, dtype=torch.float32, device=self.dev)
        self.i0 = torch.zeros(1, dtype=torch.int32, device=self.dev)        # stands in for unused pointers

    def faults(self, n, persist=False, bad_sm=-1):
        z = lambda k: torch.zeros(k, dtype=torch.int32, device=self.dev)
        nb = triton.cdiv(n, self.block)
        unset = lambda: torch.full((nb,), -1, dtype=torch.int32, device=self.dev)
        return Faults(z(n), z(n), z(n), z(nb), z(n), z(n), unset(), unset(), persist, bad_sm)

    def run(self, variant, x, y, fused=True, faults=None, shift=0):
        """Launch the kernels of one variant; read counts() afterwards.

        shift (separate checkers only): check block (pid + shift) mod nb, so the copy most likely runs on
        another SM than the kernel that computed y. Where it ran is recorded in faults.sm2.
        Returns the compiled k_std kernel (its n_regs and n_spills on a GPU; None in the interpreter).
        """
        n, mode, f = x.numel(), VARIANTS.index(variant), faults
        assert n <= self.n and x.dtype == torch.float32 and y.dtype == torch.float16 and x.dim() == y.dim() == 1
        assert x.is_contiguous() and y.is_contiguous() and y.numel() == n, "the kernels index with stride 1"
        assert fused or mode, "a separate checker needs a check"
        assert shift == 0 or not fused, "only a separate checker can check another block"
        assert shift >= 0, "the kernel's signed remainder would give a negative block index"
        if mode:
            self.ctr.zero_()
        fe, fo, flag, d, ok, vb = (f.fe, f.fo, f.flag, f.d, f.ok, f.vb) if f else (self.i0,) * 6
        sm = (f.sm if fused else f.sm2) if f else self.i0
        persist, bad_sm = (int(f.persist), int(f.bad_sm)) if f else (0, -1)
        k = k_std[(triton.cdiv(n, self.block),)](
            x, y, self.bad, self.cnt, self.list, self.xval, self.coef, self.marg, self.r,
            fe, fo, flag, d, ok, vb, sm, 0, persist, bad_sm, shift, n,
            MODE=mode, FUSED=fused, INJ=f is not None, BLOCK=self.block, **self.const)
        if mode >= 4:
            k_fallback[(FB_GRID,)](self.xval, self.list, self.cnt, y, self.bad, fe, fo, flag, persist, bad_sm,
                                   INJ=f is not None, BLOCK=self.block)
        return k

    def mm(self, variant, a, b, c):
        """C = GELU(A @ B) with the check fused into the matmul epilogue."""
        mode = MM_VARIANTS.index(variant)
        (M, K), N = a.shape, b.shape[1]
        assert M * N <= self.n and a.dtype == b.dtype == c.dtype == torch.float16
        assert a.is_contiguous() and b.is_contiguous() and c.is_contiguous() and c.shape == (M, N)
        BM = 16 if M <= 16 else TILE
        if mode >= 2:
            self.ctr.zero_()
        k_mm[(triton.cdiv(M, BM), triton.cdiv(N, TILE))](
            a, b, c, self.bad, self.cnt, self.list, self.xval, self.coef, self.marg, self.r, 0, M, N, K,
            MODE=mode, BM=BM, BN=TILE, BK=32, num_warps=4, num_stages=2, **self.const)
        if mode >= 4:
            k_fallback[(FB_GRID,)](self.xval, self.list, self.cnt, c.reshape(-1), self.bad, self.i0, self.i0,
                                   self.i0, 0, -1, INJ=False, BLOCK=self.block)

    def cost(self, stage, x, y):
        """Launch the cost stage `stage` (a name in COST_STAGES) of fused GELU on x, storing y.

        Returns the compiled main kernel (its n_regs and n_spills on a GPU; None in the interpreter)."""
        s, n = COST_STAGES.index(stage), x.numel()
        assert n <= self.n and x.dtype == torch.float32 and y.dtype == torch.float16 and x.dim() == y.dim() == 1
        assert x.is_contiguous() and y.is_contiguous() and y.numel() == n, "the kernels index with stride 1"
        if s:
            self.ctr.zero_()
        k = k_cost[(triton.cdiv(n, self.block),)](
            x, y, self.bad, self.cnt, self.list, self.xval, self.coef, self.marg, 0, n,
            STAGE=min(s, 4), BLOCK=self.block, **self.const)
        if s == 5:
            k_fallback[(FB_GRID,)](self.xval, self.list, self.cnt, y, self.bad, self.i0, self.i0, self.i0, 0, -1,
                                   INJ=False, BLOCK=self.block)
        return k

    def counts(self):
        bad, unc = self.ctr.tolist()
        return {"bad": bad, "uncertified": unc}
