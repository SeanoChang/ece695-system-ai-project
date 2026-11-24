"""GPU experiments: false alarms, every FP32 input, fault injection, timing, fused matmul, real models.

Usage (one idle GPU; about 10 minutes, mostly the model downloads on the first run):
    CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. .venv/bin/python scripts/run_gpu.py | tee results/gpu.txt
    ... scripts/run_gpu.py --only clean,inject        run some experiments
    TRITON_INTERPRET=1 PYTHONPATH=. python scripts/run_gpu.py --quick --only clean,inject   (CPU, tiny, logic only)

Experiments: clean, exhaustive, inject, sm_fault, timing, matmul, cost, models, share. One that fails is reported; the rest run.
"""
import argparse
import pathlib
import sys
import traceback
from typing import NamedTuple

import numpy as np
import torch
import torch.nn.functional as F
import triton

from gelucheck import gpu
from gelucheck.formats import fp16_cell
from gelucheck.gelu import gelu

DEV = gpu.device()
CHECKED = ("dup", "range", "rb_block", "rb_list", "rb_sum")
SEPARATE = ("dup", "range", "rb_list", "rb_sum")
C = 0.7071067811865476


def half(n):
    return torch.empty(n, dtype=torch.float16, device=DEV)


def b16(y):
    return y.view(torch.int16)


def draw(name, n, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, generator=g) if name == "standard normal" else torch.rand(n, generator=g) * 10 - 5
    return x.to(DEV)


def cross(b, x, y_other, f):
    """A separate Route B checker on another binary's outputs.

    Returns the flags on certified outputs (the table's own verdict) and the flags of the second pass
    (a duplicate by our binary, so it disagrees wherever the two binaries do).
    """
    n = x.numel()
    b.run("rb_list", x, y_other, fused=False, faults=f)
    flag, ok = f.flag[:n] > 0, f.ok[:n] > 0
    return int((flag & ok).sum()), int((flag & ~ok).sum())


def exp_clean(args):
    """Clean data: how often is the kernel not correctly rounded, and does any checker raise a false alarm?"""
    n = 1 << (13 if args.quick else 22)
    b = gpu.Bench(n)
    for name in ("standard normal", "uniform [-5, 5)"):
        x, y = draw(name, n), half(n)
        b.run("none", x, y)
        ref, _, _ = fp16_cell(gelu(x.cpu().numpy().astype(np.float64)))
        ncr = int((ref != y.cpu().numpy().view(np.uint16)).sum())
        print(f"\n{name}, n = {n:,}: kernel outputs not correctly rounded {ncr} ({ncr / n:.2e})")
        print("  check      | fused: false alarms | separate checker: false alarms | note")
        for v in CHECKED:
            b.run(v, x, half(n))
            c = b.counts()
            sep = "-"
            if v in SEPARATE:
                b.run(v, x, y, fused=False)
                sep = str(b.counts()["bad"])
            note = ""
            if v in ("rb_list", "rb_sum"):
                note = f"certified {100 * (1 - c['uncertified'] / n):.2f}%, second pass on {c['uncertified']}"
            if v == "rb_block":
                nb = triton.cdiv(n, b.block)
                note = f"blocks recomputed {c['uncertified']} of {nb} ({100 * c['uncertified'] / nb:.1f}%)"
            print(f"  {v:10s} | {c['bad']:19d} | {sep:>30s} | {note}")
        yt = F.gelu(x).half()
        diff = int((b16(yt) != b16(y)).sum())
        print(f"  PyTorch F.gelu output as y (another binary of the same formula): differs from ours on {diff}")
        b.run("dup", x, yt, fused=False)
        print(f"    separate dup flags {b.counts()['bad']} (a duplicate matches only its own binary)")
        on_cert, second = cross(b, x, yt, b.faults(n))
        print(f"    separate Route B flags {on_cert} certified outputs (must be 0: the table accepts every kernel "
              f"inside the bound)\n      and {second} in the second pass (that pass is a duplicate by our binary)")


def fp32_spans(quick):
    """Bit patterns of every FP32 value in [-5, 5), in chunks."""
    top = 0x40A00000                                 # 5.0
    size = 1 << (12 if quick else 25)
    for lo, hi in ((0, top), (0x80000000, 0x80000000 + top + 1)):
        starts = range(lo, hi, size)
        if quick:
            starts = [lo, lo + (hi - lo) // 2 // size * size, hi - size]
        for s in starts:
            yield s, min(s + size, hi)


def exp_exhaustive(args):
    """Every FP32 input in [-5, 5): the error bound and the false alarms, with no sampling."""
    b = f = None
    kern = torch.tensor(gpu.route_b.build().kern32, device=DEV).double()
    tot = dict(n=0, unc=0, tri_out=0, ratio=0.0, cross=0, cross2=0, crossdup=0, **{v: 0 for v in CHECKED})
    for s, e in fp32_spans(args.quick):
        t = torch.arange(s, e, dtype=torch.int64, device=DEV)
        x = torch.where(t >= 2**31, t - 2**32, t).to(torch.int32).view(torch.float32)
        n = x.numel()
        if b is None:
            b = gpu.Bench(1 << (12 if args.quick else 25))
            f = b.faults(b.n)
        y = half(n)
        for v in CHECKED:
            b.run(v, x, y)
            c = b.counts()
            tot[v] += c["bad"]
            if v == "rb_list":
                tot["unc"] += c["uncertified"]
        x64 = x.double()
        g = 0.5 * x64 * torch.special.erfc(-x64 * C)
        k = kern[torch.clamp(torch.floor((x64 + 5) * 16), 0, 159).long()]
        yt32 = F.gelu(x)
        tot["ratio"] = max(tot["ratio"], float(((yt32.double() - g).abs() / k).max()))
        y64 = y.double()
        tot["tri_out"] += int(((y64 < (g - k).half().double()) | (y64 > (g + k).half().double())).sum())
        yt = yt32.half()
        on_cert, second = cross(b, x, yt, f)
        tot["cross"] += on_cert
        tot["cross2"] += second
        b.run("dup", x, yt, fused=False)
        tot["crossdup"] += b.counts()["bad"]
        tot["n"] += n
    n = tot["n"]
    print(f"\n{'SAMPLE of the' if args.quick else 'every'} FP32 input in [-5, 5): {n:,} bit patterns "
          f"(+0 and -0 both count)")
    print(f"  PyTorch F.gelu (FP32): max error / stored bound = {tot['ratio']:.3f} (must be <= 1)")
    print(f"  our kernel: FP16 outputs outside the bound = {tot['tri_out']} (must be 0)")
    print(f"  Route B certifies {100 * (1 - tot['unc'] / n):.2f}% of these values (a count of values, not of real inputs)")
    for v in CHECKED:
        print(f"  fused {v:8s} false alarms {tot[v]}")
    print(f"  separate dup on PyTorch's outputs: flags {tot['crossdup']} (a duplicate matches only its own binary)")
    print(f"  separate Route B on PyTorch's outputs: flags {tot['cross']} certified outputs (must be 0) and "
          f"{tot['cross2']} in the second pass (a duplicate by our binary)")


def exp_inject(args):
    """One single-bit fault per block of 1024. Detection is counted over faults that changed the stored output."""
    n = 1 << (13 if args.quick else 20)
    rounds = 2 if args.quick else 32
    b = gpu.Bench(n)
    nb = n // b.block
    x, y0, y = draw("standard normal", n, 1), half(n), half(n)
    b.run("none", x, y0)
    base = torch.arange(nb, device=DEV) * b.block
    groups = (("fraction bits 0-9", 0, 10), ("fraction bits 10-22", 10, 23), ("exponent bits 23-30", 23, 31),
              ("sign bit 31", 31, 32))

    def blocks_caught(v, f, pos):
        hit = f.flag[pos] > 0
        return hit | (f.d != 0) if v == "rb_sum" else hit

    print(f"\n{rounds * nb:,} faults per site, standard-normal inputs, n = {n:,} per launch")
    sites = (("FP32 erf result, first computation only", "erf", False),
             ("FP32 GELU output, first computation only", "out", False),
             ("FP32 erf result, every computation (defective unit)", "erf", True))
    for label, site, persist in sites:
        caught = {v: 0 for v in CHECKED}
        spurious = {v: 0 for v in CHECKED}
        eff_total, cert_eff = 0, 0
        per_bit = np.zeros((32, 3), np.int64)        # injected, effective, caught by the range check
        for r in range(rounds):
            g = torch.Generator().manual_seed(100 + r)
            pos = base + torch.randint(0, b.block, (nb,), generator=g).to(DEV)
            bit = torch.randint(0, 32, (nb,), generator=g)
            mask = torch.tensor((1 << bit.numpy().astype(np.int64)).astype(np.uint32).view(np.int32), device=DEV)
            for v in CHECKED:
                f = b.faults(n, persist)
                (f.fe if site == "erf" else f.fo)[pos] = mask
                b.run(v, x, y, faults=f)
                eff = b16(y)[pos] != b16(y0)[pos]
                hit = blocks_caught(v, f, pos)
                caught[v] += int((hit & eff).sum())
                spurious[v] += int((hit & ~eff).sum())
                if v == "rb_list":
                    cert_eff += int((eff & (f.ok[pos] > 0)).sum())
                if v == "range":
                    for k, col in ((torch.ones_like(eff), 0), (eff, 1), (hit & eff, 2)):
                        per_bit[:, col] += np.bincount(bit.numpy()[k.cpu().numpy()], minlength=32)
            eff_total += int(eff.sum())
        print(f"\n  site: {label}")
        print(f"    faults that changed the stored FP16 output: {eff_total} of {rounds * nb} "
              f"({100 * eff_total / (rounds * nb):.1f}%); {100 * cert_eff / max(eff_total, 1):.2f}% of them on certified outputs")
        for v in CHECKED:
            print(f"    {v:8s} caught {caught[v]:6d} = {100 * caught[v] / max(eff_total, 1):6.2f}%"
                  f"   flagged with no output change: {spurious[v]}")
        if not persist:
            for name, lo, hi in groups:
                inj, eff_g, rng_g = per_bit[lo:hi].sum(0)
                print(f"    {name:20s}: changed the output {100 * eff_g / max(inj, 1):5.1f}%; "
                      f"range check caught {100 * rng_g / max(eff_g, 1):5.1f}% of those")

    print("\n  site: stored FP16 output (one bit flipped in memory after the kernel)")
    print("    a fused check reads the register, so it cannot see this fault: 0% by construction")
    caught = {v: 0 for v in SEPARATE}
    for r in range(rounds):
        g = torch.Generator().manual_seed(500 + r)
        pos = base + torch.randint(0, b.block, (nb,), generator=g).to(DEV)
        yf = y0.clone()
        b16(yf)[pos] ^= (1 << torch.randint(0, 16, (nb,), generator=g)).to(torch.int16).to(DEV)
        for v in SEPARATE:
            f = b.faults(n)
            b.run(v, x, yf, fused=False, faults=f)
            caught[v] += int(blocks_caught(v, f, pos).sum())
    for v in SEPARATE:
        print(f"    separate {v:8s} caught {caught[v]:6d} of {rounds * nb} = {100 * caught[v] / (rounds * nb):6.2f}%")

    print("\n  site: two faults in one block, FP32 GELU output, first computation only (checksum cancellation)")
    blocks = missed = 0
    for r in range(rounds):
        g = torch.Generator().manual_seed(900 + r)
        off = torch.stack([torch.randperm(b.block, generator=g)[:2] for _ in range(nb)]).to(DEV)
        pos = (base[:, None] + off).reshape(-1)
        f = b.faults(n)
        f.fo[pos] = (1 << torch.randint(13, 31, (2 * nb,), generator=g)).to(torch.int32).to(DEV)
        b.run("rb_sum", x, y, faults=f)
        eff = ((b16(y)[pos] != b16(y0)[pos]) & (f.ok[pos] > 0)).reshape(nb, 2).all(1)   # both wrong and certified
        blocks += int(eff.sum())
        missed += int((eff & (f.d == 0)).sum())
    print(f"    rb_sum: blocks with two wrong certified outputs {blocks}; checksum missed {missed} "
          f"(theory: about {blocks / 65537:.2f})")


def exp_sm_fault(args):
    """A defective SM: every computation on one SM gets the fault, checks that run there included."""
    n = 1 << (13 if args.quick else 20)
    b = gpu.Bench(n)
    nb = n // b.block
    x, y0, y = draw("standard normal", n, 4), half(n), half(n)
    b.run("none", x, y0)
    p1, p2 = b.faults(n), b.faults(n)
    b.run("none", x, y, faults=p1)
    b.run("none", x, y, faults=p2)
    s1, s2 = p1.sm.cpu().numpy(), p2.sm.cpu().numpy()
    sms = np.unique(s1)
    per_sm = np.bincount(s1)[sms]
    print(f"\n  n = {n:,} ({nb} blocks of {b.block}); the kernel ran on {len(sms)} SMs, "
          f"{per_sm.min()}-{per_sm.max()} blocks each")
    print(f"  a block ran on the same SM in two launches: {100 * np.mean(s1 == s2):.1f}% of blocks")
    chosen = np.random.default_rng(11).choice(sms, size=min(8, len(sms)), replace=False)
    rounds = 1 if args.quick else 4
    print(f"  defective SMs tried: {', '.join(str(int(k)) for k in chosen)}; {rounds} rounds each; one random")
    print("  bit of the FP32 erf result flipped in one element per block, active only where that SM computes.")
    print("  A wrong block: its fault changed the stored output. Caught: any alarm in the block. An alarm on a")
    print("  correct block comes from a check that itself ran on the defective SM.")
    base = torch.arange(nb, device=DEV) * b.block
    fused = ("dup", "rb_list", "rb_sum")
    separate = (("dup", 0), ("dup", nb // 2), ("rb_list", 0))
    names = [f"fused {v}" for v in fused] + [f"separate {v}" + (", shifted" if sh else "") for v, sh in separate]
    tot = {k: np.zeros(4, np.int64) for k in names}  # wrong blocks, caught, alarms on correct blocks, same SM

    def tally(name, f, y_out, check_sm):
        eff = b16(y_out)[pos] != b16(y0)[pos]
        alarm = (f.flag.reshape(nb, b.block) > 0).any(1) | (f.d != 0)
        same = (check_sm == bad) & eff if check_sm is not None else torch.zeros_like(eff)
        tot[name] += [int(eff.sum()), int((eff & alarm).sum()), int((~eff & alarm).sum()), int(same.sum())]

    for bad in (int(k) for k in chosen):
        for r in range(rounds):
            g = torch.Generator().manual_seed(1000 + r)
            pos = base + torch.randint(0, b.block, (nb,), generator=g).to(DEV)
            bit = torch.randint(0, 32, (nb,), generator=g)
            mask = torch.tensor((1 << bit.numpy().astype(np.int64)).astype(np.uint32).view(np.int32), device=DEV)

            def faults():
                f = b.faults(n, bad_sm=bad)
                f.fe[pos] = mask
                return f

            for v, name in zip(fused, names[:3]):
                f = faults()
                b.run(v, x, y, faults=f)
                tally(name, f, y, None)
            for (v, sh), name in zip(separate, names[3:]):
                f = faults()
                b.run("none", x, y, faults=f)       # the faulty kernel
                f2 = faults()
                b.run(v, x, y, fused=False, faults=f2, shift=sh)
                tally(name, f2, y, f2.sm2 if v == "dup" else None)
    print(f"\n  {'check':24s} {'wrong blocks':>12s} {'caught':>16s} {'alarms on correct blocks':>25s} "
          f"{'check ran on the bad SM too':>28s}")
    for name in names:
        w, c, a, sm = tot[name]
        same = f"{sm} ({100 * sm / max(w, 1):.1f}%)" if name.startswith("separate dup") else "-"
        print(f"  {name:24s} {w:12d} {c:7d} = {100 * c / max(w, 1):6.2f}% {a:25d} {same:>28s}")


class Time(NamedTuple):
    """Kernel time in ms: the median over the repeats, and the smallest and largest repeat."""
    med: float
    lo: float
    hi: float

    def us(self):
        return f"{self.med * 1e3:9.1f} us ({self.lo * 1e3:.1f}-{self.hi * 1e3:.1f})"


def bench_all(fns, args):
    """Time each function args.repeat times, in turns, so that a burst of load from another job on the GPU
    falls on one repeat of every function instead of on all repeats of one. Each repeat is a do_bench mean
    over about 200 ms. Every function must already be compiled (called once)."""
    if args.quick:
        for fn in fns.values():
            fn()
        return {k: Time(float("nan"), float("nan"), float("nan")) for k in fns}
    s = {k: [] for k in fns}
    for _ in range(args.repeat):
        for k, fn in fns.items():
            s[k].append(triton.testing.do_bench(fn, warmup=25, rep=200))
    return {k: Time(float(np.median(v)), min(v), max(v)) for k, v in s.items()}


def exp_timing(args):
    """Standalone kernels, FP32 in and FP16 out. Sizes are tokens times the 4096-wide MLP of Pythia-410M."""
    print("\nEvery checked variant includes the reset of its 8-byte counters (one tiny launch).")
    print("rb_list and rb_sum include the second pass. A 'separate' row is the time of the check without the GELU")
    print("kernel: the checker kernel, its counter reset and, for rb_list and rb_sum, the second pass.")
    print(f"Each time: median of {args.repeat} repeats taken in turns (smallest-largest in brackets).")
    for tokens in (16,) if args.quick else (16, 512, 4096):
        n = tokens * 4096
        b = gpu.Bench(n)
        x, y = draw("standard normal", n, 2), half(n)
        fns = {"none": lambda: b.run("none", x, y)}
        for v in CHECKED:
            fns[v] = lambda v=v: b.run(v, x, y)
        for v in SEPARATE:
            fns["separate " + v] = lambda v=v: b.run(v, x, y, fused=False)
        fns["separate dup, shifted"] = lambda: b.run("dup", x, y, fused=False, shift=triton.cdiv(n, b.block) // 2)
        fns["torch F.gelu, FP32 out"] = lambda: F.gelu(x)
        for fn in fns.values():                      # compiles every kernel; y ends as the clean output
            fn()
        b.run("none", x, y)
        t = bench_all(fns, args)
        base = t["none"].med
        print(f"\n  {tokens} tokens ({n:,} elements); unprotected GELU {base * 1e3:.1f} us "
              f"= {6 * n / (base * 1e-3) / 1e9:.0f} GB/s")
        for name, v in t.items():
            extra = f"   none + checker x{(base + v.med) / base:.2f}" if name.startswith("separate") else ""
            print(f"    {name:24s} {v.us()}   x{v.med / base:.2f} of unprotected{extra}")


def exp_matmul(args):
    """C = GELU(A @ B), A (M x 1024) and B (1024 x 4096) in FP16: the Pythia-410M MLP up-projection."""
    K, N = (64, 128) if args.quick else (1024, 4096)
    for M in (40,) if args.quick else (16, 512, 2048):
        g = torch.Generator().manual_seed(3)
        a = (torch.randn(M, K, generator=g) / K**0.5).half().to(DEV)     # keeps GELU inputs near N(0, 1)
        w = torch.randn(K, N, generator=g).half().to(DEV)
        c = torch.empty(M, N, dtype=torch.float16, device=DEV)
        b = gpu.Bench(M * N)
        print(f"\n  M = {M} tokens")
        fns, note = {}, {}
        for v in gpu.MM_VARIANTS:
            try:
                b.mm(v, a, w, c)
                if v not in ("matmul", "none"):      # these two have no check, so no counters
                    cnt = b.counts()
                    note[v] = f"   false alarms {cnt['bad']}, second pass / recomputed {cnt['uncertified']}"
                fns[v] = lambda v=v: b.mm(v, a, w, c)
            except Exception as err:                 # one variant that does not compile must not stop the rest
                note[v] = f"failed: {type(err).__name__}: {str(err)[:200]}"
        t = bench_all(fns, args)
        base = t["none"].med if "none" in t else float("nan")
        for v in gpu.MM_VARIANTS:
            if v in t:
                print(f"    {v:8s} {t[v].us()}   x{t[v].med / base:.3f} of matmul + GELU{note.get(v, '')}")
            else:
                print(f"    {v:8s} {note[v]}")


def exp_cost(args):
    """Where Route B's time goes: fused GELU plus one more part of rb_list at each step."""
    labels = {"none": "GELU and store (unprotected)",
              "count": "+ alarm counter (reset, 1 atomic per block)",
              "predict": "+ table lookup and cubic, compare all",
              "cell": "+ margin and cell test, compare certified",
              "append": "+ append uncertified inputs (= rb_list kernel)",
              "list": "+ second pass over the list (= rb_list)"}
    print("\nEach step adds one part to the step before and uses it in a real check; no code is added only to")
    print("keep work alive. A step's cost is its median minus the median of the step before; the range in")
    print("brackets is the widest it can be from the two steps' smallest and largest repeats. 'regs' is the")
    print("registers per thread of the main kernel (fewer blocks fit on an SM as it grows).")
    print(f"Each time: median of {args.repeat} repeats taken in turns (smallest-largest in brackets).")
    for tokens in (16,) if args.quick else (512, 4096):
        n = tokens * 4096
        b = gpu.Bench(n)
        x, y = draw("standard normal", n, 2), half(n)
        fns, alarms, regs = {}, {}, {}
        for s in gpu.COST_STAGES:
            k = b.cost(s, x, y)                      # compiles; and gives the alarms of a clean run
            alarms[s] = b.counts()["bad"] if s != "none" else 0
            regs[s] = getattr(k, "n_regs", None)
            fns[s] = lambda s=s: b.cost(s, x, y)
        for v in ("dup", "rb_list"):
            regs[v] = getattr(b.run(v, x, y), "n_regs", None)
            fns[v] = lambda v=v: b.run(v, x, y)
        t = bench_all(fns, args)
        base = t["none"].med
        print(f"\n  {tokens} tokens ({n:,} elements); alarms on clean data: 'predict' {alarms['predict']} (outputs "
              f"it cannot certify), all other steps {sum(alarms.values()) - alarms['predict']}")
        print(f"    {'step':48s} {'regs':>4s} {'time':>26s}   {'step cost (range)':>28s} {'of unprotected':>14s}")
        prev = None
        for s in gpu.COST_STAGES:
            step = ""
            if prev is not None:
                d, lo, hi = t[s].med - t[prev].med, t[s].lo - t[prev].hi, t[s].hi - t[prev].lo
                step = f"{d * 1e3:+8.1f} us ({lo * 1e3:+.1f}..{hi * 1e3:+.1f})   {d / base:+10.2f}"
            print(f"    {labels[s]:48s} {str(regs[s]):>4s} {t[s].us():>26s}   {step}")
            prev = s
        print(f"    k_std for comparison: dup {t['dup'].us()}, {regs['dup']} regs; "
              f"rb_list {t['rb_list'].us()}, {regs['rb_list']} regs")


def capture(model, call):
    """GELU inputs and outputs of every GELU module during call()."""
    xs, ys = [], []

    def hook(mod, inp, out):                         # must return None: a returned value replaces the output
        xs.append(inp[0].detach().flatten())
        ys.append(out.detach().flatten())

    hooks = [m.register_forward_hook(hook) for m in model.modules() if "gelu" in type(m).__name__.lower()]
    with torch.no_grad():
        call()
    for h in hooks:
        h.remove()
    if not xs:
        raise RuntimeError("no GELU module was called")
    return torch.cat(xs), torch.cat(ys), len(xs)


def exp_models(args):
    """GELU inputs captured from real FP16 models on real text (this repo's README)."""
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
    text = (pathlib.Path(__file__).resolve().parents[1] / "README.md").read_text()
    words = text.split()

    def report(name, x16, y_model, calls):
        x = x16.float().contiguous()                 # exact: every FP16 value is an FP32 value
        n = x.numel()
        b = gpu.Bench(n)
        y = half(n)
        out = float(((x < -5) | (x >= 5) | x.isnan()).float().mean())
        print(f"\n  {name}: {calls} GELU calls, {n:,} inputs; outside [-5, 5): {100 * out:.3f}%; "
              f"|x| mean {float(x.abs().mean()):.2f}, max {float(x.abs().max()):.1f}")
        for v in ("dup", "rb_list", "rb_sum"):
            b.run(v, x, y)
            c = b.counts()
            cert = f", certified {100 * (1 - c['uncertified'] / n):.2f}%" if v != "dup" else ""
            print(f"    fused {v:8s} false alarms {c['bad']}{cert}")
        ym = y_model.contiguous()
        print(f"    the model's own FP16 GELU outputs differ from our kernel on {int((b16(ym) != b16(y)).sum())}")
        b.run("dup", x, ym, fused=False)
        print(f"    separate dup on the model's own outputs flags {b.counts()['bad']}")
        on_cert, second = cross(b, x, ym, b.faults(n))
        print(f"    separate Route B on the model's own outputs flags {on_cert} certified outputs and {second} "
              f"in the second pass (a duplicate by our binary)")

    tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-410m")
    lm = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-410m", torch_dtype=torch.float16).to(DEV).eval()
    ids = tok(text, return_tensors="pt", truncation=True, max_length=512).input_ids.to(DEV)
    report(f"Pythia-410M, 1 x {ids.shape[1]} tokens", *capture(lm, lambda: lm(ids)))
    lm = None     # frees the GPU memory before BERT loads
    tokb = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
    bert = AutoModel.from_pretrained("google-bert/bert-base-uncased", torch_dtype=torch.float16).to(DEV).eval()
    step = max(1, len(words) // 32)
    batch = tokb([" ".join(words[i * step:(i + 1) * step]) for i in range(32)], return_tensors="pt", padding=True,
                 truncation=True, max_length=128).to(DEV)
    report(f"BERT-base, {batch.input_ids.shape[0]} x {batch.input_ids.shape[1]} tokens",
           *capture(bert, lambda: bert(**batch)))


def exp_share(args):
    """Share of GPU kernel time spent in GELU kernels, eager FP16 PyTorch."""
    from torch.profiler import ProfilerActivity, profile
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

    def share(fn):
        fn()
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            fn()
            torch.cuda.synchronize()
        kern = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
        total = sum(e.time_range.elapsed_us() for e in kern)
        g = sum(e.time_range.elapsed_us() for e in kern if "gelu" in e.name.lower())
        if g == 0:
            print("    (no kernel named *gelu* found; names seen:", sorted({e.name[:60] for e in kern})[:8], ")")
        return 100 * g / total if total else float("nan")

    with torch.no_grad():
        tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-410m")
        lm = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-410m", torch_dtype=torch.float16).to(DEV).eval()
        ids = torch.randint(0, tok.vocab_size, (1, 512), device=DEV)
        print(f"\n  Pythia-410M prefill, 1 x 512 tokens:                GELU {share(lambda: lm(ids)):.2f}% of GPU time")
        dec = lambda: lm.generate(ids[:, :32], max_new_tokens=64, do_sample=False, pad_token_id=0)
        print(f"  Pythia-410M, 32-token prompt + 64 generated tokens: GELU {share(dec):.2f}% of GPU time")
        lm = None     # frees the GPU memory before BERT loads
        bert = AutoModel.from_pretrained("google-bert/bert-base-uncased", torch_dtype=torch.float16).to(DEV).eval()
        bids = torch.randint(0, 30000, (32, 128), device=DEV)
        print(f"  BERT-base, 32 x 128 tokens:                         GELU {share(lambda: bert(bids)):.2f}% of GPU time")


EXPERIMENTS = {"clean": exp_clean, "exhaustive": exp_exhaustive, "inject": exp_inject, "sm_fault": exp_sm_fault,
               "timing": exp_timing,
               "matmul": exp_matmul, "cost": exp_cost, "models": exp_models, "share": exp_share}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=",".join(EXPERIMENTS), help="comma-separated experiments")
    ap.add_argument("--quick", action="store_true", help="tiny sizes and no timing: a test of the script itself")
    ap.add_argument("--repeat", type=int, default=5, help="timing repeats per kernel; the median is reported")
    args = ap.parse_args()
    if args.repeat < 1:
        ap.error("--repeat must be at least 1")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU interpreter (no real timing)"
    print(f"torch {torch.__version__} | triton {triton.__version__} | {gpu_name}{' | QUICK' if args.quick else ''}")
    failed = []
    for name in args.only.split(","):
        print(f"\n== {name}: {EXPERIMENTS[name].__doc__.strip().splitlines()[0]}")
        try:
            EXPERIMENTS[name](args)
        except Exception:
            failed.append(name)
            print(f"  FAILED:\n{traceback.format_exc(limit=4)}")
    if failed:
        print(f"\nFAILED experiments: {', '.join(failed)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
