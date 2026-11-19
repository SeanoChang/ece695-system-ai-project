"""GPU probe: how accurate is the real GELU kernel, and does Route B's exact match survive it?

Measures, on real GPU arithmetic:
  1. the error of CUDA erf / erfc in FP32 ulps;
  2. how often each GELU kernel's FP16 output is not correctly rounded;
  3. whether the kernels agree with each other bit for bit (PyTorch, formula, Triton, erfc form);
  4. Route B: certified outputs that differ from each kernel, for the two targets of route_b.certify;
  5. whether Triton's erf and PyTorch's erf return the same FP32 bits.

Usage: CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. .venv/bin/python scripts/gpu_probe.py     (about 1 minute)
       PYTHONPATH=. python scripts/gpu_probe.py --emulate                            (CPU stand-in, no GPU)
"""
import argparse
import math

import numpy as np

from gelucheck import route_b
from gelucheck.formats import fp16_cell
from gelucheck.gelu import gelu

F32, C = np.float32, 0.7071067811865476
_erf, _erfc = np.frompyfunc(math.erf, 1, 1), np.frompyfunc(math.erfc, 1, 1)


def emulate(x):
    """CPU stand-in: erf and erfc correctly rounded to FP32, then the kernel's FP32 operations."""
    z = (x * F32(C)).astype(F32)
    e = _erf(z.astype(np.float64)).astype(np.float64).astype(F32)
    ec = _erfc(-z.astype(np.float64)).astype(np.float64).astype(F32)
    y = (F32(0.5) * x * (F32(1) + e)).astype(F32)
    return {"erf": e, "erfc": ec, "torch formula (erf)": y, "torch F.gelu": y,
            "torch formula (erfc)": (F32(0.5) * x * ec).astype(F32)}


def on_gpu(x):
    import torch
    import torch.nn.functional as F
    t = torch.from_numpy(x).cuda()
    z = t * C
    e, ec = torch.erf(z), torch.special.erfc(-z)
    out = {"erf": e, "erfc": ec, "torch formula (erf)": 0.5 * t * (1.0 + e), "torch F.gelu": F.gelu(t),
           "torch formula (erfc)": 0.5 * t * ec}
    try:
        import triton
        import triton.language as tl
        try:
            from triton.language.extra import libdevice
        except ImportError:
            from triton.language.extra.cuda import libdevice

        @triton.jit
        def k(x_ptr, y_ptr, y2_ptr, e_ptr, n, BLOCK: tl.constexpr):
            offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
            m = offs < n
            v = tl.load(x_ptr + offs, mask=m, other=1.0)
            tl.store(y_ptr + offs, 0.5 * v * (1.0 + tl.erf(v * 0.7071067811865476)), mask=m)
            tl.store(y2_ptr + offs, 0.5 * v * libdevice.erfc(-v * 0.7071067811865476), mask=m)
            tl.store(e_ptr + offs, tl.erf(v * 0.7071067811865476), mask=m)

        y, y2, e2 = torch.empty_like(t), torch.empty_like(t), torch.empty_like(t)
        k[(triton.cdiv(t.numel(), 1024),)](t, y, y2, e2, t.numel(), BLOCK=1024)
        out["triton (erf)"], out["triton (erfc)"], out["triton erf"] = y, y2, e2
    except Exception as err:                      # report and keep the PyTorch measurements
        print(f"  Triton kernel failed: {type(err).__name__}: {str(err)[:300]}")
    return {name: v.float().cpu().numpy() for name, v in out.items()}


def bits16(y32):
    return y32.astype(np.float16).view(np.uint16)


def report(name, x, out, tab):
    x64 = x.astype(np.float64)
    z64 = (x * F32(C)).astype(F32).astype(np.float64)
    g = gelu(x64)
    ref, _, _ = fp16_cell(g)
    print(f"\n=== {name} inputs, n = {x.size:,}")
    for fn, ref_fn, arg in (("erf", _erf, z64), ("erfc", _erfc, -z64)):
        r = ref_fn(arg).astype(np.float64)
        ulp = np.abs(out[fn].astype(np.float64) - r) / np.spacing(np.abs(r).astype(F32)).astype(np.float64)
        print(f"{fn:5s} error in FP32 ulps: max {ulp.max():.2f}; above 0.5: {100 * (ulp > 0.5).mean():.2f}%; "
              f"above 1: {100 * (ulp > 1).mean():.3f}%; above 2: {int((ulp > 2).sum())}")
    if "triton erf" in out:
        de = out["triton erf"] != out["erf"]
        ulps = np.abs(out["triton erf"].astype(np.float64) - out["erf"]) / np.spacing(np.abs(out["erf"]))
        print(f"triton erf vs torch erf: FP32 bits differ on {int(de.sum())} ({de.mean():.2e}); max {ulps.max():.0f} ulp")
    kernels = {k: v for k, v in out.items() if k not in ("erf", "erfc", "triton erf")}
    base = bits16(out["torch F.gelu"])
    _, j = route_b.evaluate(tab, x)
    ins = (x >= tab.lo) & (x < tab.hi)
    print("kernel                           | not correctly rounded | differs from torch F.gelu | max error / bound")
    for kname, y in kernels.items():
        b = bits16(y)
        ratio = (np.abs(y.astype(np.float64) - g)[ins] / tab.kern32[j][ins]).max()
        print(f"{kname:32s} | {int((b != ref).sum()):9d} ({(b != ref).mean():.2e}) | {int((b != base).sum()):9d} "
              f"({(b != base).mean():.2e}) | {ratio:6.2f}{'' if 'erfc' in kname else '  (erf form: must be <= 1)'}")
    print("Route B target                   | certified | certified but differs from: "
          + ", ".join(k for k in kernels if "erfc" not in k))
    for target in ("exact", "kernel"):
        ok, pb = route_b.certify(tab, x, "FP16", target)
        miss = ", ".join(str(int((ok & (pb != bits16(y))).sum())) for k, y in kernels.items() if "erfc" not in k)
        print(f"{target:32s} | {100 * ok.mean():8.2f}% | {miss}")
    ok, _ = route_b.certify(tab, x, "FP16", "kernel")
    dup = bits16(out["torch formula (erf)"]) != bits16(out["torch formula (erfc)"])
    print(f"erf form vs erfc form: differ on {int(dup.sum())} ({dup.mean():.2e}); "
          f"among uncertified inputs {100 * dup[~ok].mean():.2f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1 << 22)
    ap.add_argument("--emulate", action="store_true", help="CPU stand-in for the GPU (for testing this script)")
    args = ap.parse_args()
    run = emulate if args.emulate else on_gpu
    if not args.emulate:
        import torch
        print("torch", torch.__version__, "|", torch.cuda.get_device_name(0))
    rng = np.random.default_rng(0)
    tab = route_b.build(-5.0, 5.0, 1 / 16)
    for name, x in (("standard normal", rng.standard_normal(args.n).astype(F32)),
                    ("uniform [-5, 5]", rng.uniform(-5, 5, args.n).astype(F32))):
        report(name, x, run(x), tab)


if __name__ == "__main__":
    main()
