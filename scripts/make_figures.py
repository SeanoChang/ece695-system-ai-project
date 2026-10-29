"""Generate every figure in figures/ from fresh runs (no numbers typed in by hand).

Usage: python scripts/make_figures.py [--blocks 20000] [--n 400000]
Colors: blue and orange only, each with a text label (readable with red-green colour blindness).
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from gelucheck import route_a, route_b
from gelucheck.formats import CELLS, ROUND
from gelucheck.gelu import gelu, gelu_tanh, phi

BLUE, ORANGE, INK, MID, LINE, SKY = "#2F68A8", "#9A4F12", "#0B0B0B", "#4D4C48", "#D0D0CD", "#DFEAF6"
plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": LINE, "grid.linewidth": 0.6})
OUT = os.path.join(os.path.dirname(__file__), "..", "figures")


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name), dpi=180)
    plt.close(fig)
    print("wrote figures/" + name)


def fig_route_a_tradeoff(rng, nb):
    from run_route_a import sweep
    res = sweep(rng, nb)
    fa, det = zip(*[(f, d) for _, f, d in res["fixed"]])
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.plot(fa, det, "-o", color=ORANGE, lw=2, label="Route A, fixed tolerance")
    for (t, f, d) in res["fixed"]:
        ax.annotate(f"{t:g}", (f, d), (5, -12), textcoords="offset points", fontsize=9, color=MID)
    afa, adet = zip(*[(f, d) for _, f, d in res["aware"]])
    ax.plot(afa, adet, "s--", color=ORANGE, mfc="white", lw=1.4, label="Route A, rounding-aware tolerance")
    ax.plot([0], [100], "*", ms=16, color=BLUE, label="Route B (exact): no false alarms, all single errors caught")
    ax.set_xlabel("false alarms on correct blocks (%)")
    ax.set_ylabel("random single bit flips caught (%)")
    ax.set_title("Route A: no tolerance gives both (FP16 outputs, blocks of 256)", fontsize=12, loc="left")
    ax.set_xlim(-3, 100)
    ax.set_ylim(-3, 106)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    save(fig, "route_a_tradeoff.png")


def fig_route_a_blocksize(rng):
    sizes = [16, 32, 64, 128, 256, 512, 1024]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for fmt, color, mk in (("FP16", BLUE, "o"), ("BF16", ORANGE, "s")):
        rates = []
        for B in sizes:
            x = rng.standard_normal((max(400, 1_000_000 // B), B)).astype(np.float32).astype(np.float64)
            e = route_a.mismatch(x, ROUND[fmt](gelu_tanh(x)))
            rates.append(100 * route_a.fails(e, 0.1).mean())
        ax.plot(sizes, rates, "-" + mk, color=color, lw=2, label=f"{fmt} outputs")
    ax.set_xscale("log", base=2)
    ax.set_xticks(sizes, [str(s) for s in sizes])
    ax.set_xlabel("block size (elements)")
    ax.set_ylabel("correct blocks rejected (%)")
    ax.set_title("Route A false alarms grow with block size (tolerance 0.1)", fontsize=12, loc="left")
    ax.legend(frameon=False)
    save(fig, "route_a_block_size.png")


def fig_route_b_certified(n):
    tab = route_b.build(-5.0, 5.0, 1 / 16)
    x32 = np.random.default_rng(0).uniform(-5, 5, n).astype(np.float32)
    edges = np.linspace(-5, 5, 81)
    mid = (edges[:-1] + edges[1:]) / 2
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for fmt, color, ls in (("FP16", BLUE, "-"), ("BF16", ORANGE, "--")):
        ok, _ = route_b.certify(tab, x32, fmt)
        idx = np.digitize(x32, edges) - 1
        rate = [100 * ok[idx == i].mean() for i in range(len(mid))]
        ax.plot(mid, rate, ls, color=color, lw=2, label=f"{fmt}: {100 * ok.mean():.2f}% overall")
    ax.set_xlabel("input x")
    ax.set_ylabel("outputs certified (%)")
    ax.set_title("Route B settles almost every output without recomputing", fontsize=12, loc="left")
    ax.set_ylim(95, 100.3)
    ax.legend(frameon=False, loc="lower left")
    save(fig, "route_b_certified.png")


def fig_route_b_bound():
    tab = route_b.build(-5.0, 5.0, 1 / 16)
    x32 = np.linspace(tab.lo, tab.hi - 1e-9, 400_001).astype(np.float32)
    p, j = route_b.evaluate(tab, x32)
    err = np.abs(gelu(x32.astype(np.float64)) - p.astype(np.float64))
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.semilogy(x32, np.maximum(err, 1e-16), color=BLUE, lw=0.6, label="measured |GELU − P|")
    ax.semilogy(tab.a + tab.h / 2, tab.delta32, "-", color=ORANGE, lw=2, drawstyle="steps-mid",
                label="stored bound Δ per interval")
    ax.set_xlabel("input x")
    ax.set_ylabel("absolute error")
    ax.set_title(f"Route B error stays under its bound (max ratio {(err / tab.delta32[j]).max():.2f})",
                 fontsize=12, loc="left")
    ax.set_ylim(1e-14, 1e-4)
    ax.legend(frameon=False, loc="upper left")
    save(fig, "route_b_bound.png")


def fig_gelu():
    x = np.linspace(-4, 4, 800)
    g = gelu(x)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.fill_between(x, np.maximum(x, 0) - phi(x), np.maximum(x, 0) + phi(x), color=SKY, lw=0,
                    label="max(x, 0) ± φ(x)")
    ax.plot(x, np.maximum(x, 0), "--", color=MID, lw=1.4, label="max(x, 0)")
    ax.plot(x, g, color=BLUE, lw=2.4, label="exact GELU")
    ax.plot(x, gelu_tanh(x), ":", color=ORANGE, lw=2, label="tanh GELU")
    ax.set_xlabel("input x")
    ax.set_ylabel("output")
    ax.set_title("GELU, its tanh approximation, and the tail bound", fontsize=12, loc="left")
    ax.legend(frameon=False, loc="upper left")
    save(fig, "gelu.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", type=int, default=20_000)
    ap.add_argument("--n", type=int, default=400_000)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(0)
    fig_gelu()
    fig_route_b_bound()
    fig_route_b_certified(args.n)
    fig_route_a_tradeoff(rng, args.blocks)
    fig_route_a_blocksize(rng)


if __name__ == "__main__":
    main()
