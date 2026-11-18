"""Checks for the claims in gelu_checksums_research_proposal.tex."""
import math, numpy as np
rng = np.random.default_rng(1)
k, c = 2 * math.sqrt(2 / math.pi), 0.044715
x = rng.standard_normal(64)
# (1) tanh form == sigmoid form
t = 0.5 * x * (1 + np.tanh(math.sqrt(2 / math.pi) * (x + c * x**3)))
s = x / (1 + np.exp(-k * (x + c * x**3)))
print("(1) max |tanh form - sigmoid form| =", np.abs(t - s).max())
# (2) batch identity in float64, via logs to avoid under/overflow
lhs = np.sum(np.log(np.abs(x - s))); rhs = -k * (x.sum() + c * (x**3).sum()) + np.sum(np.log(np.abs(s)))
print("(2) log|prod(x-y)| vs log|rhs|, float64:", lhs, rhs, "sign ok:", np.prod(np.sign(x - s)) == np.prod(np.sign(s)))
# (2b) same identity when y is stored in FP16 (what a kernel writes)
y16 = s.astype(np.float16).astype(np.float64)
d = x - y16
print("(2b) FP16 outputs: elements with x - y == 0:", int((d == 0).sum()), "of", x.size,
      "; inputs where that happens:", np.round(np.sort(x[d == 0]), 2)[:8])
big = np.array([3.0, 4.0, 5.0, 6.0])
print("     x - fp16(y) at x=3,4,5,6:", (big - (big / (1 + np.exp(-k * (big + c * big**3)))).astype(np.float16)).tolist())
# (3) |g(x) - max(x,0)| <= phi(x)
xs = np.linspace(-30, 30, 600001)
Phi = 0.5 * (1 + np.vectorize(math.erf)(xs / math.sqrt(2)))
phi = np.exp(-xs**2 / 2) / math.sqrt(2 * math.pi)
gap = np.abs(xs * Phi - np.maximum(xs, 0))
print("(3) max ratio |g - relu| / phi =", (gap[phi > 1e-300] / phi[phi > 1e-300]).max())
# (4) table sizes and speedup arithmetic
print("(4)", 40 * 16, "intervals,", 640 * 5 * 4, "bytes;", 160 * 5 * 4, "bytes for [-5,5]")
print("(5) 2T/1.2T =", 2 / 1.2, "; latency cut =", 1 - 1.2 / 2)
