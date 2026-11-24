# Known limits

Small defects and limits of scope that we know and have not fixed. None of them changes a reported result
unless the entry says so.

## Limits of the method

- **The kernel error bound depends on NVIDIA's tests.** `route_b.kernel_bound` assumes that `erff` is within
  2 ulp of the correctly rounded result. NVIDIA states this limit from "extensive but not exhaustive tests".
  The `exhaustive` experiment of `scripts/run_gpu.py` checks the bound for every FP32 input in [−5, 5), on the
  GPU and library in use only.
- **Certified outputs only.** The properties that separate Route B from duplication (it does not repeat a fault
  from a defective unit, and it accepts every kernel inside the bound) hold for certified outputs. The second
  pass for uncertified outputs is a duplicate by the same compiled code.
- **A fused check does not see a store fault.** A check inside the GELU kernel compares the value in the
  register. A fault in the store or in memory needs a separate checker (`fused=False`).
- **Sampled maxima.** `route_b.build` takes the maxima of `|g''''|`, `|g'|` and `|g|` on each interval from 401
  samples and adds 1%. This is not a proof. `build(samples=1)` gives a wrong table and is not rejected.
- **The table covers [−5, 5).** Inputs outside it, and inputs whose FP16 output is ±0, are never certified.
- **One multiplier vector.** Every block uses the same multipliers `r`. Two wrong outputs cancel with
  probability about 2⁻¹⁶ over the choice of `r`, not over the blocks of one run.

## Limits of the GPU experiments

- **The interpreter is not the compiler.** Triton's CPU interpreter has no fused multiply-add and its own `erf`.
  A test that passes there shows that the logic is right, not that the compiled kernel is.
- **Timing includes the counter reset.** Every checked variant resets its 8-byte counters with one small launch
  before the kernel. At 16 tokens this fixed cost is a visible part of the time.
- **The second pass uses a fixed grid of 40 programs** and reads the list length on the device. We did not tune
  the grid.
- **The duplicate reads its input once.** The second copy gets the same register value through an XOR with a
  run-time zero. A fault in the load of `x` reaches both copies.
- **Text for the model inputs.** The `models` experiment feeds this repository's README to Pythia-410M and BERT.
  It is one text of one kind.
- **The GPU tests take 18.5 minutes** on dlm (`results/pytest_gpu.txt`). The slowest tests spend their time in
  host-side NumPy on 2 million elements (`np.setdiff1d` and the bit copies in `tests/test_gpu.py:139-140`), not
  on the GPU. `results/pytest_gpu.txt` is copied from the terminal, with the repeated pytest frames of two
  stacks cut.
- **The cost ladder carries its keep-alive terms.** Stages `count`, `load` and `poly` of `k_cost` keep their
  work through a multiply by a run-time 0.0, a compare and an OR per element, and every later stage keeps
  them. Each step therefore includes one such term. The experiment prints their total as the last step
  minus `k_std` rb_list.
- **No test sees a missing second pass in the `list` cost stage.** On clean data the second pass changes no
  counter, so `Bench.cost("list")` without its `k_fallback` launch passes every test. The cost experiment
  prints the `list` stage next to `k_std` rb_list, which would show the gap.
- **The range check is slower than duplication** in the standalone kernel (×1.42 against ×1.36 at 4096 tokens).
  We did not look into why.
- **The range check flags correct outputs for extreme inputs:** NaN, −∞, and x ≥ 65520, where the FP16 output
  is +∞. No experiment gives such inputs to the range check, and FP16 activations cannot reach 65520.

## Route A

- **"Bit flips caught" is not conditioned** (`scripts/run_route_a.py:23`). It counts every flipped block that
  fails, also the blocks that fail when clean. At tolerance 0.01, 96.7% are "caught" while 91.55% of clean
  blocks are rejected. Conditioned on a clean pass, 96.7% becomes 82.5% and 74.5% becomes 61.5%.
- **The mismatch ignores signs** (`gelucheck/route_a.py:19`). It compares magnitudes only. A sign test raises
  detection at tolerance 10 from 0.8% to 22.2% and clean rejection from 0.01% to 2.31%.
- **The rounding-aware tolerance is first order.**
- **`figures/route_a_tradeoff.png`** shows Route B as a point typed in at (0, 100)
  (`scripts/make_figures.py:41`). It is not measured in that figure's setup.

## Route B on the CPU

- **"bound" is the wrong word** (`scripts/run_route_b.py:68`). For that fixed pair of errors, `1/(p−1)` is the
  exact escape probability. The measured 325 of 20 million is inside the noise (expected 305, standard
  deviation 17.5).
- **The two-error experiment does not call `route_b.checksum`** (`scripts/run_route_b.py:66`). It repeats the
  formula inline.
- **`Phi` and `gelu` reject a Python scalar** (`gelucheck/gelu.py:17`). Pass an array.
- **The BF16 reference rounds twice** (`gelucheck/formats.py`, `bf16_bits`): float64 to FP32 to BF16. A float64
  value just above a BF16 midpoint can go to the wrong neighbour. The docstring states it. The test of BF16
  rounding uses `<=` (`tests/test_gelucheck.py:37`), so it does not test ties to even.
- **The largest finite FP16 value** has an infinite upper cell boundary. No GELU output on [−5, 5) is near it.
