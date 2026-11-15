# Topic 5 helper: Random Directions vs. gradient optimization

The objective functions in Python-Audio-Coder (`optimfuncMDCT`, `optimfuncLDFB`) are built for small N: from the
sine window at N = 64 or 256, L-BFGS-B improves them only from 1.252e9 to 1.251e9 in 3000 evaluations, and at the
coder's N = 1024 they do not work at all. This folder gives you a window with a few parameters at the coder's
block length, objectives that can be improved, and a counter for "equal numbers of evaluations".
**The comparison of the optimizers is your work**; `compare_optimizers.py` is only an example of the mechanics.

| File | What it is |
| --- | --- |
| `window_design.py` | `window_from_params(p, N)`, objectives `stopband_energy`, `coder_objective`, `ldfb_objective`, the evaluation counter `counted`, `install(coder_dir)` |
| `compare_optimizers.py` | Example: Random Directions vs. L-BFGS-B, 3 runs, CSV and figure |
| `audio_encoder_window.py`, `audio_decoder_window.py` | Copies of the coder that read the window from the file named by `AC_WINDOW_FILE` (changed lines marked `# topic5`; without the variable the output is byte-identical to the original) |

## Setup (Colab or local)

```python
!python starter_kit/setup_topic.py --topic 5     # clones the Random Directions repository (and the others) at the kit's commits
import os, sys
os.environ['AC_RD_DIR'] = os.path.abspath('LowDelayMultichannelSourceSeparation_Random-Directions_Demo')
sys.path.insert(0, os.path.abspath('starter_kit/topic5'))
import window_design as W
!python starter_kit/topic5/compare_optimizers.py          # about 30 s: topic5_example/convergence.png
```

## The window family

`window_from_params(p, N)` gives the first 1.5·N window values the coder needs, from K parameters (K = 4…12).
The rising half of the window is sin(π/2 · g(u)), u = (n + 0.5)/N, with a smooth warp
g(u) = u + Σ p_k sin(2πku)/(2πk). For p = 0 it is exactly the sine window. Every p gives a symmetric window with
w(n)² + w(n+N)² = 1 (Princen–Bradley), so analysis and synthesis use the same window and reconstruction is perfect
(error about 2·10⁻¹⁵). The docstring explains why.

## Objectives (all: lower is better)

| Objective | What it measures | Seconds per evaluation (12 Nov 2026, laptop CPU) | Gradient? |
| --- | --- | --- | --- |
| `stopband_energy(p, N)` (variant a) | Share of the window's spectrum more than 3 subband widths away from 0, in dB. Sine window −31.5 dB, KBD β = 8 −43.7 dB | 0.0003 s | numerical, smooth |
| `ldfb_objective(p, N)` (variant b) | The same measure for the low-delay filter bank's prototype filter; `p` scales the course's LDFB coefficients smoothly (`ldfb_coeff_from_params`). Needs `FilterBanks_FastPythonImplementation_Classes` (`AC_FB_DIR`), see `topic3/README.md` | 0.004 s | numerical; very sensitive far from p = 0 |
| `coder_objective(p, N, clip, quality, coder_dir)` (extension) | Bitrate of the real coder from the file size, plus a penalty when `percloss` rises above the sine window's | about 3 s for a 6-s clip at N = 1024 | none (step function): needs a black-box optimizer |

For `coder_objective`, first copy the window coder files into your coder folder: `W.install('Python-Audio-Coder')`.

## Counting evaluations

`optimrandomdir()` evaluates its trials in parallel worker processes (joblib), so a counter variable in your
objective would stay 0 in the main process. Wrap the objective instead:

```python
import functools
f = W.counted(functools.partial(W.stopband_energy, N=1024), 'rd_run1.log', budget=2000)
try:
    optimrandomdir(f, p0, iterations=2000)
except W.BudgetExhausted:
    pass                                  # the budget is used up; the best point is in the log
f.write_csv('rd_run1.csv')                # evaluation, value, best_so_far, seconds
best_value, best_p = f.best()
```

Every evaluation writes one fixed-length line to the log file, from whichever process runs it; the count is the
file size divided by the line length. This counts the parallel trials, the line search of Random Directions and the
numerical-gradient evaluations of L-BFGS-B alike (tested: 500 of 500 evaluations from 9 processes counted).
With parallel workers a few evaluations can finish just after the budget is reached; `history()` and `write_csv()`
keep only the first `budget`. The objective must be a module-level function (use `functools.partial`, not a lambda).

## The example

`compare_optimizers.py` (K = 8, N = 1024, `stopband_energy`, 2000 evaluations, 3 runs from the same start points
for both optimizers) writes `convergence.csv`, `best_windows.csv` and `convergence.png`. Measured on 12 Nov 2026
(8 CPU cores, 26 s in total): both optimizers go from about −30 dB to −43 to −44 dB; L-BFGS-B converges after
150–220 evaluations and then stops, Random Directions needs about 300–1000. Options: `--runs`, `--budget`, `--K`,
`--objective ldfb`, `--spread` (start points). With `--objective ldfb`, start close to p = 0 (`--spread 0.05`):
random starts far from the course coefficients can give a very poor filter bank.

## What remains your work

- 10 runs per optimizer, convergence vs. evaluations **and** vs. run time (both are in the CSV), mean and spread.
- A third optimizer if you want (CMA-ES, see `bbo_demo_rd_vs_cmaes.py` in BlackBoxOptimizerSPcomparison).
- Your choice of objective, K and budget, and why.
- The optimized window in the coder: save it with `np.savetxt('best.txt', W.window_from_params(best_p, 1024))`, then
  `k.quality_sweep(clips, encoder='audio_encoder_window.py', decoder='audio_decoder_window.py',
  env={'AC_WINDOW_FILE': os.path.abspath('best.txt')}, method='optimized window')` on the **held-out** (evaluation)
  clips, next to the sine window.
- Variant (b): the same with `ldfb_objective`; put the optimized coefficients (`ldfb_coeff_from_params`, saved with
  `np.savetxt`) into the coder of `topic3/` with the environment variable `AC_LDFB_COEFF_FILE`.
