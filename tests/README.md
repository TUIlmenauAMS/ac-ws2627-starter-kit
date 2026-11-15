# Kit checks (for tutors)

One command shows whether everything students rely on still works: after any change to the kit, before
rebuilding `starter_kit.zip`, and when a course repository has changed.

```
python starter_kit/tests/run_checks.py --quick                         # about 7 minutes
python starter_kit/tests/run_checks.py                                 # everything, about 9-12 minutes
SQAM_ZIP=/path/to/TECH3253_SQAM_FLAC.zip python starter_kit/tests/run_checks.py    # with the SQAM test set
python starter_kit/tests/run_checks.py --only rice,errors --workdir ~/kit_checks   # some checks, reusing clones
python starter_kit/tests/run_checks.py --list                          # all check ids
```

Needs Python 3.12 or newer, git and access to github.com, and the packages
`numpy scipy matplotlib pandas soundfile dahuffman joblib` (plus `nbconvert ipykernel` for the notebook check).
To test new pins in `setup_topic.py` before they are on GitHub, set `AC_REPO_BASE` to a folder of local clones
(`AC_REPO_BASE=file:///path/to/clones/`; each clone needs `git config uploadpack.allowAnySHA1InWant true`).
No absolute paths: everything happens in the work folder (`--workdir`, default a new temporary folder, which is
kept so you can look at the outputs; pass the same `--workdir` again to reuse the clones). The kit itself is never
changed. Every check prints PASS, FAIL or SKIP with its run time; the exit code is non-zero if any check failed.
The results also go to `check_results.csv` in the work folder.

Test set: with the environment variable `SQAM_ZIP` set, the checks use the SQAM test set and
`reference_results_sqam-v1.csv`; otherwise the fallback set and `reference_results_librosa-v1.csv`.

## What each check means

| Check | What it does | If it fails |
| --- | --- | --- |
| `setup.clone` | Clones Python-Audio-Coder at the commit pinned in `setup_topic.py`, as the demo notebook does | Network, or the repository moved |
| `setup.testset` | Builds the test set (SQAM from `SQAM_ZIP`, else the fallback set) | Wrong zip (SHA-256), or the clip hashes changed |
| `reference` | `quality_sweep` on all clips (`--quick`: tuning clips) at quality 50/100/200/400 reproduces the reference file: bits/sample within 1e-3, SNR within 0.01 dB, percloss within 1 % | **The most important check.** Students' numbers would no longer match: the coder repository or a measurement function changed |
| `monotonic` | For every clip, bitrate and SNR rise and percloss falls with quality (measured rows and both reference files) | A clip with digital silence, or a broken coder |
| `rice.roundtrip` | Rice coder: random integers for b = 0…12, signed and unsigned; empty input; non-integer input raises ValueError | `rice_coder/ricecoder.py` changed |
| `rice.coders` | Both lossless coders on one stereo and one mono clip: exact up to the documented end loss (predictive: remainder of the 20-ms blocks + 10 samples; IntMDCT: last 4 blocks, mono duplicated) | Lossless coders or `lossless_check` changed |
| `errors` | Clear errors for documented mistakes: mono/stereo mismatch, decoded signal shorter than the delay, `trim_end` too large, wrong `decoded_suffix`/`compressed_ext`, CSV with other columns, broken SQAM zip (fallback with warning) | Error handling in `ac_kit.py` changed |
| `topics.*` | The starting code of the topics runs at the pinned commits (which include the fixes) (ported from `work_packages/reference_tests/`): KBD perfect reconstruction (T1), the fixed quantizer uses the window and is byte-identical with the sine window, N = 256/2048 with delay N (T2), LDFBclass/MDCTclass perfect reconstruction (T3, T4), optimization examples, Random Directions and the CMA-ES demo (T5), LPC/LMS/WCLMS examples (T7), nfilts = 32 (T12), pre-/post-filter on mono and stereo (T13) | A pinned commit or a kit file changed |
| `preecho.*` | `find_onsets` on synthetic clicks (±2 ms), `pre_echo_db` (noise before vs. after onsets), N = 2048 gives more pre-echo than N = 256 at similar SNR, `quality_sweep(extra_metrics=...)` keeps the standard columns | The pre-echo functions changed |
| `topic3.*` | Low-delay filter bank: N = 128…2048 perfect reconstruction with delay N, same shapes as `MDCTanafb`, `filterbank_report`, coder copies byte-identical with `--filterbank mdct` | `topic3/` or the FilterBanks repository changed |
| `topic5.*` | Window family (sine at p = 0, perfect reconstruction), objectives improvable, evaluation counter exact across Random Directions' worker processes, window coder copies byte-identical, `compare_optimizers.py` | `topic5/` or the Random Directions repository changed |
| `baseline.*` | `baseline.py` for topics 2 and 12 (lossy, rows equal the reference), 7 and 9 (lossless, exact); topic 15 only with `RUN_T15=1` (LibriSpeech download, needs pesq, pystoi, ffmpeg) | `baseline.py` or the coders changed |
| `notebook` | Zips the kit as for the students (without `testset/` and `__pycache__`), runs `starter_kit_demo.ipynb` with nbconvert (test-set cell pointed at the same test set; the topic cell set to topic 2) and looks for "setup matches the reference" | The notebook no longer runs top to bottom |

Not in `--quick`: `topics.t1_kbd_coder`, `topics.t5_examples`, `topics.t5_bbo`, `topics.t7_lmslossless`,
`topics.t7_wclms_lossless` (and the reference check runs on the
tuning clips only). `baseline.t15` runs only with `RUN_T15=1` (about 2 minutes with the dev-clean download).

Where no Jupyter kernel can start (some sandboxes forbid the local sockets a kernel needs), the notebook check runs
the notebook's code cells one after another in a plain Python process instead and says so in its output.

## Run times

Measured on 12 Nov 2026 (MacBook, 8 cores, Python 3.13, NumPy 2.5, SciPy 1.18), clones already present:

| Run | Time |
| --- | --- |
| `--quick`, fallback set | 7.1 min (36 checks pass, 6 skipped), including the first clones |
| full, fallback set | 9.1 min (41 pass, Topic 15 skipped) |
| full, SQAM (`SQAM_ZIP` set) | 11.4 min (41 pass, Topic 15 skipped) |

The reference check takes most of the time (about 3.5 s per clip and quality value). First clones add about a minute.
