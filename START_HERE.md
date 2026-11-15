# Start here: what to run for your topic

One section per topic, each with the same five parts. Commands run in a terminal in your project folder
(the folder that contains `starter_kit/`). **In Colab, put `!` in front of each command.**

Every topic starts the same way: run `starter_kit/starter_kit_demo.ipynb` once. It gets the kit, clones
Python-Audio-Coder at the version the kit was tested with, builds the test set and checks your setup against the
reference. Below, "the coder" means that folder. Then set up the rest of what your topic needs with one command:
```
python starter_kit/setup_topic.py --topic N
```
It clones the other course repositories your topic needs (at the kit's versions) and copies the kit files listed
below into the coder. Running it again is safe. Measurement rules: `starter_kit/README.md`.

## Topic 1: Sine vs. KBD windows in the MDCT coder

**You need:** the coder from the demo notebook (its quantizer uses the window you pass in). No other repository or
package (KBD: `scipy.signal.windows.kaiser_bessel_derived`).

**Your baseline (for your proposal):** `python starter_kit/baseline.py --topic 1 --variant a --out results/baseline.csv`
writes `results/baseline.csv` (sine window; variant a: tonal + polyphonic tuning clips, b: speech + transient).

**What you change:** the line `fb=np.sin(...)` in **both** `audio_encoder.py` and `audio_decoder.py` of your copy
of the coder. The coder takes the first 1.5·N values of a 2N-point window.

**Verification check (passes by the Checkpoint):** perfect reconstruction without quantization (error at numerical
precision): `MDCTsynfb(MDCTanafb(x, N, fb), fb)`, aligned with `k.align(x, xr, delay=N)`, max. absolute error.

**Common mistakes:**
- Changing the window only in the encoder: the decoder must use the same window.
- Using a clone of the coder from before November 2026: its `psyac_quantization.py` overwrites your window with the sine window.
- Reporting the bitrate that `audio_encoder.py` prints: use `k.bits_per_sample` (file size).

## Topic 2: MDCT length vs. delay vs. quality

**You need:** the coder from the demo notebook, one copy per N:
```
cp -r Python-Audio-Coder coder_N256      # then N=1024 -> N=256 in audio_encoder.py AND audio_decoder.py
```
No other repository or package. The coder runs unchanged with N = 128, 256, 512, 2048.

**Your baseline (for your proposal):** `python starter_kit/baseline.py --topic 2 --variant a --out results/baseline.csv`
writes `results/baseline.csv` (N = 1024; variant a: speech + transient tuning clips, b: tonal + polyphonic).

**What you change:** `N=1024` in `audio_encoder.py` and `audio_decoder.py` (one coder copy per length).

**Verification check (passes by the Checkpoint):** delay formula checked against a measurement: `k.find_delay(x, y)` on a
decoded clip (the decoded file is delayed by N). Pre-echo on at least 3 transient clips:
`k.quality_sweep(clips, delay=N, block=N, coder_dir='coder_N256', extra_metrics=[k.pre_echo_db])`
(definition: README, "Pre-echo").

**Common mistakes:**
- Keeping `delay=1024, block=1024` in the sweep: pass `delay=N, block=N` for every N.
- Changing N in only one of the two scripts.
- Measuring pre-echo on the decoded signal's onsets: `pre_echo_db` finds them in the original.

## Topic 3: Low-delay filter bank vs. MDCT at the same delay

**You need:** `starter_kit/topic3/` and the filter bank repository (for the coefficients):
`python starter_kit/setup_topic.py --topic 3`. Then follow "Setup" in `starter_kit/topic3/README.md` (sets `AC_FB_DIR`, copies the LDFB coder files into the coder).

**Your baseline (for your proposal):** `python starter_kit/baseline.py --topic 3 --variant a --out results/baseline.csv`
writes `results/baseline.csv` (MDCT, N = 1024, speech + transient + polyphonic tuning clips).

**What you change:** run `audio_encoder_ldfb.py` / `audio_decoder_ldfb.py` with `--filterbank mdct|ldfb --N n`
(`extra_args` of `k.quality_sweep`); choose the two delay budgets and the matching N.

**Verification check (passes by the Checkpoint):** delay of both filter banks derived and verified, reconstruction error
without quantization: `filterbank_report(N)` in `topic3/ldfb_filebased.py` (measured with impulses).

**Common mistakes:**
- Using `LDFB.py` from Python-Audio-Coder: no coefficients for the coder's N, and a different structure.
- Using a clone of the filter bank repository from before November 2026: its `pyrecplayfastLDFB_classes.py` fails on import (global N) and reconstructs only to 46.9 dB.
- Confusing the block delay N (what the kit needs to align) with the system delay 2N − 1 (what you report).

## Topic 4: Transient-triggered window switching

**You need:** the filter bank repository: `python starter_kit/setup_topic.py --topic 4`.
Import the classes only; the file's live demo (`__main__`) needs pyaudio, PyQt5 and cv2, which you do not need.

**Your baseline (for your proposal):** `python starter_kit/baseline.py --topic 4 --variant a --out results/baseline.csv`
writes `results/baseline.csv` (fixed long MDCT, N = 1024, speech + transient + polyphonic tuning clips).

**What you change:** `MDCTclass` in your copy of `pyrecplayfastLDFB_classes.py` (window per block, synthesis
switching one block later, short blocks zero-padded), your transient detector, and a file-based coder around it
(`_analysis`/`_synthesis` in `topic3/ldfb_filebased.py` show how to loop a block class over a file).

**Verification check (passes by the Checkpoint):** perfect reconstruction across every switch: `forward()`/`inverse()` without
quantization, aligned with `k.align(x, xr, delay=N)`; pre-echo: `k.pre_echo_db` (README, "Pre-echo").

**Common mistakes:**
- Running the file's `__main__` (real-time demo) instead of importing the classes.
- Leaving the switching flags out of the bitrate: they are side information.
- Switching the synthesis window in the same block as the analysis window (it must follow one block later).

## Topic 5: Random Directions vs. gradient optimization of filter banks

**You need:** `starter_kit/topic5/` and the Random Directions repository (joblib is on Colab):
`python starter_kit/setup_topic.py --topic 5`. It also clones `BlackBoxOptimizerSPcomparison` (optional: CMA-ES
demo) and `FilterBanks_FastPythonImplementation_Classes` (variant b). Read `starter_kit/topic5/README.md`.

**Your baseline (for your proposal):** `python starter_kit/baseline.py --topic 5 --variant a --out results/baseline.csv`
writes `results/baseline.csv` (sine window, speech + transient + polyphonic tuning clips).

**What you change:** the optimizer runs on `window_design.py`'s objectives (your choice of K, budget, objective);
the optimized window goes into the coder through `audio_encoder_window.py` and `AC_WINDOW_FILE`.

**Verification check (passes by the Checkpoint):** equal evaluation budgets for all optimizers: `counted(...)` in
`topic5/window_design.py`; results on held-out clips: `k.quality_sweep(k.load_manifest(split='evaluation'), ...)`.

**Common mistakes:**
- Counting evaluations with a variable in the objective: Random Directions evaluates in worker processes.
- Using `optimfuncMDCT`/`optimfuncLDFB` at N = 1024: they are built for small N and do not improve there.
- Passing a lambda as objective: parallel workers need a module-level function (`functools.partial`).

## Topic 6: Fixed vs. adaptive Rice parameter

**You need:** the kit's Rice coder (`bitstream` and `audio.coders` do not install on current Python):
`python starter_kit/setup_topic.py --topic 6` copies `rice_coder/ricecoder.py` and `rice_coder/lossless_*.py` into
the coder. No other package.

**Your baseline (for your proposal):** `python starter_kit/baseline.py --topic 6 --variant a --out results/baseline.csv`
writes `results/baseline.csv` (predictive lossless coder on all tuning clips: bits, exactness, end loss, times).

**What you change:** how the Rice parameter is chosen in `lossless_predictive_audio_encoder.py` and
`lossless_predictive_audio_decoder.py` (block-adaptive or causal online); `ricecodertest.py` for small tests.

**Verification check (passes by the Checkpoint):** sample-exact reconstruction on every clip: `k.lossless_check(x, y, delay=0,
trim_end=...)` (README, "Lossless topics"); bitrate with all side information: `k.bits_per_sample`.

**Common mistakes:**
- Calling a result exact without reporting the samples at the end that the coder does not reconstruct (484 on the 6-s clips).
- Copying the original lossless coders: they need `bitstream`/`audio.coders`.
- Leaving the transmitted Rice parameters out of the bitrate.

## Topic 7: LPC vs. LMS vs. WCLMS prediction for lossless coding

**You need:** the coder, the kit's Rice coder and `fspeech.wav` (the examples read it):
`python starter_kit/setup_topic.py --topic 7` copies `rice_coder/ricecoder.py` and `rice_coder/lossless_*.py` into
the coder and `fspeech.wav` from ADSP_Tutorials (only that file is downloaded).

**Your baseline (for your proposal):** `python starter_kit/baseline.py --topic 7 --variant a --out results/baseline.csv`
writes `results/baseline.csv` (predictive NLMS coder on all tuning clips). It builds the test set first if needed.

**What you change:** the predictor in `lossless_predictive_audio_encoder.py` / `_decoder.py` (LPC, LMS, WCLMS),
starting from `lpcexample.py`, `lmslosslessexample.py`, `wclms_losslessexample.py`; LPC coefficients count as side information.

**Verification check (passes by the Checkpoint):** sample-exact reconstruction: `k.lossless_check` (README, "Lossless
topics"); runtime on stated hardware: the `encode_s`/`decode_s` columns.

**Common mistakes:**
- Using a clone of the coder from before November 2026: its `sound.py` needs pyaudio (not on Colab) and its
  `lpcexample.py` uses `np.int` (removed in NumPy 1.24).
- Missing `fspeech.wav`: it is in ADSP_Tutorials, not in Python-Audio-Coder (`setup_topic.py` copies it).

## Topic 8: Neural vs. classical predictor for lossless coding

**You need:** everything from Topic 7 (the classical baselines): `python starter_kit/setup_topic.py --topic 8`, and
PyTorch (on Colab already).

**Your baseline (for your proposal):** `python starter_kit/baseline.py --topic 8 --variant a --out results/baseline.csv`
writes `results/baseline.csv` (predictive NLMS coder on all tuning clips).

**What you change:** the predictor in `lossless_predictive_audio_encoder.py` / `_decoder.py`: a small causal
network whose predictions are rounded to integers with identical computation in encoder and decoder.

**Verification check (passes by the Checkpoint):** sample-exact reconstruction: `k.lossless_check`; model size in the bitrate
when weights are transmitted (variant b): `k.bits_per_sample` of all transmitted files; train/validation/test clips separate.

**Common mistakes:**
- Encoder and decoder on different devices or with non-deterministic settings: no bit-exact predictions. Run both on the CPU.
- Not rounding the predictions to integers before the residual.
- Training and testing on the same clips.

## Topic 9: Predictive vs. integer-MDCT lossless coding

**You need:** the kit's Rice coder (both lossless coders of the book, with the Rice lines replaced):
`python starter_kit/setup_topic.py --topic 9` copies `rice_coder/*.py` into the coder.

**Your baseline (for your proposal):** `python starter_kit/baseline.py --topic 9 --variant a --out results/baseline.csv`
writes `results/baseline.csv` (both coders on all tuning clips; mono clips duplicated to stereo for IntMDCT).

**What you change:** variant a: nothing in the coders (your analysis of residuals and coefficients);
variant b: one coder (block length, predictor order), compared with the original.

**Verification check (passes by the Checkpoint):** sample-exact for both coders and the dropped end samples reported:
`k.lossless_check(x, y, delay=1024)` (IntMDCT) or `delay=0` (predictive), with `trim_end` (README, "Lossless topics").

**Common mistakes:**
- Giving the IntMDCT coder a mono file: it needs stereo (duplicate the channel).
- Forgetting the IntMDCT coder's delay of 1024 samples.
- Not reporting the end loss: IntMDCT 4096 samples, predictive 484 samples per 6-s clip.

## Topic 10: Quality loss under repeated encoding

**You need:** the coder from the demo notebook. No other repository or package.

**Your baseline (for your proposal):** `python starter_kit/baseline.py --topic 10 --variant a --out results/baseline.csv`
writes `results/baseline.csv` (one generation, speech + transient + polyphonic tuning clips).

**What you change:** the generation loop (README, "Repeated coding (Topic 10)"): at least 10 generations, the
quality setting, and for variant b a time shift of the frames between generations.

**Verification check (passes by the Checkpoint):** decoder delay compensated before every comparison, SNR and `percloss` per
generation against the original: `k.evaluate(x0, y, fs, delay=0)` after removing each generation's 1024 samples.

**Common mistakes:**
- Not removing the 1024-sample delay in each generation: the delays add up.
- Writing the next generation without rounding and clipping to 16 bit (the README loop does both).
- Comparing clips that contain digital silence: `percloss` explodes there.

## Topic 11: Psychoacoustic vs. energy vs. uniform bit allocation

**You need:** the coder from the demo notebook. No other repository or package.

**Your baseline (for your proposal):** `python starter_kit/baseline.py --topic 11 --variant a --out results/baseline.csv`
writes `results/baseline.csv` (masking-based rule; variant a: speech + transient, b: tonal + polyphonic).

**What you change:** the step-size rule in `MDCT_psayac_quant_enc()` (`PythonPsychoacoustics/psyac_quantization.py`):
add a uniform and an energy-based rule; `MDCTsyn_dequant_dec()` must use the same step sizes.

**Verification check (passes by the Checkpoint):** at least 4 operating points per method at matched actual bitrate:
`k.quality_sweep` (bitrate from the file size); noise spectrum against the masking threshold for one frame.

**Common mistakes:**
- Comparing methods at the same `quality` value instead of the same measured bitrate.
- Reporting the bitrate that `audio_encoder.py` prints.
- Expecting differences below about 1.4 bit/sample: that is the coder's bitrate floor.

## Topic 12: Sensitivity of the psychoacoustic model

**You need:** the coder from the demo notebook. No other repository or package.

**Your baseline (for your proposal):** `python starter_kit/baseline.py --topic 12 --variant a --out results/baseline.csv`
writes `results/baseline.csv` (unchanged model, speech + transient + polyphonic tuning clips).

**What you change:** one parameter at a time: `alpha=0.8` in `psyac_quantization.py`, `nfilts=64` in **both**
`audio_encoder.py` and `audio_decoder.py`, spreading function and threshold in quiet in `PythonPsychoacoustics/psyacmodel.py`.

**Verification check (passes by the Checkpoint):** quality always measured with the unmodified model and SNR: `k.evaluate`
(the kit's `percloss` always uses `psyacmodel_reference.py`, even when you change `psyacmodel.py`).

**Common mistakes:**
- Measuring quality with your modified model: it would grade itself.
- Changing `nfilts` in only one of the two scripts.
- Importing `psyacloss.py` from Python-Audio-Coder: it loads your modified model; use the kit's (`k.percloss_value`).

## Topic 13: Psychoacoustic pre-/post-filter vs. MDCT coder

**You need:** the coder and the kit's Rice coder (lossless coding of the quantized signal):
`python starter_kit/setup_topic.py --topic 13` copies `rice_coder/ricecoder.py` into the coder and clones
`AdaptivePrePostFilters` (optional: related work).

**Your baseline (for your proposal):** `python starter_kit/baseline.py --topic 13 --variant a --out results/baseline.csv`
writes `results/baseline.csv` (the MDCT coder, speech + transient + polyphonic tuning clips).

**What you change:** the chain `psyacprefilterToFile.py` → quantization → lossless coding (`ricecoder.py`) →
`psyacpostfilterFromFile.py`; filter order and coefficient update interval.

**Verification check (passes by the Checkpoint):** delay of both coders derived and stated in ms, checked with `k.find_delay(x, y)`
(pre-/post-filter measured: 128 samples, 2.9 ms); filter coefficients in the bitrate: `k.bits_per_sample` of all transmitted files.

**Common mistakes:**
- Using a clone of the coder from before November 2026: its post-filter fails on mono files.
- The pre-filter writes `<name>pref.wav` and `<name>mT.wav`: give the post-filter `<name>pref.wav`.
- Leaving the transmitted filter coefficients (masking thresholds) out of the bitrate.

## Topic 14: Mid/side vs. left/right stereo coding

**You need:** the coder from the demo notebook. No other repository or package.

**Your baseline (for your proposal):** `python starter_kit/baseline.py --topic 14 --variant a --out results/baseline.csv`
writes `results/baseline.csv` (left/right coding of the stereo and polyphonic tuning clips; the speech clips are mono).

**What you change:** the channel loop in `audio_encoder.py` / `audio_decoder.py`: fixed mid/side (variant a) or
per-frame switching with flags (variant b).

**Verification check (passes by the Checkpoint):** quality on the decoded left and right channels: `k.evaluate(x, y, fs)` on the
stereo files; inter-channel correlation per clip: column `lr_correlation` of `k.load_manifest()`.

**Common mistakes:**
- Measuring quality on mid/side signals instead of the decoded left/right.
- Leaving the switching flags out of the bitrate.
- Testing on mono clips: there is nothing to gain.

## Topic 15: Low-delay neural speech codec

**You need:** `starter_kit/topic15/` (read its README) and two packages (Colab has PyTorch, soundfile and ffmpeg):
```
pip install pesq pystoi
```
LibriSpeech is downloaded by `speech_kit.download_librispeech()` (dev-clean 0.35 GB, train-clean-100 6.3 GB); keep it on Google Drive.

**Your baseline (for your proposal):** `python starter_kit/baseline.py --topic 15 --variant a --out results/baseline.csv`
writes `results/baseline.csv` (Opus, 10-ms frames, 8/12/16 kbit/s on 5 dev-clean utterances: bitrate, PESQ, STOI, delay).

**What you change:** your codec model (PyTorch): train on train-clean-100, choose settings on dev-clean, report on test-clean.

**Verification check (passes by the Checkpoint):** speaker-disjoint splits: `s.check_speakers_disjoint`; bitrate from the
transmitted indices: `s.huffman_bytes` and `s.bitrate_kbps`; `s.pesq_wb`, `s.stoi_score`, `s.real_time_factor`.

**Common mistakes:**
- Bitrate from the bottleneck dimension instead of the transmitted bytes (`entropy_bits` is only a lower bound).
- `torchaudio.datasets.LIBRISPEECH` needs `torchcodec` since torchaudio 2.9: use `s.read_audio`.
- Real-time factor measured on the GPU: measure on a stated CPU.
