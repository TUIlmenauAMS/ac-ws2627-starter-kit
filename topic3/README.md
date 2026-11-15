# Topic 3 helper: low-delay filter bank in the course coder

`LDFB.py` in Python-Audio-Coder, named in the topic description, cannot be used (no coefficients for the
coder's N, and a different structure). The working low-delay filter bank is `LDFBclass` in
[FilterBanks_FastPythonImplementation_Classes](https://github.com/TUIlmenauAMS/FilterBanks_FastPythonImplementation_Classes).
It works block by block, while the coder filters whole files. This folder connects the two, so you can
spend your time on the comparison.

| File | What it is |
| --- | --- |
| `ldfb_filebased.py` | `ldfb_coefficients(N)`, `LDFBanafb(x, N, coeff)`, `LDFBsynfb(y, coeff)` (same conventions as `MDCTanafb`/`MDCTsynfb`), `filterbank_report(N)`, `install(coder_dir)` |
| `audio_encoder_ldfb.py`, `audio_decoder_ldfb.py` | Copies of the coder with two extra options: `--filterbank mdct` or `ldfb`, and `--N` (number of subbands). Changed lines are marked `# topic3`. With the defaults (`mdct`, `N = 1024`) the output is byte-identical to the original coder |

## Setup (Colab or local)

```python
!python starter_kit/setup_topic.py --topic 3     # clones the filter bank repository at the kit's commit
import os, sys
os.environ['AC_FB_DIR'] = os.path.abspath('FilterBanks_FastPythonImplementation_Classes')
sys.path.insert(0, os.path.abspath('starter_kit/topic3'))
import ldfb_filebased as L
L.install('Python-Audio-Coder')      # copies the files above into the coder folder (originals stay)
L.filterbank_report(512)             # delays, reconstruction SNR, stopband attenuation: LDFB vs MDCT
```

`AC_FB_DIR` tells the encoder and decoder where the coefficient file is. Supported N: 128, 256, 512, 1024, 2048
(the file has the coefficients for N = 512; the others are interpolated with `fbinterp2.fbinterp()`).

## Example: both filter banks in a sweep

```python
clips = k.load_manifest(split='tuning', classes=['speech'])
for fbank in ('mdct', 'ldfb'):
    k.quality_sweep(clips, out_csv='topic3_example.csv', method='%s N=512' % fbank,
                    delay=512, block=512,            # the delay the kit needs to align: N for both
                    encoder='audio_encoder_ldfb.py', decoder='audio_decoder_ldfb.py',
                    extra_args=['--filterbank', fbank, '--N', '512'])
k.plot_rate_quality('topic3_example.csv')
```

The decoder reads the filter bank and N from the compressed file, so it needs no arguments.
The settings record adds about 40 bytes per file (it is counted in the bitrate; for the default MDCT with
N = 1024 nothing is added).

## Delay: two numbers

`filterbank_report(N)` measures both with impulses:

- **Block delay** (N for both filter banks): an impulse at sample p of the input file comes out at sample p + N of
  the decoded file. This is the `delay` that `k.evaluate()` and `k.quality_sweep()` need.
- **System (algorithmic) delay** = block delay + N − 1: in a real-time system a block is processed only when its
  last sample has arrived, so the first sample of a block waits N − 1 more samples. Report this one, in samples
  and ms.

Measured (12 Nov 2026, fs = 44.1 kHz, analysis filter of band 4):

| N | Filter bank | Block delay | System delay | Filter taps | Reconstruction SNR | Stopband attenuation, 2 / 5 band widths away |
| --- | --- | --- | --- | --- | --- | --- |
| 512 | LDFB | 512 | 1023 (23.2 ms) | 2048 | 309 dB | 22.0 / 58.8 dB |
| 512 | MDCT | 512 | 1023 (23.2 ms) | 1024 | 309 dB | 23.5 / 39.9 dB |
| 1024 | LDFB | 1024 | 2047 (46.4 ms) | 4096 | 309 dB | 22.0 / 58.8 dB |
| 1024 | MDCT | 1024 | 2047 (46.4 ms) | 2048 | 309 dB | 23.5 / 39.9 dB |

So at the same N, the LDFB has the same delay as the MDCT but filters twice as long: it gains attenuation far
from the band, not next to it. The coefficient file's name (`fb2048t1023d512`: 2048 taps, delay 1023, 512 bands)
agrees with the measured system delay for N = 512. Reconstruction is perfect (numerical precision) for both,
with `pyrecplayfastLDFB_classes.py` at the kit's commit (versions before the fix of Nov 2026 give 46.9 dB).

## What remains your work

- Choose the two delay budgets (in ms) and, for each, the LDFB and the MDCT that meet it. Check how the delay
  of each depends on N, and whether an MDCT with the same delay as the LDFB must have a different N.
- Compare the frequency responses (plot them; `analysis_impulse_response()` gives the filters).
- Run the sweeps (at least 4 operating points) on the evaluation clips, per filter bank and delay budget.
- Interpret the results: where does the longer filter help, and where does the psychoacoustic model (which was
  tuned for the MDCT) limit it?
