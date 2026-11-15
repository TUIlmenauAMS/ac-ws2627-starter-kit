# Rice coder without `bitstream` / `audio.coders`

The lossless coders in [Python-Audio-Coder](https://github.com/TUIlmenauAMS/Python-Audio-Coder)
need the packages `bitstream` and `audio.coders`. Neither installs on current Python
(3.12/3.13 with NumPy 2, as on Google Colab), and the course notebook still runs them under Python 2.7.
This folder replaces both with `ricecoder.py`, which needs only NumPy.

Needed for Topics 6, 7 and 9 (and Topic 8, which uses Topic 7's baselines).

## Files

| File | What it is |
| --- | --- |
| `ricecoder.py` | Rice encoder/decoder, same bit format as `audio.coders.rice` |
| `lossless_rice_audio_encoder.py`, `lossless_rice_audio_decoder.py` | IntMDCT + Rice lossless coder (Book Ch. 8), Rice lines replaced |
| `lossless_predictive_audio_encoder.py`, `lossless_predictive_audio_decoder.py` | NLMS predictive lossless coder, Rice lines replaced |
| `ricecodertest.py` | The repository's small Rice demo, rewritten for `ricecoder.py` |
| `IntMDCTfb.py`, `MDCTfb.py`, … | Unchanged copies from the repository, so the folder runs on its own |
| `test_vs_original.py` | The comparison test (needs the original packages, see below); for tutors, not for students |

Only the import and the Rice encode/decode lines differ from the originals.

## Usage (Colab or local)

```python
!git clone https://github.com/TUIlmenauAMS/Python-Audio-Coder
%cd Python-Audio-Coder
# copy ricecoder.py and the four lossless_*.py files from this folder into it (overwrite), then:
!python lossless_rice_audio_encoder.py fantasy-orchestra.wav
!python lossless_rice_audio_decoder.py fantasy-orchestra.lacodrice
```

In your own code:

```python
from ricecoder import rice_encode, rice_decode
data = rice_encode(x, b=3)                  # integers -> bytes (last byte padded with 0-bits)
y = rice_decode(data, b=3, count=len(x))    # bytes -> integers
```

The patched coders call `rice_encode(..., drop_partial_byte=True)`: like the original, they drop the incomplete last byte of every subband/block, so their compressed files stay byte-identical to the original's.

## Verified against the original (11 Nov 2026)

Reference: `bitstream` 3.0.1 and `audio.coders` 5.0.2, built on Python 3.11 with NumPy 1.26.

- **Rice code, 800 random cases** (b = 0…15, signed and unsigned, 1–700 values each): identical bits, identical bytes (padded and truncated). Each decoder decodes the other's output.
- **Full coders**, `fantasy-orchestra.wav` (stereo, 11.8 s) with the IntMDCT coder; `test48khz.wav` (mono, 4 s) and the first 3 s of `fantasy-orchestra.wav` with the predictive coder: compressed files and decoded WAV files **byte-identical** to the original.
- The same identical output on **Python 3.12 and 3.13 with NumPy 2.5**, with `bitstream`/`audio.coders` not installed.

## Behaviour of the original coders (unchanged, keep in mind for lossless results)

- **IntMDCT + Rice:** the output is delayed by 1024 samples (N). After aligning, it is sample-exact except the last 4,869 samples (about 110 ms of the test file): dropping the partial byte loses the last values of each subband, so the decoder reads 4 blocks fewer.
- **Predictive:** sample-exact except the last 10 samples (the predictor order).

So "verify sample-exact reconstruction" means: align by the coder's delay and compare over the samples the coder keeps, as the topic descriptions already say ("align and trim").

Speed: about as fast as the original (IntMDCT decoder about 7.5 s instead of 5.6 s for the 11.8 s stereo file).
