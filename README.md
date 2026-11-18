# Audio Coding seminar WS 2026/27 – starter kit

[![Open the demo notebook in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TUIlmenauAMS/ac-ws2627-starter-kit/blob/main/starter_kit_demo.ipynb)

Every group measures bitrate and quality the same way, so all papers report comparable numbers.
Start with **`starter_kit_demo.ipynb`** in Google Colab: it sets everything up, runs a first
rate–quality sweep and checks that your numbers match the reference. Then read your topic's section
in **`START_HERE.md`**: what your topic needs (one setup command) and the one command for the baseline in your proposal.
The notebook's GitLab cell puts your setup into your group's GitLab repository; in every later Colab session you
run the first cell and the GitLab cell, and continue from your repository. `gitlab_repo.save("...")` saves your work.

## Contents

| | What it is |
| --- | --- |
| `ac_kit.py` | Bit counter, delay alignment, SNR, psychoacoustic loss, quality sweep → CSV, plots, lossless check |
| `psyacloss.py`, `psyacmodel_reference.py` | `percloss()` from Python-Audio-Coder, fixed so that it imports, always with the **unmodified** psychoacoustic model |
| `sqam_test_set.py`, `sqam_clip_hashes.json` | Builds the shared test set from EBU SQAM (downloaded from the EBU, checked bit for bit) |
| `testset_fallback/`, `make_test_set.py` | Fallback test set (CC BY / public domain clips), used only if SQAM cannot be downloaded |
| `reference_results_sqam-v1.csv`, `reference_results_librosa-v1.csv` | Every clip at quality 50/100/200/400 with the unmodified coder, per test set: your setup must reproduce these numbers |
| `topic15/` | Topic 15 only: LibriSpeech data, PESQ/STOI, bitrate, delay, real-time factor, Opus baseline |
| `rice_coder/` | Rice coder without `bitstream`/`audio.coders` (they do not install on current Python): Topics 6–9 |
| `setup_topic.py` | `python starter_kit/setup_topic.py --topic 7`: clones the course repositories your topic needs at the versions the kit was tested with, and copies the kit files the topic needs (plain folders, no `.git` of their own) |
| `gitlab_repo.py` | The demo notebook's GitLab cell: puts your setup into your group's GitLab repository (first time) and works from that repository in later sessions; `save("...")` saves your work |
| `starter_kit_demo.ipynb` | Colab notebook: setup, first sweep, plots, reference check, your topic's baseline |
| `START_HERE.md` | One section per topic: what you need, your baseline command, what you change, the checks |
| `baseline.py` | `python starter_kit/baseline.py --topic 7 --variant a --out results/baseline.csv`: the baseline for your proposal |
| `topic3/` | Topic 3: low-delay filter bank with the coder's interface, delay and filter measurements |
| `topic5/` | Topic 5: window with a few parameters, objectives, evaluation counter, optimizer example |
| `tests/` | For tutors: `python starter_kit/tests/run_checks.py --quick` checks that the kit still works |

## Setup (Colab)

```python
!unzip -q starter_kit.zip                  # or: !git clone -q https://github.com/TUIlmenauAMS/ac-ws2627-starter-kit starter_kit && rm -rf starter_kit/.git
!python starter_kit/setup_topic.py --topic 1      # your topic: the course code at the kit's versions
!pip install -q dahuffman
import os, sys
os.environ['AC_CODER_DIR'] = os.path.abspath('Python-Audio-Coder')
sys.path.insert(0, os.path.abspath('starter_kit'))
import ac_kit as k
TESTSET = k.prepare_test_set()     # EBU SQAM (175.5 MB download), or the fallback set with a warning
```

## Working on your own computer (VS Code, Jupyter, terminal)

Colab is the tested default, but the kit does not depend on it. You can work locally instead of Colab, or next to it (same repository, same results files).

1. Install Python 3.10 or newer and git.
2. In your project folder:
   ```
   python -m venv .venv            (then activate it: source .venv/bin/activate, on Windows .venv\Scripts\activate)
   pip install numpy scipy matplotlib pandas soundfile dahuffman jupyter
   git clone https://github.com/TUIlmenauAMS/ac-ws2627-starter-kit starter_kit     (or unzip starter_kit.zip)
   (then delete the folder starter_kit/.git, so that the kit's files become part of your own repository)
   python starter_kit/setup_topic.py --topic 1      (your topic number)
   ```
3. Open `starter_kit/starter_kit_demo.ipynb` in VS Code (Python and Jupyter extensions) or with `jupyter notebook`, started from that project folder, and run all cells. It must end with "setup matches the reference", exactly as in Colab.

Notes: the first cell works unchanged (it skips the upload step when the kit folder is already there). Topic 15 also needs PyTorch and ffmpeg. The kit was tested on Linux and in Colab, not on Windows or macOS: if something fails there, post the error message in the forum. Your results must not depend on your computer: the reference check tells you if they do.

## Versions of the course code

`setup_topic.py` clones the course repositories at fixed commits, so every group works with the same code,
also if a repository changes during the semester. In November 2026 a few small bugs were fixed in
Python-Audio-Coder (Colab-safe `sound.py`, the window in `psyac_quantization.py`, the `psyacloss.py` import,
`lpcexample.py`, the optimization examples, `LDFB.py`, mono files in `psyacpostfilterFromFile.py`) and in
FilterBanks_FastPythonImplementation_Classes (`LDFBclass` now reconstructs perfectly). The commits the kit uses
include these fixes: do not use a clone from before November 2026.

The cloned folders have no `.git` folder of their own: a small file `.kit_source` in each records the repository
and commit. So you commit them to your group's repository as ordinary files, together with your changes. (A folder
with its own `.git` would be stored only as a link, and your changes to it would be missing.)

## Your group's GitLab repository

The demo notebook's GitLab cell ("Your group's GitLab repository") does this for you (`gitlab_repo.py`). Give it your repository's https address, your
name and e-mail, and a GitLab access token (user settings → Access tokens, scopes `read_repository` and
`write_repository`; kept only for the session, never committed).

- **First time** (one member, after the baseline cell): it copies `starter_kit/`, the course code the kit set up and
  `results/` into the repository, adds a `.gitignore` (the SQAM test recordings, downloads and caches are never
  committed), commits, pushes and prints the commit hash. With `results/baseline.csv` in it, that is the baseline
  commit for your proposal.
- **Every later session** (every member): run the notebook's first cell, then the GitLab cell. It clones the
  repository and switches into it: working folder, kit and coder come from your repository, and the test set is
  prepared there.
- **Saving your work** (before you close Colab): `gitlab_repo.save("what you changed")`. It gets your teammates'
  latest changes, commits yours and pushes. If a teammate changed the same lines, main is left as it is and your
  version is pushed to a branch of its own (`save-<name>-<date>`): decide together which to keep.
- **Changing a file in Colab:** Files panel (folder icon on the left) → double-click the file; it is saved automatically.

On your own computer, clone your repository with your usual git login and work in it (`gitlab_repo.use_repository(url, token=None, ...)` does the same as the notebook cell).

## Measurement rules (use them in your paper)

- **Bitrate** = size of the compressed file in bits ÷ number of audio samples (all channels): `k.bits_per_sample(file, k.num_samples(x))`.
  Never the bitrate that `audio_encoder.py` prints: it counts the number of Huffman table entries, not their size in bytes.
- **Quality** = SNR (dB, higher is better) and `percloss` (lower is better), after removing the coder delay: `k.evaluate(x, y, fs, delay=1024)`.
  Report both. Always measure with the kit's `percloss`, never with a model you changed.
- **Rate–quality curves**: the coder has no target-bitrate setting; sweep its `quality` parameter. At least 4 operating points per method; quality 50, 100, 200, 400 gives about 1.3–3.6 bit/sample.
- **Delay**: the course MDCT coder delays its output by N = 1024 samples. If you change N or the filter bank, pass the new delay and block length: `k.quality_sweep(..., delay=N, block=N)`. Derive the delay, then check it with `k.find_delay(x, y)`.
- **Clips**: choose your settings on the `tuning` clips, report results on the `evaluation` clips (`k.load_manifest(split='evaluation')`).
- **Test set**: report its id (`sqam-v1`, or `librosa-v1` if you had to use the fallback). Every results row records it; results on different test sets are not comparable. Every row also records the kit version (column `kit`, `k.KIT_VERSION`).
- **Results files**: keep the CSV that `quality_sweep` writes next to every figure.

## Things to know about the coder and the measures

- **Bitrate floor.** The coder Huffman-codes every MDCT coefficient with at least 1 bit and adds about 0.4 bit/sample for the scale factors, so it never goes much below about 1.4 bit/sample, even at very low quality. Differences between methods show up above that floor.
- **Partial last block.** The coder only codes complete blocks of N samples and returns garbage for a partial last block. `evaluate()` compares only complete blocks; the test clips are whole blocks already.
- **Clip edges.** `evaluate()` leaves out the first and last 1024 samples: the first and last analysis frame of `percloss` see the fade and zero padding at the clip edges, where any coder noise would otherwise dominate the value.
- **`percloss` and silence.** `percloss` divides the error spectrum by the masking threshold of the original. Where the original is (digitally) silent, the threshold is almost zero and `percloss` explodes. The test clips avoid this: SQAM excerpts are taken from passages without digital silence (SQAM has digital silence between phrases), and the fallback snare clips have a noise floor 60 dB below the peak. If you add your own clips, avoid digital silence. Speech and transient clips give higher `percloss` values than music (pauses and decays): compare methods on the same clip, not values across clips.
- **SQAM rules.** Non-commercial research and teaching only, and no redistribution: never commit the clips to your repository; your README points to `prepare_test_set()` instead.
- **Fallback sources.** The fallback clips come from Ogg Vorbis files (github.com/librosa/data), so they are already lossy-coded, and their speech is band-limited to 8 kHz. Say so if you had to use them.
- **IntMDCT lossless coder** (`lossless_rice_audio_*`) needs a **stereo** file. For mono clips, duplicate the channel.

## Pre-echo (Topics 2 and 4)

A block transform spreads the quantization noise of a block over the whole block, so with long blocks noise
appears *before* an attack (a drum hit, castanets). The kit measures it the same way for everyone:

```python
fs, x = k.read_wav(clip); fs, y = k.read_wav(out['decoded'])
r = k.pre_echo_db(x, y, fs, delay=N)          # {'pre_echo_db': mean, 'pre_echo_onsets': count, 'per_onset_db': [...], ...}
k.quality_sweep(clips, delay=N, block=N, coder_dir='coder_N256', extra_metrics=[k.pre_echo_db])   # adds two columns
```

Definition (use it in your paper):
1. **Onsets** are found in the *original* with `k.find_onsets(x, fs)`: 2-ms frames; an onset is a frame whose energy
   is more than 20 dB above the lowest frame energy in the 5 ms before it (and not more than 60 dB below the loudest
   frame); onsets closer than 100 ms are merged. Stereo is mixed to mono for this. Accuracy: about one frame (2 ms).
2. For every onset at sample n0: the energy of the coding **error** (decoded − original, after removing the delay)
   in the 20 ms that end 1 ms before n0, divided by the energy of the **original** in the 20 ms from n0 on, in dB.
3. `pre_echo_db` is the mean of these values over all onsets. Lower (more negative) is better.

Example (fallback snare clip, quality 200): N = 128: −41.1 dB, N = 256: −33.9 dB, N = 1024: −23.1 dB,
N = 2048: −20.8 dB. Measure on at least 3 transient clips and compare at similar SNR or bitrate.
With `extra_metrics`, the new columns come after the standard ones; the standard columns do not change.

## Lossless topics (6–9)

The two lossless coders of the course do not reconstruct the end of the file, so check where exact reconstruction stops and that everything before it is exact:

```python
r = k.lossless_check(x, y, delay=1024)            # IntMDCT coder (delay 1024); predictive coder: delay=0
n = r['first_mismatch'] if r['first_mismatch'] is not None else r['compared']
assert k.lossless_check(x, y, delay=1024, trim_end=r['compared'] - n)['exact']
print('exact for the first', n, 'samples; not reconstructed at the end:', len(x) - n)
```

On the 6-s test clips: the IntMDCT coder loses the last 4096 samples (4 blocks), the predictive coder the last 484 (the remainder of its 20-ms blocks plus 10 samples). Report this number, as the topic descriptions ask.

## Repeated coding (Topic 10)

```python
cur = k.load_manifest(classes=['polyphonic'], split='evaluation')[0]['path']; fs, x0 = k.read_wav(cur)
for g in range(1, 11):
    out = k.run_coder(cur, 100, workdir='gen%d' % g)
    fs, y = k.read_wav(out['decoded'])
    y = y[1024:1024 + len(x0)]          # remove this generation's delay before the next one
    cur = 'gen%d/next.wav' % g
    import scipy.io.wavfile as w, numpy as np; w.write(cur, fs, np.int16(np.clip(np.round(y), -32768, 32767)))
    print(g, k.evaluate(x0, y, fs, delay=0))
```

## Your own modified coder

Copy `Python-Audio-Coder` to a folder of your own, change it there, and pass that folder:
`k.quality_sweep(clips, out_csv='results.csv', method='KBD beta=4', coder_dir='my_coder')`.
If your encoder takes extra command-line arguments, pass `extra_args=[...]`.

Tested on 18 Nov 2026 with Python 3.13, NumPy 2.5 and SciPy 1.18 (`tests/run_checks.py` with the fallback test set; the SQAM test set on 15 Nov), at the commits in `setup_topic.py`.

## Licence

The kit's code is under the MIT licence (`LICENSE`). The clips in `testset_fallback/` have their own licences
(CC BY 4.0 or public domain, see `testset_fallback/ATTRIBUTION.md`). The EBU SQAM recordings are not part of the
kit: each user downloads them from the EBU (`sqam_test_set.py`).
