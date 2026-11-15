"""Builds the FALLBACK test set of the Audio Coding seminar (WS 2026/27), id 'librosa-v1'.

The main test set is EBU SQAM (sqam_test_set.py). This one is shipped with the kit and used only
when SQAM cannot be downloaded; ac_kit.prepare_test_set() chooses.

Sources: example recordings of the librosa project (github.com/librosa/data), only clips under
CC BY 4.0 or public domain (no non-commercial licences), plus synthetic signals (no licence needed).
Every clip is cut to a short excerpt of whole 1024-sample blocks, resampled to 44.1 kHz, faded in/out over 5 ms,
normalized to -1 dBFS peak and stored as 16-bit WAV. The two snare clips get a pink noise
floor 60 dB below their peak, because the recording has exact digital silence between hits. testset/manifest.csv lists class,
tuning/evaluation split, licence and attribution; testset/ATTRIBUTION.md is the credit list
that CC BY requires.

Note: the librosa sources are Ogg Vorbis files, i.e. already lossy-coded. That is fine for
lossless coding and for comparisons inside the course coder, but mention it in your paper.
A lossless test set (e.g. EBU SQAM) can replace this one later with the same manifest format.

Usage:  python make_test_set.py [path/to/librosa-data/audio] [output folder]
        (without a path, it clones https://github.com/librosa/data into ./librosa_data)
"""

import csv
import os
import subprocess
import sys

import numpy as np
import scipy.io.wavfile as wavfile
from scipy.signal import resample_poly

FS = 44100
FADE_S = 0.005

LIBRI = "LibriSpeech SLR12 (via librosa/data), https://www.openslr.org/12/"
# file, class, split, source file, start s, length s, mode, licence, attribution
CLIPS = [
    ("speech_1.wav", "speech", "tuning", "198-209-0000.hq.ogg", 0.0, 6.0, "mono", "CC BY 4.0",
     "Jane Austen, 'Sense and Sensibility' ch. 18, read by Heather Barnett; " + LIBRI),
    ("speech_2.wav", "speech", "evaluation", "5703-47212-0000.hq.ogg", 0.0, 6.0, "mono", "CC BY 4.0",
     "Marion Bryce, 'The Ashiel Mystery' ch. 16, read by Garth Comira; " + LIBRI),
    ("speech_3.wav", "speech", "evaluation", "3436-172162-0000.hq.ogg", 3.0, 6.0, "mono", "CC BY 4.0",
     "Thomas Bulfinch, 'The Age of Chivalry' ch. 9, read by Anders Lankford; " + LIBRI),
    ("tonal_trumpet.wav", "tonal", "tuning", "sorohanro_-_solo-trumpet-06.hq.ogg", 0.0, 3.2, "stereo",
     "CC BY 4.0", "Mihai Sorohan, 'solo trumpet 06', https://freesound.org/s/77711/"),
    ("tonal_robin.wav", "tonal", "evaluation", "456440__inspectorj__bird-whistling-robin-single-13.hq.ogg",
     0.0, 2.7, "stereo", "CC BY 4.0",
     "InspectorJ, 'Bird Whistling, Robin, Single, 13', https://freesound.org/people/InspectorJ/sounds/456440/"),
    ("transient_snare_sparse.wav", "transient", "tuning", "snare-accelerate.hq.ogg", 0.0, 6.0, "mono+floor",
     "CC BY 4.0", "Brian McFee, 'Accelerating snare drum' (snare sample by sakebeats, "
     "https://freesound.org/people/sakebeats/sounds/176345/), via librosa/data; synthetic pink noise floor added at -60 dB"),
    ("transient_snare_dense.wav", "transient", "evaluation", "snare-accelerate.hq.ogg", 24.0, 6.0, "mono+floor",
     "CC BY 4.0", "Brian McFee, 'Accelerating snare drum' (snare sample by sakebeats, "
     "https://freesound.org/people/sakebeats/sounds/176345/), via librosa/data; synthetic pink noise floor added at -60 dB"),
    ("poly_strings.wav", "polyphonic", "tuning",
     "Hungarian_Dance_number_5_-_Allegro_in_F_sharp_minor_(string_orchestra).hq.ogg", 0.0, 6.0, "stereo",
     "Public domain (CC PDM 1.0)", "J. Brahms, Hungarian Dance No. 5, performed by the US Army Strings, "
     "https://musopen.org/music/43805-hungarian-dance-no-5-in-f-sharp-minor-woo-1-string-orch"),
    ("poly_orchestra.wav", "polyphonic", "evaluation",
     "Kevin_MacLeod_-_P_I_Tchaikovsky_Dance_of_the_Sugar_Plum_Fairy.hq.ogg", 36.0, 6.0, "stereo", "CC BY 4.0",
     "P. I. Tchaikovsky, 'Dance of the Sugar Plum Fairy', arranged by Kevin MacLeod (incompetech.com), "
     "https://freemusicarchive.org/music/Kevin_MacLeod/Classical_Sampler/Dance_of_the_Sugar_Plum_Fairy"),
    ("poly_jazz.wav", "polyphonic", "evaluation", "Kevin_MacLeod_-_Vibe_Ace.hq.ogg", 14.0, 6.0, "stereo",
     "CC BY 4.0", "Kevin MacLeod, 'Vibe Ace' (incompetech.com), "
     "https://freemusicarchive.org/music/Kevin_MacLeod/Jazz_Sampler/Vibe_Ace"),
    ("noise_pink.wav", "noise", "tuning", None, 0.0, 6.0, "pink", "none (synthetic)",
     "Synthetic pink noise, independent left/right channels (seed 1)"),
    ("noise_underwater.wav", "noise", "evaluation", "glacier-bay-humpback.hq.ogg", 10.0, 6.0, "mono",
     "Public domain (CC PDM 1.0)", "Humpback whale song with underwater background noise, recorded by "
     "NPS - Chris Gabriele, Glacier Bay National Park, https://www.nps.gov/glba/learn/nature/soundclips.htm"),
    ("stereo_panned.wav", "stereo", "tuning",
     "Hungarian_Dance_number_5_-_Allegro_in_F_sharp_minor_(string_orchestra).hq.ogg", 6.0, 6.0, "panned",
     "Public domain (CC PDM 1.0)", "Derived from the Brahms recording above: mono mix panned "
     "(left 1.0, right 0.6), strongly correlated channels"),
    ("stereo_phase.wav", "stereo", "evaluation",
     "Hungarian_Dance_number_5_-_Allegro_in_F_sharp_minor_(string_orchestra).hq.ogg", 12.0, 6.0, "phase",
     "Public domain (CC PDM 1.0)", "Derived from the Brahms recording above: mono mix, right channel "
     "delayed by 1 ms (44 samples), phase-shifted channels"),
]


def _load(path, start, length):
    import soundfile as sf
    x, fs = sf.read(path, always_2d=True)
    x = x[int(start * fs): int((start + length) * fs)]
    if fs != FS:
        g = np.gcd(FS, fs)
        x = resample_poly(x, FS // g, fs // g, axis=0)
    return x


def make_clip(src_dir, src, start, length, mode):
    if mode == "pink":
        rng = np.random.default_rng(1)
        n = int(length * FS)
        white = rng.standard_normal((n, 2))
        f = np.fft.rfftfreq(n, 1 / FS)
        shape = np.zeros_like(f)
        shape[f >= 20] = 1 / np.sqrt(f[f >= 20])
        return np.fft.irfft(np.fft.rfft(white, axis=0) * shape[:, None], n, axis=0)
    x = _load(os.path.join(src_dir, src), start, length)
    if mode == "mono":
        return x.mean(axis=1)
    if mode == "mono+floor":
        # the snare recording has exact digital silence between the hits, where percloss
        # (error / masking threshold of the original) explodes; real recordings always have a
        # noise floor, so a pink noise floor 60 dB below the peak is added
        m = x.mean(axis=1)
        m = m / np.max(np.abs(m))
        noise = make_clip(None, None, 0.0, len(m) / FS + 0.1, "pink")[: len(m), 0]
        return m + noise / np.sqrt(np.mean(noise ** 2)) * 10 ** (-60 / 20)
    if mode == "stereo":
        return x
    m = x.mean(axis=1)
    if mode == "panned":
        return np.stack((1.0 * m, 0.6 * m), axis=1)
    if mode == "phase":
        d = int(round(0.001 * FS))
        return np.stack((m, np.concatenate((np.zeros(d), m[:-d]))), axis=1)
    raise ValueError(mode)


def finish(x):
    # whole blocks of 1024 samples: the course coder only codes complete blocks and
    # returns garbage for a partial last block
    n = len(x) // 1024 * 1024
    x = x[:n]
    k = int(FADE_S * FS)
    fade = np.ones(n)
    ramp = 0.5 - 0.5 * np.cos(np.pi * np.arange(k) / k)
    fade[:k], fade[-k:] = ramp, ramp[::-1]
    x = x * (fade[:, None] if x.ndim == 2 else fade)
    x = x / np.max(np.abs(x)) * 10 ** (-1 / 20)
    return np.int16(np.round(x * 32767))


def main():
    src_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join("librosa_data", "audio")
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "testset_fallback"
    if not os.path.isdir(src_dir):
        subprocess.run(["git", "clone", "-q", "--depth", "1", "https://github.com/librosa/data", "librosa_data"],
                       check=True)
        src_dir = os.path.join("librosa_data", "audio")
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for name, cls, split, src, start, length, mode, lic, attr in CLIPS:
        x = finish(make_clip(src_dir, src, start, length, mode))
        wavfile.write(os.path.join(out_dir, name), FS, x)
        ch = 1 if x.ndim == 1 else x.shape[1]
        corr = "" if ch == 1 else "%.2f" % np.corrcoef(x[:, 0].astype(float), x[:, 1].astype(float))[0, 1]
        rows.append({"file": name, "class": cls, "split": split, "channels": ch, "fs": FS,
                     "duration_s": round(len(x) / FS, 2), "lr_correlation": corr,
                     "source": src or "synthetic", "excerpt_start_s": start, "licence": lic, "attribution": attr,
                     "testset": "librosa-v1"})
        print("%-28s %-10s %-10s ch=%d %5.2f s %s" % (name, cls, split, ch, len(x) / FS, corr))
    with open(os.path.join(out_dir, "manifest.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(out_dir, "ATTRIBUTION.md"), "w") as f:
        f.write("# Test set: sources and licences\n\n"
                "All clips are excerpts, resampled to 44.1 kHz, faded in/out (5 ms), normalized to -1 dBFS peak "
                "and decoded from Ogg Vorbis sources of github.com/librosa/data. Licences: CC BY 4.0 "
                "(https://creativecommons.org/licenses/by/4.0/) or public domain, as listed.\n\n")
        for r in rows:
            f.write("- **%s** (%s): %s. Licence: %s.\n" % (r["file"], r["class"], r["attribution"], r["licence"]))


if __name__ == "__main__":
    main()
