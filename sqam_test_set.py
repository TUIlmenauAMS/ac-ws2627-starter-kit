"""Shared test set from EBU SQAM (Tech 3253), the standard material for audio-coder evaluation.

The SQAM recordings may be used for non-commercial R&D ("By downloading this file you agree not
to use this material for commercial purposes other than as a R&D tool", qc.ebu.io/testmaterials/523/),
but they are not licensed for redistribution. So the kit does not contain them: every user
downloads the zip from the EBU (175.5 MB) and this script cuts the same clips from it, checked
against SHA-256 hashes, so everyone gets bit-identical clips.

Usage:  python sqam_test_set.py [sqam zip] [output folder]
        (without a zip path it downloads it from the EBU)
Python: build_sqam_test_set(zip_path=None, out_dir='testset')
"""

import csv
import hashlib
import io
import json
import os
import sys
import urllib.request
import zipfile

import numpy as np
import scipy.io.wavfile as wavfile

SQAM_URL = "https://qc.ebu.io/testmaterials/523/1/download/"
SQAM_PAGE = "https://qc.ebu.io/testmaterials/523/"
SQAM_SHA256 = "7d6fcd0fc42354637291792534b61bf129612f221f8efef97b62e8942a8686aa"
SQAM_BYTES = 175545976
TESTSET_ID = "sqam-v1"
FS = 44100
BLOCK = 1024
HASH_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sqam_clip_hashes.json")

LICENCE = "EBU SQAM, non-commercial R&D use only; not for redistribution"
# file, class, split, SQAM track, excerpt start (s), mode, description (EBU Tech 3253 track list)
# Excerpts: 6 s (whole 1024-sample blocks) inside a passage without digital silence.
CLIPS = [
    ("speech_1.wav", "speech", "tuning", 49, 0.56, "mono", "Female speech, English"),
    ("speech_2.wav", "speech", "evaluation", 50, 0.56, "mono", "Male speech, English"),
    ("speech_3.wav", "speech", "evaluation", 53, 9.59, "mono", "Female speech, German"),
    ("tonal_trumpet.wav", "tonal", "tuning", 21, 23.41, "stereo", "Trumpet, melodious phrase"),
    ("tonal_glockenspiel.wav", "tonal", "evaluation", 35, 35.99, "stereo", "Glockenspiel, melodious phrase"),
    ("tonal_soprano.wav", "tonal", "evaluation", 44, 0.56, "stereo", "Vocal, soprano"),
    ("transient_claves.wav", "transient", "tuning", 26, 18.58, "stereo", "Claves, single tone rhythm"),
    ("transient_castanets.wav", "transient", "evaluation", 27, 0.56, "stereo", "Castanets, single tone rhythm"),
    ("transient_sidedrum.wav", "transient", "evaluation", 28, 54.54, "stereo", "Side drum, with snares roll"),
    ("poly_wind.wav", "polyphonic", "tuning", 66, 0.51, "stereo", "Wind ensemble, Stravinsky"),
    ("poly_orchestra.wav", "polyphonic", "evaluation", 65, 29.98, "stereo", "Orchestra, R. Strauss"),
    ("poly_pop.wav", "polyphonic", "evaluation", 69, 3.00, "stereo", "Pop, ABBA"),
    ("poly_choir.wav", "polyphonic", "evaluation", 64, 0.53, "stereo", "Choir, Orff"),
    ("noise_pink.wav", "noise", "tuning", 2, 2.00, "stereo", "Band-limited pink noise"),
    ("noise_cymbal.wav", "noise", "evaluation", 31, 67.99, "stereo", "Cymbal, soft stick roll"),
    ("stereo_panned.wav", "stereo", "tuning", 65, 40.00, "panned", "Orchestra, R. Strauss: mono mix panned (left 1.0, right 0.6)"),
    ("stereo_phase.wav", "stereo", "evaluation", 65, 50.00, "phase", "Orchestra, R. Strauss: mono mix, right channel delayed by 1 ms"),
]


def download_sqam(path="TECH3253_SQAM_FLAC.zip"):
    """Download the SQAM zip from the EBU (175.5 MB) unless a verified copy exists. Returns the path."""
    if os.path.isfile(path) and sha256(path) == SQAM_SHA256:
        return path
    print("downloading EBU SQAM (175.5 MB) from", SQAM_URL)
    print("by downloading you accept the EBU condition: non-commercial R&D use only")
    req = urllib.request.Request(SQAM_URL, headers={"User-Agent": "Mozilla/5.0 (audio coding seminar starter kit)"})
    with urllib.request.urlopen(req, timeout=300) as r, open(path + ".part", "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    os.replace(path + ".part", path)
    if sha256(path) != SQAM_SHA256:
        raise RuntimeError("downloaded SQAM zip has an unexpected SHA-256 (the EBU file may have changed)")
    return path


def sha256(path=None, data=None):
    h = hashlib.sha256()
    if data is not None:
        h.update(data)
    else:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()


def _read_track(zf, track):
    import soundfile as sf
    x, fs = sf.read(io.BytesIO(zf.read("%02d.flac" % track)), dtype="float64", always_2d=True)
    if fs != FS:
        raise ValueError("SQAM track %d has %d Hz" % (track, fs))
    return x


def _finish(x):
    """Whole 1024-sample blocks, 5 ms fades, -1 dBFS peak, 16 bit (as for the fallback set)."""
    n = len(x) // BLOCK * BLOCK
    x = x[:n]
    k = int(0.005 * FS)
    fade = np.ones(n)
    ramp = 0.5 - 0.5 * np.cos(np.pi * np.arange(k) / k)
    fade[:k], fade[-k:] = ramp, ramp[::-1]
    x = x * (fade[:, None] if x.ndim == 2 else fade)
    x = x / np.max(np.abs(x)) * 10 ** (-1 / 20)
    return np.int16(np.round(x * 32767))


def make_clip(zf, track, start, mode, seconds=6.0):
    x = _read_track(zf, track)
    n = int(seconds * FS) // BLOCK * BLOCK
    s = int(round(start * FS))
    x = x[s:s + n]
    if mode == "stereo":
        return x
    m = x.mean(axis=1)
    if mode == "mono":          # SQAM speech is mono, stored as two identical channels
        return m
    if mode == "panned":
        return np.stack((1.0 * m, 0.6 * m), axis=1)
    if mode == "phase":
        d = int(round(0.001 * FS))
        return np.stack((m, np.concatenate((np.zeros(d), m[:-d]))), axis=1)
    raise ValueError(mode)


def build_sqam_test_set(zip_path=None, out_dir="testset", check_hashes=True):
    """Cut the shared clips from the SQAM zip into out_dir (manifest.csv, ATTRIBUTION.md).
    Returns the test-set id. With check_hashes, every clip must match the pinned SHA-256."""
    zip_path = zip_path or download_sqam()
    if sha256(zip_path) != SQAM_SHA256:
        raise RuntimeError("%s is not the expected SQAM zip (SHA-256 differs)" % zip_path)
    pinned = json.load(open(HASH_FILE)) if (check_hashes and os.path.isfile(HASH_FILE)) else None
    os.makedirs(out_dir, exist_ok=True)
    rows, hashes = [], {}
    with zipfile.ZipFile(zip_path) as zf:
        for name, cls, split, track, start, mode, desc in CLIPS:
            x = _finish(make_clip(zf, track, start, mode))
            buf = io.BytesIO()
            wavfile.write(buf, FS, x)
            data = buf.getvalue()
            hashes[name] = sha256(data=data)
            if pinned is not None and pinned.get(name) != hashes[name]:
                raise RuntimeError("clip %s differs from the reference clip (SHA-256)" % name)
            with open(os.path.join(out_dir, name), "wb") as f:
                f.write(data)
            ch = 1 if x.ndim == 1 else x.shape[1]
            corr = "" if ch == 1 else "%.2f" % np.corrcoef(x[:, 0].astype(float), x[:, 1].astype(float))[0, 1]
            rows.append({"file": name, "class": cls, "split": split, "channels": ch, "fs": FS,
                         "duration_s": round(len(x) / FS, 2), "lr_correlation": corr,
                         "source": "SQAM track %02d" % track, "excerpt_start_s": start,
                         "licence": LICENCE, "attribution": "EBU SQAM (Tech 3253), track %02d: %s" % (track, desc),
                         "testset": TESTSET_ID})
    with open(os.path.join(out_dir, "manifest.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(out_dir, "ATTRIBUTION.md"), "w") as f:
        f.write("# Test set %s: EBU SQAM\n\nExcerpts of the EBU Sound Quality Assessment Material (EBU Tech 3253), "
                "downloaded from %s. Use: non-commercial R&D only; do not redistribute these files "
                "(do not put them in your GitLab repository: add this download step instead).\n"
                "Each clip: 6-s excerpt of whole 1024-sample blocks, 5 ms fades, normalized to -1 dBFS peak, 16 bit.\n\n"
                % (TESTSET_ID, SQAM_PAGE))
        for r in rows:
            f.write("- **%s** (%s, %s): %s.\n" % (r["file"], r["class"], r["split"], r["attribution"]))
    return TESTSET_ID, hashes


if __name__ == "__main__":
    z = sys.argv[1] if len(sys.argv) > 1 else None
    out = sys.argv[2] if len(sys.argv) > 2 else "testset"
    tid, h = build_sqam_test_set(z, out, check_hashes="--write-hashes" not in sys.argv)
    if "--write-hashes" in sys.argv:
        json.dump(h, open(HASH_FILE, "w"), indent=1)
    print("built", tid, "in", out)
