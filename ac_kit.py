"""Audio Coding seminar WS 2026/27 -- shared measurement starter kit.

Every group measures bitrate and quality the same way with these functions:

    bits_per_sample(compressed_file, num_samples)   bitrate from the actual file size
    align(orig, decoded, delay)                     remove the coder delay, cut to equal length
    find_delay(orig, decoded)                       estimate a delay (to check one you derived)
    snr_db(orig, decoded)                           signal-to-noise ratio in dB
    percloss_value(orig, decoded, fs)               psychoacoustic loss (percloss from psyacloss.py)
    evaluate(orig, decoded, fs, delay)              both quality measures after alignment
    lossless_check(orig, decoded, delay, trim_end)  sample-exact check for lossless coders
    find_onsets(x, fs)                              transient onsets of a signal (sample positions)
    pre_echo_db(orig, decoded, fs, delay)           coding error just before onsets (Topics 2 and 4)
    run_coder(wav, quality)                         encode + decode one file with the course coder
    quality_sweep(clips, qualities, out_csv)        rate-quality points for several clips -> CSV
                                                    (extra_metrics=[pre_echo_db] adds columns)
    plot_rate_quality(csv_file)                     rate-quality curves from that CSV

Conventions (use them in your paper):
  * Bitrate = size of the compressed file in bits / number of audio samples, counting the
    samples of all channels (bits per sample). Do NOT use the bitrate audio_encoder.py prints:
    it counts the number of Huffman table entries instead of their size in bytes.
  * The course MDCT coder (N = 1024) delays its output by N = 1024 samples. align() removes it.
  * Quality = SNR (dB, higher is better) and percloss (lower is better), both computed on
    the aligned signals, all channels together, over complete 1024-sample blocks, without
    the first and last block (clip edges).
  * Test set: EBU SQAM excerpts built by prepare_test_set() (default), or the fallback set
    shipped with the kit. Every results row records which one ('testset' column).
  * percloss divides the error spectrum by the masking threshold of the original. Where the
    original is (near) silent, the threshold is close to zero and percloss explodes: keep test
    clips free of digital silence, and always report SNR next to it.
"""

import csv
import os
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
import scipy.io.wavfile as wavfile

KIT_DIR = os.path.dirname(os.path.abspath(__file__))
KIT_VERSION = "2026-11-18"   # printed by the notebooks and written to the 'kit' column of every results row


def _coder_dir(coder_dir=None):
    """Folder of the coder: the argument, else $AC_CODER_DIR (read at call time), else ./Python-Audio-Coder."""
    return os.path.abspath(coder_dir or os.environ.get("AC_CODER_DIR", "Python-Audio-Coder"))


CODER_DELAY = 1024  # samples, course MDCT coder with N = 1024


# ---------------------------------------------------------------- reading and bitrate

def read_wav(path):
    """Read a WAV file; returns (fs, x) with x as float64, shape (samples,) or (samples, channels)."""
    fs, x = wavfile.read(path)
    return fs, x.astype(np.float64)


def num_samples(x):
    """Number of audio samples over all channels."""
    return int(np.prod(np.shape(x)))


def bits_per_sample(compressed_file, num_samples):
    """Bitrate from the actual file size: bits in the compressed file / audio samples (all channels)."""
    return 8.0 * os.path.getsize(compressed_file) / num_samples


def bitrate_kbps(compressed_file, duration_s):
    """Bitrate in kbit/s from the actual file size."""
    return 8.0 * os.path.getsize(compressed_file) / duration_s / 1000.0


# ---------------------------------------------------------------- alignment and quality

def align(orig, decoded, delay=CODER_DELAY):
    """Remove `delay` samples from the start of `decoded` and cut both to the same length."""
    orig = np.asarray(orig, dtype=np.float64)
    decoded = np.asarray(decoded, dtype=np.float64)
    if orig.shape[1:] != decoded.shape[1:]:
        raise ValueError("original and decoded signal have different channel layouts: %s vs %s"
                         % (orig.shape, decoded.shape))
    decoded = decoded[delay:]
    n = min(len(orig), len(decoded))
    if n == 0:
        raise ValueError("nothing to compare: the decoded signal has only %d samples after removing "
                         "the delay of %d" % (len(decoded), delay))
    return orig[:n], decoded[:n]


def find_delay(orig, decoded, max_delay=4096):
    """Delay (in samples) of `decoded` relative to `orig`, found by cross-correlation.

    Use it to check the delay you derived from theory; report the derived one.
    """
    o = np.asarray(orig, dtype=np.float64)
    d = np.asarray(decoded, dtype=np.float64)
    if o.ndim > 1:
        o, d = o.sum(axis=1), d.sum(axis=1)
    n = min(len(o), len(d) - max_delay)
    if n <= 0:
        raise ValueError("decoded signal is shorter than max_delay")
    o = o[:n]
    nfft = 1 << int(np.ceil(np.log2(n + len(d))))
    corr = np.fft.irfft(np.fft.rfft(d, nfft) * np.conj(np.fft.rfft(o, nfft)), nfft)
    return int(np.argmax(corr[: max_delay + 1]))


def snr_db(orig, decoded):
    """SNR in dB of `decoded` against `orig` (aligned, same length), all channels together."""
    orig = np.asarray(orig, dtype=np.float64)
    err = orig - np.asarray(decoded, dtype=np.float64)
    e = np.sum(err ** 2)
    if e == 0:
        return float("inf")
    return float(10 * np.log10(np.sum(orig ** 2) / e))


def percloss_value(orig, decoded, fs):
    """Psychoacoustic loss percloss() from psyacloss.py (lower is better), unmodified model."""
    if KIT_DIR not in sys.path:
        sys.path.insert(0, KIT_DIR)
    from psyacloss import percloss  # the kit's fixed copy
    return float(percloss(np.asarray(orig, dtype=np.float64), np.asarray(decoded, dtype=np.float64), fs))


def evaluate(orig, decoded, fs, delay=CODER_DELAY, block=1024, edge=1024):
    """Align, then return {'snr_db': ..., 'percloss': ..., 'compared': samples}.

    Only complete blocks of `block` samples of the original are compared: the course coder
    does not reconstruct a partial last block, and those few garbage samples would otherwise
    dominate percloss (the shared test clips are whole blocks already). block=None compares all.
    `edge` samples at both ends are left out: the first and last analysis frame of percloss
    see the clip's fade and zero padding, where the masking threshold is almost zero, so any
    coder noise there would dominate the whole value (up to 10^6 times the rest).
    """
    orig = np.asarray(orig, dtype=np.float64)
    if block:
        orig = orig[: len(orig) // block * block]
    o, d = align(orig, decoded, delay)
    if edge and len(o) > 4 * edge:
        o, d = o[edge:-edge], d[edge:-edge]
    return {"snr_db": snr_db(o, d), "percloss": percloss_value(o, d, fs), "compared": len(o)}


def lossless_check(orig, decoded, delay=0, trim_end=0):
    """Sample-exact check for lossless coders.

    delay: samples to drop from the start of `decoded` (IntMDCT coder: 1024).
    trim_end: samples at the end that the coder does not reconstruct (report this number).
    Returns {'exact': bool, 'compared': n, 'not_compared': samples of the original left out,
             'mismatches': k, 'first_mismatch': i or None, 'max_abs_diff': m}.
    'exact' only covers the compared samples: always report 'not_compared' with it.
    """
    n_orig = len(orig)
    o, d = align(orig, decoded, delay)
    if trim_end:
        if trim_end >= len(o):
            raise ValueError("trim_end=%d leaves nothing to compare (%d samples overlap)" % (trim_end, len(o)))
        o, d = o[:-trim_end], d[:-trim_end]
    diff = np.abs(np.rint(o) - np.rint(d))
    if diff.ndim > 1:
        diff = diff.max(axis=1)
    bad = np.flatnonzero(diff > 0)
    return {"exact": len(bad) == 0, "compared": len(o), "not_compared": n_orig - len(o), "mismatches": int(len(bad)),
            "first_mismatch": int(bad[0]) if len(bad) else None,
            "max_abs_diff": float(diff.max()) if len(diff) else 0.0}


# ---------------------------------------------------------------- pre-echo (Topics 2 and 4)

def _mono(x):
    x = np.asarray(x, dtype=np.float64)
    return x.mean(axis=1) if x.ndim > 1 else x


def find_onsets(x, fs, min_gap_ms=100, threshold_db=20, frame_ms=2.0, rise_ms=5.0, floor_db=-60):
    """Positions (in samples) of transient onsets in a signal, e.g. drum hits or castanet clicks.

    How it works:
      1. Stereo is mixed to mono (mean of the channels).
      2. The signal is cut into frames of `frame_ms` (2 ms) without overlap, and the energy of
         each frame is computed in dB.
      3. A frame is an onset if its energy is more than `threshold_db` (20 dB) above the
         lowest frame energy in the `rise_ms` (5 ms) before it, and not more than |floor_db|
         (60 dB) below the loudest frame of the signal (this ignores tiny changes in near-silence).
      4. Onsets closer than `min_gap_ms` (100 ms) to the previous onset are merged into it
         (only the first one is kept).
    The returned position is the first sample of the onset frame, so it is accurate to about
    one frame (2 ms). Use it on the ORIGINAL signal, never on the decoded one.
    Returns a NumPy array of sample positions (int), sorted.
    """
    m = _mono(x)
    flen = max(1, int(round(frame_ms * 1e-3 * fs)))
    nfr = len(m) // flen
    if nfr < 2:
        return np.zeros(0, dtype=int)
    e = np.sum(m[: nfr * flen].reshape(nfr, flen) ** 2, axis=1)
    e_db = 10 * np.log10(e + 1e-12 * max(e.max(), 1e-30))
    k = max(1, int(rise_ms // frame_ms))      # frames that count as "within rise_ms"
    onsets = []
    for i in range(1, nfr):
        before = e_db[max(0, i - k):i].min()
        if e_db[i] - before > threshold_db and e_db[i] > e_db.max() + floor_db:
            pos = i * flen
            if not onsets or pos - onsets[-1] >= min_gap_ms * 1e-3 * fs:
                onsets.append(pos)
    return np.array(onsets, dtype=int)


def pre_echo_db(orig, decoded, fs, delay=CODER_DELAY, window_ms=20, guard_ms=1, onsets=None):
    """Pre-echo: how much coding error appears just BEFORE transient onsets (lower is better).

    Definition (use it in your paper):
      1. Remove the coder delay: o, d = align(orig, decoded, delay); error e = d - o.
      2. Find the onsets in the original with find_onsets(o, fs) (or pass `onsets`).
      3. For every onset at sample n0:
             pre  = the `window_ms` (20 ms) of the ERROR that end `guard_ms` (1 ms) before n0
             post = the 20 ms of the ORIGINAL that start at n0
             value = 10*log10( energy(pre) / energy(post) )  in dB.
         The guard leaves out the onset frame itself, whose position is only known to about 2 ms.
         Onsets whose windows do not fit into the signal are skipped.
      4. The result is the mean of the per-onset values in dB.
    A block transform spreads the quantization noise of a block over the whole block, so a
    long block (large N) puts noise in front of an attack: the value rises with N.
    All channels count together. Returns
      {'pre_echo_db': mean in dB (nan if no onset), 'pre_echo_onsets': number of onsets used,
       'per_onset_db': list of values, 'positions': list of onset positions in samples}.
    In quality_sweep(..., extra_metrics=[pre_echo_db]) the first two become CSV columns.
    """
    o, d = align(orig, decoded, delay)
    err = d - o
    if onsets is None:
        onsets = find_onsets(o, fs)
    w = int(round(window_ms * 1e-3 * fs))
    g = int(round(guard_ms * 1e-3 * fs))
    post_len = int(round(20e-3 * fs))
    vals, used = [], []
    for n0 in np.asarray(onsets, dtype=int):
        a, b = n0 - g - w, n0 - g
        if a < 0 or n0 + post_len > len(o):
            continue
        e_pre = np.sum(err[a:b] ** 2)
        e_post = np.sum(o[n0:n0 + post_len] ** 2)
        if e_post <= 0:
            continue
        vals.append(float(10 * np.log10(max(e_pre, 1e-20) / e_post)))
        used.append(int(n0))
    return {"pre_echo_db": float(np.mean(vals)) if vals else float("nan"), "pre_echo_onsets": len(vals),
            "per_onset_db": vals, "positions": used}


# ---------------------------------------------------------------- running the coder

def run_coder(wav_path, quality=100, workdir=None, coder_dir=None,
              encoder="audio_encoder.py", decoder="audio_decoder.py",
              compressed_ext=".acod", decoded_suffix="rek.wav", extra_args=(), env=None):
    """Encode and decode one WAV file with the course coder (or your modified copy of it).

    The coder is run as `python <encoder> <wav> <quality> [extra_args]` inside `coder_dir`
    (default: $AC_CODER_DIR or ./Python-Audio-Coder), then `python <decoder> <compressed file>`.
    Files are written to `workdir` (a new temporary folder if None).
    env: extra environment variables for the encoder and decoder processes, e.g.
      {'AC_WINDOW_FILE': 'w.txt'} (Topic 5); safer than changing os.environ when runs are parallel.
    Returns {'compressed': path, 'decoded': path, 'encode_s': t, 'decode_s': t}.
    """
    coder_dir = _coder_dir(coder_dir)
    if not os.path.isfile(os.path.join(coder_dir, encoder)):
        raise FileNotFoundError("encoder %s not found in %s (set coder_dir or AC_CODER_DIR)" % (encoder, coder_dir))
    workdir = os.path.abspath(workdir or tempfile.mkdtemp(prefix="ac_"))
    os.makedirs(workdir, exist_ok=True)
    base = os.path.splitext(os.path.basename(wav_path))[0] + "_q%g" % quality
    local_wav = os.path.join(workdir, base + ".wav")
    shutil.copyfile(wav_path, local_wav)
    compressed = os.path.join(workdir, base + compressed_ext)
    decoded = os.path.join(workdir, base + decoded_suffix)
    for old in (compressed, decoded):   # never read a file left over from an earlier run
        if os.path.exists(old):
            os.remove(old)

    def _run(args):
        t0 = time.perf_counter()
        r = subprocess.run([sys.executable] + args, cwd=coder_dir, capture_output=True, text=True,
                           env=dict(os.environ, **env) if env else None)
        if r.returncode != 0:
            raise RuntimeError("%s failed:\n%s" % (args[0], r.stderr[-2000:]))
        return time.perf_counter() - t0

    t_enc = _run([encoder, local_wav, "%g" % quality] + list(extra_args))
    if not os.path.isfile(compressed):
        raise RuntimeError("%s did not write %s (check compressed_ext)" % (encoder, compressed))
    t_dec = _run([decoder, compressed])
    if not os.path.isfile(decoded):
        raise RuntimeError("%s did not write %s (check decoded_suffix)" % (decoder, decoded))
    return {"compressed": compressed, "decoded": decoded, "encode_s": t_enc, "decode_s": t_dec}


FALLBACK_ID = "librosa-v1"


def prepare_test_set(source="auto", sqam_zip=None, testset_dir=None):
    """Create the shared test set in starter_kit/testset (once) and return its id.

    source='sqam'     EBU SQAM: downloads the zip from the EBU (175.5 MB) unless `sqam_zip` points
                      to a copy, checks its SHA-256 and cuts the clips (bit-identical for everyone).
    source='fallback' the CC BY / public-domain clips shipped in starter_kit/testset_fallback.
    source='auto'     SQAM; if that fails, the fallback, with a clear warning.
    An existing test set is kept; delete the testset folder to rebuild it.
    Report the returned id in your paper: results on different test sets are not comparable.
    """
    testset_dir = testset_dir or os.path.join(KIT_DIR, "testset")
    if os.path.isfile(os.path.join(testset_dir, "manifest.csv")):
        return _testset_id(testset_dir)
    if source in ("auto", "sqam"):
        try:
            if KIT_DIR not in sys.path:
                sys.path.insert(0, KIT_DIR)
            import sqam_test_set
            zip_path = sqam_zip or sqam_test_set.download_sqam()
            tid, _ = sqam_test_set.build_sqam_test_set(zip_path, testset_dir)
            print("test set:", tid, "(EBU SQAM)")
            return tid
        except Exception as e:
            shutil.rmtree(testset_dir, ignore_errors=True)
            if source == "sqam":
                raise
            print("=" * 72)
            print("WARNING: EBU SQAM could not be prepared (%s: %s)" % (type(e).__name__, e))
            print("Using the FALLBACK test set (%s). Your numbers are NOT comparable with" % FALLBACK_ID)
            print("results on SQAM. Try again later, or ask in the Q&A forum.")
            print("=" * 72)
    shutil.copytree(os.path.join(KIT_DIR, "testset_fallback"), testset_dir)
    print("test set:", FALLBACK_ID, "(fallback)")
    return FALLBACK_ID


def _testset_id(testset_dir):
    with open(os.path.join(testset_dir, "manifest.csv"), newline="") as f:
        row = next(csv.DictReader(f))
    return row.get("testset") or FALLBACK_ID


def load_manifest(testset_dir=None, split=None, classes=None):
    """Clips of the shared test set as a list of dicts (file, class, split, testset, ...), optionally filtered."""
    testset_dir = testset_dir or os.path.join(KIT_DIR, "testset")
    if not os.path.isfile(os.path.join(testset_dir, "manifest.csv")):
        raise FileNotFoundError("no test set in %s: run prepare_test_set() first" % testset_dir)
    with open(os.path.join(testset_dir, "manifest.csv"), newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["path"] = os.path.join(testset_dir, r["file"])
        r["testset"] = r.get("testset") or FALLBACK_ID
    if split:
        rows = [r for r in rows if r["split"] == split]
    if classes:
        rows = [r for r in rows if r["class"] in classes]
    return rows


FIELDS = ["testset", "clip", "class", "method", "quality", "bits_per_sample", "kbps", "snr_db", "percloss",
          "encode_s", "decode_s", "compressed_bytes", "kit"]


def _metric_columns(metric, orig, decoded, fs, delay):
    """Run one extra metric; returns {column: value} with only the scalar results (for the CSV)."""
    v = metric(orig, decoded, fs, delay)
    if not isinstance(v, dict):
        v = {metric.__name__: v}
    out = {}
    for key, val in v.items():
        if isinstance(val, (bool, int, float, np.integer, np.floating, str)):
            out[key] = round(float(val), 3) if isinstance(val, (float, np.floating)) else val
    return out


def quality_sweep(clips, qualities=(50, 100, 200, 400), out_csv="results.csv", method="baseline",
                  delay=CODER_DELAY, block=1024, workdir=None, append=True, extra_metrics=(), **coder_kwargs):
    """Rate-quality sweep: run the coder for every clip and quality value, write one CSV row each.

    clips: list of WAV paths, or rows from load_manifest().
    qualities: values of the coder's quality parameter (at least 4 operating points).
    method: a label for your configuration (e.g. 'sine window', 'KBD alpha=4').
    delay, block: coder delay in samples and block length (both N for the course coder);
      change them when you change N or the filter bank.
    extra_metrics: functions f(orig, decoded, fs, delay) that add columns at the END of each row,
      e.g. extra_metrics=[pre_echo_db]. A function returns a number (column named after the
      function) or a dict (one column per number in it; lists are left out). The standard
      columns and their values do not change.
    Other keyword arguments go to run_coder() (coder_dir, encoder, decoder, extra_args, ...).
    Rows are appended to out_csv, so several methods can share one file; running the same
    sweep again adds its rows again (plot_rate_quality uses the latest). append=False starts a new file.
    Returns the list of result rows.
    """
    rows = []
    workdir = workdir or tempfile.mkdtemp(prefix="ac_sweep_")
    new_file = not (append and os.path.exists(out_csv))
    extra_cols = None                     # columns after FIELDS: from the file, or from the first row
    if not new_file:
        with open(out_csv, newline="") as f:
            header = next(csv.reader(f), [])
        if header[:len(FIELDS)] != FIELDS:
            raise ValueError("%s was written by another kit version (different columns): "
                             "use a new file name or append=False" % out_csv)
        extra_cols = header[len(FIELDS):]
    with open(out_csv, "a" if append else "w", newline="") as f:
        w = None
        if not new_file:
            w = csv.DictWriter(f, fieldnames=FIELDS + extra_cols, restval="")
        for clip in clips:
            path = clip["path"] if isinstance(clip, dict) else clip
            cls = clip.get("class", "") if isinstance(clip, dict) else ""
            tset = clip.get("testset", "") if isinstance(clip, dict) else "own clip"
            fs, x = read_wav(path)
            for q in qualities:
                out = run_coder(path, q, workdir=workdir, **coder_kwargs)
                _, y = read_wav(out["decoded"])
                qual = evaluate(x, y, fs, delay, block)
                row = {"testset": tset, "clip": os.path.basename(path), "class": cls, "method": method, "quality": q,
                       "bits_per_sample": round(bits_per_sample(out["compressed"], num_samples(x)), 5),
                       "kbps": round(bitrate_kbps(out["compressed"], len(x) / fs), 3),
                       "snr_db": round(qual["snr_db"], 3), "percloss": "%.6g" % qual["percloss"],
                       "encode_s": round(out["encode_s"], 2), "decode_s": round(out["decode_s"], 2),
                       "compressed_bytes": os.path.getsize(out["compressed"]), "kit": KIT_VERSION}
                for metric in extra_metrics:
                    row.update(_metric_columns(metric, x, y, fs, delay))
                if w is None:             # new file: the header follows the first row
                    extra_cols = [c for c in row if c not in FIELDS]
                    w = csv.DictWriter(f, fieldnames=FIELDS + extra_cols, restval="")
                    w.writeheader()
                unknown = [c for c in row if c not in FIELDS + extra_cols]
                if unknown:
                    raise ValueError("%s has no columns %s: use a new file name or append=False" % (out_csv, unknown))
                w.writerow(row)
                f.flush()
                rows.append(row)
                print("%-28s q=%-6g %6.3f bit/sample  SNR %6.2f dB  percloss %s" %
                      (row["clip"], q, row["bits_per_sample"], row["snr_db"], row["percloss"]))
        if w is None:                     # new file and no rows: still write the header
            csv.DictWriter(f, fieldnames=FIELDS).writeheader()
    return rows


def plot_rate_quality(csv_file, metric="snr_db", by="method", ax=None):
    """Rate-quality curves from a sweep CSV: x = bits per sample, y = metric, one line per `by`
    value and clip. Returns the matplotlib axes; save the figure next to the CSV."""
    import matplotlib.pyplot as plt
    with open(csv_file, newline="") as f:
        rows = list(csv.DictReader(f))
    ax = ax or plt.figure(figsize=(6, 4)).gca()
    # a sweep that was run twice appends its rows twice: keep the latest row per point
    latest = {(r.get("testset", ""), r[by], r["clip"], r["quality"]): r for r in rows}
    rows = list(latest.values())
    if len({r.get("testset", "") for r in rows}) > 1:
        print("warning: %s mixes results from different test sets" % csv_file)
    keys = sorted({(r[by], r["clip"]) for r in rows})
    for key in keys:
        pts = sorted((float(r["bits_per_sample"]), float(r[metric])) for r in rows if (r[by], r["clip"]) == key)
        ax.plot(*zip(*pts), marker="o", label="%s, %s" % key)
    ax.set_xlabel("bits per sample (from file size)")
    ax.set_ylabel({"snr_db": "SNR (dB)", "percloss": "psychoacoustic loss (lower is better)"}.get(metric, metric))
    if metric == "percloss":
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)
    return ax
