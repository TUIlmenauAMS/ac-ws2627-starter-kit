"""Topic 5: a window described by a few parameters, objectives for it, and an evaluation counter.

    w = window_from_params(p, N)        first 1.5*N window values for the coder (p = 0: sine window)
    stopband_energy(p, N)               objective (a): window leakage in dB, fast, smooth
    coder_objective(p, N, clip, ...)    objective (extension): bitrate of the real coder + quality penalty
    ldfb_objective(p, N)                objective (b): leakage of the low-delay filter bank (topic3/)
    f = counted(objective, 'log.txt', budget=2000)
                                        counts every evaluation, also in parallel worker processes
    install(coder_dir)                  copies audio_encoder_window.py/audio_decoder_window.py there

Why parameters instead of the 1536 window values: the coder's window (N = 1024) has 1.5*N = 1536
values. The objective functions in Python-Audio-Coder (optimfuncMDCT) are built for small N and do
not improve from the sine window at N = 64 or 256. A smooth family with K = 4..12 parameters keeps
every candidate a valid, smooth window, and both optimizers can work with it.
"""

import functools
import os
import shutil
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
KIT = os.path.dirname(HERE)
_MARGIN = 0.05            # the warp's slope stays >= this value (keeps it strictly increasing)


# ------------------------------------------------------------------ the window family

def warp(u, p):
    """The warp g(u) = u + sum_k p_k sin(2 pi k u) / (2 pi k), k = 1..K, for u in [0, 1].

    g(0) = 0, g(1) = 1, and g(1 - u) = 1 - g(u) for any p (each sine term is odd around u = 1/2).
    Its slope is g'(u) = 1 + sum_k p_k cos(2 pi k u) >= 1 - sum_k |p_k|. To keep g strictly
    increasing, p is scaled down when sum_k |p_k| > 1 - 0.05 (then the slope stays >= 0.05).
    Returns (g(u), p actually used).
    """
    p = np.asarray(p, dtype=np.float64).ravel()
    s = np.sum(np.abs(p))
    if s > 1 - _MARGIN:
        p = p * (1 - _MARGIN) / s
    k = np.arange(1, len(p) + 1)[:, None]
    g = u + np.sum(p[:, None] * np.sin(2 * np.pi * k * u) / (2 * np.pi * k), axis=0)
    return g, p


def window_from_params(p, N, full=False):
    """Window for the course coder from K parameters p (K = 4..12 is sensible).

    The window has 2*N values. Its rising half (n = 0..N-1) is
        w(n) = sin( pi/2 * g(u) ),   u = (n + 0.5) / N,
    with the warp g from warp(); the falling half is the mirror image, w(2N-1-n) = w(n).
    For p = 0, g(u) = u and w is exactly the sine window sin(pi/(2N) (n + 0.5)) of the coder.
    Returns the first 1.5*N values (what audio_encoder.py and audio_decoder.py use as `fb`),
    or all 2*N values with full=True.

    Why every such window gives perfect reconstruction:
      1. The coder only takes the first 1.5*N values. symFmatrix() builds the folding matrix F
         from them and computes the remaining coefficients so that F can be inverted exactly
         (det F = +-1). Any values work, as long as the first N/2 are not zero; here all values
         are strictly positive because 0 < g(u) < 1.
      2. Moreover, these windows are symmetric and satisfy the Princen-Bradley condition
         w(n)^2 + w(n+N)^2 = 1, because g(1-u) = 1 - g(u) turns sin into cos:
         sin^2(pi/2 g(u)) + sin^2(pi/2 (1 - g(u))) = sin^2 + cos^2 = 1.
         So analysis and synthesis use the same window (an orthogonal MDCT), like the sine
         and the KBD windows.
    """
    u = (np.arange(N) + 0.5) / N
    g, _ = warp(u, p)
    rise = np.sin(np.pi / 2 * g)
    w = np.concatenate([rise, rise[::-1]])
    return w if full else w[: int(1.5 * N)]


def sine_window(N):
    """The coder's sine window (first 1.5*N values)."""
    return np.sin(np.pi / (2 * N) * (np.arange(int(1.5 * N)) + 0.5))


# ------------------------------------------------------------------ objective (a): leakage

def stopband_energy(p, N, mainlobe_bands=3, oversample=16):
    """Objective (a): how much of the window's spectrum lies outside its main lobe, in dB.

    The full 2N-point window is transformed with an FFT of oversample*2N points (>= 16*N).
    Frequencies are measured in subband widths: one subband of the MDCT with N bands is
    1/(2N) cycles per sample wide, which is one bin of a 2N-point DFT. The main lobe is
    |f| < mainlobe_bands subband widths (default 3). The result is
        10*log10( energy of the spectrum outside the main lobe / total energy )
    in dB; lower is better (less leakage into far-away subbands). The sine window gives
    -31.5 dB, a KBD window with beta = 8 gives -43.7 dB (N = 1024).
    Fast (about 1 ms per call) and smooth, so gradient methods can use it (numerical gradient).
    """
    w = window_from_params(p, N, full=True)
    W = np.abs(np.fft.rfft(w, oversample * len(w))) ** 2
    bins = np.arange(len(W)) / oversample          # frequency in subband widths
    total = W[0] + 2 * np.sum(W[1:])
    outside = 2 * np.sum(W[bins >= mainlobe_bands])
    return float(10 * np.log10(outside / total))


# ------------------------------------------------------------------ objective (extension): the coder

def _ac_kit():
    if KIT not in sys.path:
        sys.path.insert(0, KIT)
    import ac_kit
    return ac_kit


def coder_run(p, N, clip, quality, coder_dir, workdir=None):
    """Encode and decode `clip` with the window of parameters p (p = None: sine window).

    Uses audio_encoder_window.py/audio_decoder_window.py in coder_dir (see install()) and the
    environment variable AC_WINDOW_FILE, passed with run_coder(..., env=...), so parallel runs
    do not disturb each other. Returns {'bits_per_sample', 'snr_db', 'percloss', 'seconds'}.
    """
    k = _ac_kit()
    t0 = time.perf_counter()
    tmp = tempfile.mkdtemp(prefix="t5_", dir=workdir)
    try:
        env = None
        if p is not None:
            wfile = os.path.join(tmp, "window.txt")
            np.savetxt(wfile, window_from_params(p, N))
            env = {"AC_WINDOW_FILE": wfile}
        out = k.run_coder(clip, quality, workdir=tmp, coder_dir=coder_dir, encoder="audio_encoder_window.py",
                          decoder="audio_decoder_window.py", env=env)
        fs, x = k.read_wav(clip)
        _, y = k.read_wav(out["decoded"])
        q = k.evaluate(x, y, fs, delay=N, block=N)
        return {"bits_per_sample": k.bits_per_sample(out["compressed"], k.num_samples(x)),
                "snr_db": q["snr_db"], "percloss": q["percloss"], "seconds": time.perf_counter() - t0}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def coder_objective(p, N, clip, quality, coder_dir, baseline=None, weight=1.0):
    """Objective for the extension: the real coder with this window (lower is better).

        value = bits_per_sample + weight * max(0, percloss / baseline_percloss - 1)

    i.e. the bitrate from the file size, plus a penalty when the psychoacoustic loss rises
    above the sine window's at the same quality setting (weight 1.0: +1 bit/sample for twice
    the baseline loss). baseline: coder_run(None, N, clip, quality, coder_dir) (computed once
    and cached in this process if None: pass it to avoid the extra run in every worker).
    No gradient (bitrate and quantization are step functions of p) and slow: each call encodes
    and decodes the clip, about 3 s for a 6-s mono clip at N = 1024 (12 Nov 2026, laptop CPU).
    Use short clips and few parameters, and a black-box optimizer.
    """
    if baseline is None:
        key = (N, clip, quality, coder_dir)
        if key not in _BASELINES:
            _BASELINES[key] = coder_run(None, N, clip, quality, coder_dir)
        baseline = _BASELINES[key]
    r = coder_run(p, N, clip, quality, coder_dir)
    return float(r["bits_per_sample"] + weight * max(0.0, r["percloss"] / baseline["percloss"] - 1.0))


_BASELINES = {}


# ------------------------------------------------------------------ objective (b): low-delay filter bank

def _topic3():
    sys.path.insert(0, os.path.join(KIT, "topic3"))
    try:
        import ldfb_filebased
    finally:
        sys.path.pop(0)
    return ldfb_filebased


_LDFB_BASE = {}


def ldfb_coeff_from_params(p, N, fb_dir=None):
    """Low-delay filter bank coefficients (2.5*N values) from K parameters p.

    Starts from the course's LDFB coefficients (topic3.ldfb_coefficients(N)) and multiplies
    them by a smooth positive factor  1 + sum_k p_k cos(pi k t) / (1 + sum_k |p_k|),
    t = 0..1 along each of the five N/2-sections (F: three sections, G0, G1).
    p = 0 gives the original coefficients. The factor stays between 0 and 2, so no coefficient
    changes its sign and the first N/2 stay non-zero: the structure of the LDFB then
    guarantees perfect reconstruction for every p.
    """
    if N not in _LDFB_BASE:
        _LDFB_BASE[N] = _topic3().ldfb_coefficients(N, fb_dir)
    base = _LDFB_BASE[N]
    p = np.asarray(p, dtype=np.float64).ravel()
    t = np.linspace(0, 1, N // 2)
    k = np.arange(1, len(p) + 1)[:, None]
    shape = np.sum(p[:, None] * np.cos(np.pi * k * t), axis=0) / (1 + np.sum(np.abs(p)))
    return base * np.tile(1 + shape, 5)


def ldfb_objective(p, N, mainlobe_bands=3, fb_dir=None):
    """Objective (b): leakage of the low-delay filter bank, in dB (lower is better).

    The synthesis impulse response of subband 0 (one unit value in band 0 of the first block,
    through LDFBsynfb) is the filter bank's prototype filter. The result is
        10*log10( energy of its spectrum more than `mainlobe_bands` subband widths
                  away from the band centre / total energy ),
    the same measure as stopband_energy() but for the LDFB filter (4*N taps instead of 2*N).
    Coefficients: ldfb_coeff_from_params(p, N); needs FilterBanks_FastPythonImplementation_Classes
    (fb_dir or the environment variable AC_FB_DIR). About 5 ms per call at N = 1024.
    """
    L = _topic3()
    c = ldfb_coeff_from_params(p, N, fb_dir)
    y = np.zeros((N, 6))
    y[0, 0] = 1.0
    g = L.LDFBsynfb(y, c)
    G = np.abs(np.fft.rfft(g, 1 << int(np.ceil(np.log2(16 * len(g)))))) ** 2
    w_bands = np.arange(len(G)) / (len(G) - 1) * N                   # frequency in subband widths
    total = G[0] + 2 * np.sum(G[1:])
    outside = 2 * np.sum(G[np.abs(w_bands - 0.5) >= mainlobe_bands])
    return float(10 * np.log10(outside / total))


# ------------------------------------------------------------------ counting evaluations

class BudgetExhausted(Exception):
    """Raised by a counted objective when its evaluation budget is used up."""


class counted:
    """Wrap an objective f(p) so that every evaluation is counted and logged, also when the
    optimizer evaluates in parallel worker processes.

        f = counted(functools.partial(stopband_energy, N=1024), 'rd_run1.log', budget=2000)
        try:
            optimrandomdir(f, p0, iterations=2000)
        except BudgetExhausted:
            pass
        f.write_csv('rd_run1.csv')       # evaluation, value, best_so_far, seconds
        value, p = f.best()

    How the counting works: optimrandomdir() evaluates its random trials in parallel worker
    processes (joblib). A Python counter inside the objective would count in each worker
    separately and the main process would never see those counts. Here every evaluation
    appends ONE line of fixed length to the log file (opened in append mode, so lines from
    different processes do not overwrite each other). The number of evaluations is the file
    size divided by the line length, which every process can read. This counts the
    optimizer's parallel trials, its line search (in the main process) and, for L-BFGS-B,
    the extra evaluations of the numerical gradient.

    budget: before each evaluation the count is checked; when `budget` evaluations are done,
    BudgetExhausted is raised (catch it, the best point is in the log). With parallel workers
    a few evaluations may already be running at that moment and finish after the budget;
    history() and write_csv() keep only the first `budget` lines (in the order they finished).
    The objective must be picklable for parallel workers: a function defined at module level
    (not a lambda); bind extra arguments with functools.partial.
    """

    def __init__(self, f, log_file, budget=None, append=False):
        self.f = f
        self.log_file = os.path.abspath(log_file)
        self.budget = budget
        self.t0 = time.time()
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        if not append or not os.path.exists(self.log_file):
            open(self.log_file, "w").close()

    @staticmethod
    def _line(seconds, value, p):
        fields = ["%14.6f" % seconds, "%25.16e" % value, "%10d" % os.getpid()] + ["%25.16e" % v for v in p]
        return (",".join(fields) + "\n").encode()

    def _line_length(self):
        with open(self.log_file, "rb") as fh:
            first = fh.readline()
        return len(first)

    def count(self):
        """Number of evaluations so far (over all processes)."""
        n = self._line_length()
        return os.path.getsize(self.log_file) // n if n else 0

    def __call__(self, p, *args, **kwargs):
        if self.budget is not None and self.count() >= self.budget:
            raise BudgetExhausted("budget of %d evaluations used up" % self.budget)
        p = np.asarray(p, dtype=np.float64).ravel()
        value = float(self.f(p, *args, **kwargs))
        line = self._line(time.time() - self.t0, value, p)
        fd = os.open(self.log_file, os.O_WRONLY | os.O_APPEND | os.O_CREAT)
        try:
            os.write(fd, line)
        finally:
            os.close(fd)
        return value

    def _records(self):
        with open(self.log_file) as fh:
            for line in fh:
                parts = line.split(",")
                yield float(parts[0]), float(parts[1]), np.array([float(v) for v in parts[3:]])

    def history(self, budget=None):
        """List of dicts {'evaluation', 'value', 'best_so_far', 'seconds'}, at most `budget` rows
        (default: the counter's budget)."""
        budget = budget or self.budget
        rows, best = [], np.inf
        for i, (sec, val, _) in enumerate(self._records()):
            if budget and i >= budget:
                break
            best = min(best, val)
            rows.append({"evaluation": i + 1, "value": val, "best_so_far": best, "seconds": round(sec, 4)})
        return rows

    def best(self, budget=None):
        """(best value, its parameters) within the budget."""
        budget = budget or self.budget
        best = (np.inf, None)
        for i, (_, val, p) in enumerate(self._records()):
            if budget and i >= budget:
                break
            if val < best[0]:
                best = (val, p)
        return best

    def write_csv(self, path, budget=None, **extra_columns):
        """Write history() as a CSV file; extra_columns (e.g. optimizer='RD', run=1) go in front."""
        import csv
        rows = self.history(budget)
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(list(extra_columns) + ["evaluation", "value", "best_so_far", "seconds"])
            for r in rows:
                w.writerow(list(extra_columns.values()) + [r["evaluation"], "%.10g" % r["value"],
                                                           "%.10g" % r["best_so_far"], r["seconds"]])
        return path


# ------------------------------------------------------------------ coder files

def install(coder_dir):
    """Copy audio_encoder_window.py and audio_decoder_window.py into your copy of the coder
    (which must contain the kit's fixed PythonPsychoacoustics/psyac_quantization.py)."""
    for f in ("audio_encoder_window.py", "audio_decoder_window.py"):
        shutil.copy(os.path.join(HERE, f), os.path.join(coder_dir, f))
    return coder_dir


def objective(name, N=1024, **kwargs):
    """Convenience: the objective `name` ('stopband', 'ldfb') with N (and other arguments)
    bound, as a picklable function of p only, for counted() and the optimizers."""
    f = {"stopband": stopband_energy, "ldfb": ldfb_objective, "coder": coder_objective}[name]
    return functools.partial(f, N=N, **kwargs)
