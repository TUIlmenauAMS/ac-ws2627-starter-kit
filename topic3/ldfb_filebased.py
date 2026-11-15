"""Topic 3: the low-delay filter bank (LDFB) with the same interface as the coder's MDCT.

The course coder calls MDCTanafb(x, N, fb) and MDCTsynfb(y, fb) (MDCTfb.py) on whole signals.
The working LDFB, LDFBclass in FilterBanks_FastPythonImplementation_Classes, processes one block
at a time. This file connects the two:

    coeff = ldfb_coefficients(N)        LDFB coefficients for N subbands (N = 128 ... 2048)
    y = LDFBanafb(x, N, coeff)          analysis: x (1-D) -> y, shape (N, blocks), like MDCTanafb
    xr = LDFBsynfb(y, coeff)            synthesis: y -> xr (1-D), like MDCTsynfb
    filterbank_report(N)                delay, reconstruction SNR and stopband attenuation, LDFB vs MDCT
    install(coder_dir)                  copy the LDFB coder files into your copy of the coder

Conventions (identical to MDCTanafb/MDCTsynfb, so the coder needs no other change):
  * Only complete blocks of N samples of x are used; y has (len(x)//N + 1) blocks.
  * xr has (blocks + 1)*N samples and is delayed by N samples: xr[N + n] reconstructs x[n].

Where the coefficients come from: the file fb2048t1023d512bbitcs.mat in a clone of
https://github.com/TUIlmenauAMS/FilterBanks_FastPythonImplementation_Classes ("2048 taps,
delay 1023, 512 bands"). Pass its folder as `fb_dir`, or set the environment variable AC_FB_DIR.
For other N the coefficients are interpolated with fbinterp2.fbinterp() from the same repository.

The block classes come from pyrecplayfastLDFB_classes.py in the same repository, at the commit
the kit was tested with (setup_topic.py). That commit includes the fix of Nov 2026: older versions
reconstruct only to about 47 dB SNR and fail on import.
"""

import contextlib
import importlib.util
import io
import os
import shutil

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FB_REPO = "FilterBanks_FastPythonImplementation_Classes"
FB_FILE = "fb2048t1023d512bbitcs.mat"
SUPPORTED_N = (128, 256, 512, 1024, 2048)
_FIX_MARK = "was a view"                 # comment in the fixed block classes (Nov 2026)


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_CLASSES = None


def block_classes():
    """The pyrecplayfastLDFB_classes module (LDFBclass, MDCTclass), in its fixed version.

    Looked for next to this file (after install()), then in the filter bank repository (find_fb_dir()).
    A version without the fix of Nov 2026 is refused: its LDFBclass does not reconstruct perfectly.
    """
    global _CLASSES
    if _CLASSES is None:
        dirs = [HERE]
        try:
            dirs.append(find_fb_dir())
        except FileNotFoundError:
            pass
        for d in dirs:
            p = os.path.join(d, "pyrecplayfastLDFB_classes.py")
            if os.path.isfile(p):
                with open(p) as f:
                    if _FIX_MARK in f.read():
                        _CLASSES = _load_module("pyrecplayfastLDFB_classes_fixed", p)
                        break
        else:
            raise ImportError("no fixed pyrecplayfastLDFB_classes.py found in %s. Clone the filter bank repository "
                              "at the kit's commit with\n  python starter_kit/setup_topic.py --topic 3\n"
                              "(an older clone has the version that reconstructs only to about 47 dB)." % dirs)
    return _CLASSES


def find_fb_dir(fb_dir=None):
    """Folder with fb2048t1023d512bbitcs.mat and fbinterp2.py: the argument, else $AC_FB_DIR, else
    ./FilterBanks_FastPythonImplementation_Classes or ../FilterBanks_FastPythonImplementation_Classes."""
    candidates = [fb_dir] if fb_dir else [os.environ.get("AC_FB_DIR"), FB_REPO, os.path.join("..", FB_REPO)]
    for d in candidates:
        if d and os.path.isfile(os.path.join(d, FB_FILE)):
            return os.path.abspath(d)
    raise FileNotFoundError("%s not found (looked in %s). Clone %s with\n  python starter_kit/setup_topic.py --topic 3\n"
                            "and pass fb_dir=... or set the environment variable AC_FB_DIR to that folder."
                            % (FB_FILE, [c for c in candidates if c], FB_REPO))


def ldfb_coefficients(N, fb_dir=None, coeff_file=None):
    """LDFB coefficients for N subbands: an array of 2.5*N values.

    coeff_file (or the environment variable AC_LDFB_COEFF_FILE): a text file (np.savetxt) with
    your own 2.5*N coefficients, e.g. optimized ones (Topic 5, variant b). Then that file is used
    instead of the course's coefficients, also by audio_encoder_ldfb.py/audio_decoder_ldfb.py.

    The first 1.5*N values define the F matrix (as the MDCT window does), the next N/2 the G0
    matrix and the last N/2 the G1 matrix. They are built as the repository's __main__ does:
    take the N = 512 set from fb2048t1023d512bbitcs.mat (2048 values, of which the symmetric
    F matrix needs only the first 1.5*512), and for N != 512 interpolate it with fbinterp2.fbinterp().
    Any coefficients give perfect reconstruction (the structure guarantees it); interpolation
    only changes the frequency responses. Supported N: 128, 256, 512, 1024, 2048.
    """
    coeff_file = coeff_file or os.environ.get("AC_LDFB_COEFF_FILE")
    if coeff_file:
        c = np.loadtxt(coeff_file)
        if len(c) != int(2.5 * N):
            raise ValueError("%s has %d values; N=%d needs 2.5*N = %d" % (coeff_file, len(c), N, int(2.5 * N)))
        return c
    if N not in SUPPORTED_N:
        raise ValueError("N=%s is not supported; use one of %s" % (N, SUPPORTED_N))
    d = find_fb_dir(fb_dir)
    fb = np.loadtxt(os.path.join(d, FB_FILE))
    fbsym = np.append(fb[:int(1.5 * 512)], fb[2 * 512:])
    if N != 512:
        fbinterp2 = _load_module("fbinterp2", os.path.join(d, "fbinterp2.py"))
        with contextlib.redirect_stdout(io.StringIO()):          # fbinterp prints its grid
            fbsym = fbinterp2.fbinterp(fb=fbsym, n=N / 512, N=512)
    fbsym = np.asarray(fbsym, dtype=np.float64)
    assert len(fbsym) == int(2.5 * N), (len(fbsym), N)
    return fbsym


def sine_window(N):
    """The coder's MDCT window: the first 1.5*N values of the sine window."""
    return np.sin(np.pi / (2 * N) * (np.arange(int(1.5 * N)) + 0.5))


def _analysis(fbank, x, N):
    x = np.asarray(x, dtype=np.float64)
    nb = len(x) // N
    blocks = np.zeros((nb + 1) * N)               # one extra block flushes the delay, as MDCTanafb
    blocks[:nb * N] = x[:nb * N]
    y = np.zeros((N, nb + 1))
    for m in range(nb + 1):
        y[:, m] = fbank.forward(blocks[m * N:(m + 1) * N].copy())
    return y


def _synthesis(fbank, y):
    N, nb = y.shape
    xr = np.zeros((nb + 1) * N)                   # one extra block, as MDCTsynfb
    for m in range(nb + 1):
        col = y[:, m].copy() if m < nb else np.zeros(N)
        xr[m * N:(m + 1) * N] = fbank.inverse(col)
    return xr


def LDFBanafb(x, N, coeff):
    """LDFB analysis of a whole signal. x: 1-D array; coeff: from ldfb_coefficients(N).
    Returns y of shape (N, len(x)//N + 1): one column of N subband values per block, as MDCTanafb.
    A new LDFBclass is made for every call, so no filter memory leaks between files or channels."""
    if len(coeff) != int(2.5 * N):
        raise ValueError("coeff has %d values, LDFB with N=%d needs 2.5*N = %d" % (len(coeff), N, int(2.5 * N)))
    return _analysis(block_classes().LDFBclass(N, np.asarray(coeff, dtype=np.float64)), x, N)


def LDFBsynfb(y, coeff):
    """LDFB synthesis: y of shape (N, blocks) -> signal with (blocks+1)*N samples, delayed by N
    samples against the input of LDFBanafb (as MDCTsynfb)."""
    N = y.shape[0]
    if len(coeff) != int(2.5 * N):
        raise ValueError("coeff has %d values, LDFB with N=%d needs 2.5*N = %d" % (len(coeff), N, int(2.5 * N)))
    return _synthesis(block_classes().LDFBclass(N, np.asarray(coeff, dtype=np.float64)), y)


def MDCTanafb_blocks(x, N, fb=None):
    """The coder's MDCT, computed with MDCTclass block by block (numerically identical to
    MDCTanafb in MDCTfb.py). fb: first 1.5*N window values (default: sine window)."""
    return _analysis(block_classes().MDCTclass(N, sine_window(N) if fb is None else fb), x, N)


def MDCTsynfb_blocks(y, fb=None):
    N = y.shape[0]
    return _synthesis(block_classes().MDCTclass(N, sine_window(N) if fb is None else fb), y)


# ------------------------------------------------------------------ measurements

def analysis_impulse_response(ana, N, band, blocks=8):
    """Impulse response h(t) of the analysis filter of subband `band`, measured with impulses.

    An impulse at input position p (0 <= p < N) gives the subband values y[band, m]; the output
    of block m is available when its last input sample (m*N + N - 1) has arrived, so the value
    belongs to t = m*N + N - 1 - p samples after the impulse. N impulse positions fill in all t.
    `ana` is a function (x, N) -> y, e.g. lambda x, N: LDFBanafb(x, N, coeff).
    """
    h = np.zeros(blocks * N)
    for p in range(N):
        x = np.zeros(blocks * N)
        x[p] = 1.0
        y = ana(x, N)
        for m in range(blocks):
            t = m * N + N - 1 - p
            if 0 <= t < len(h):
                h[t] = y[band, m]
    nz = np.flatnonzero(np.abs(h) > 1e-12)
    return h[: nz[-1] + 1] if len(nz) else h[:1]


def stopband_attenuation_db(h, N, band, edge_bands=2, nfft=None):
    """Stopband attenuation of a subband filter in dB (higher is better).

    = peak gain in the passband (within half a band width of the band centre (band+0.5)*pi/N)
      minus the highest gain in the stopband: all frequencies at least `edge_bands` band widths
      (edge_bands*pi/N) away from the band centre. Computed from |FFT(h)| with at least
      16*len(h) points. With edge_bands=2 it measures the first side lobes next to the band,
      with edge_bands=5 the attenuation further away; report which one you use.
    """
    nfft = nfft or 1 << int(np.ceil(np.log2(16 * max(len(h), 2 * N))))
    H = np.abs(np.fft.rfft(h, nfft))
    w = np.linspace(0, np.pi, len(H))
    wc = (band + 0.5) * np.pi / N
    H_db = 20 * np.log10(H + 1e-300)
    passband = H_db[np.abs(w - wc) <= 0.5 * np.pi / N].max()
    stopband = H_db[np.abs(w - wc) >= edge_bands * np.pi / N].max()
    return float(passband - stopband)


def block_delay(ana, syn, N, positions=None):
    """Delay of the file-based filter bank in samples, measured with impulses: an impulse at p
    comes out at p + delay. Returns the delay (it must be the same for every p)."""
    positions = positions or (0, N // 3, N - 1, 2 * N + 5)
    found = set()
    for p in positions:
        x = np.zeros(12 * N)
        x[p] = 1.0
        xr = syn(ana(x, N))
        found.add(int(np.argmax(np.abs(xr))) - p)
    if len(found) != 1:
        raise RuntimeError("delay depends on the impulse position: %s" % sorted(found))
    return found.pop()


def reconstruction_snr_db(ana, syn, N, delay, blocks=40, seed=0):
    """SNR in dB of synthesis(analysis(x)) against x (white noise), after removing `delay`."""
    x = np.random.default_rng(seed).standard_normal(blocks * N)
    xr = syn(ana(x, N))
    n = len(x) - N                                # leave out the last block (needs the next one)
    e = x[:n] - xr[delay:delay + n]
    return float("inf") if not np.any(e) else float(10 * np.log10(np.sum(x[:n] ** 2) / np.sum(e ** 2)))


def filterbank_report(N, fs=44100, band=4, fb_dir=None, show=True):
    """LDFB and MDCT with the same N: delays, reconstruction SNR, length and stopband attenuation
    (2 and 5 band widths away from the band centre) of the analysis filter of subband `band`. Returns a dict {'LDFB': {...}, 'MDCT': {...}} and
    prints a small table.

    Two delays are reported:
      * block delay: what you see in the file-based coder. An impulse at sample p of the input
        file comes out at sample p + block delay of the output file (N for both filter banks).
        This is the `delay` that k.evaluate()/k.quality_sweep() need to align the signals.
      * system (algorithmic) delay = block delay + N - 1: in a real-time system, a block can only
        be processed when its last sample has arrived, so the first sample of a block waits N - 1
        more samples. This is the delay to report in the paper (in ms: delay / fs * 1000).
    """
    coeff = ldfb_coefficients(N, fb_dir)
    banks = {
        "LDFB": (lambda x, n: LDFBanafb(x, n, coeff), lambda y: LDFBsynfb(y, coeff)),
        "MDCT": (lambda x, n: MDCTanafb_blocks(x, n), lambda y: MDCTsynfb_blocks(y)),
    }
    out = {}
    for name, (ana, syn) in banks.items():
        d = block_delay(ana, syn, N)
        h = analysis_impulse_response(ana, N, band)
        out[name] = {"N": N, "block_delay": d, "system_delay": d + N - 1,
                     "system_delay_ms": round((d + N - 1) / fs * 1000, 2),
                     "reconstruction_snr_db": round(reconstruction_snr_db(ana, syn, N, d), 1),
                     "filter_length": len(h),
                     "stopband_att_2bands_db": round(stopband_attenuation_db(h, N, band, 2), 1),
                     "stopband_att_5bands_db": round(stopband_attenuation_db(h, N, band, 5), 1)}
    if show:
        print("N = %d, fs = %d Hz, analysis filter of band %d" % (N, fs, band))
        print("%-5s %12s %13s %8s %14s %12s %18s" % ("", "block delay", "system delay", "(ms)", "recon. SNR dB",
                                                   "filter taps", "stopband att. dB"))
        print("%-5s %12s %13s %8s %14s %12s %18s" % ("", "", "", "", "", "", "(2 / 5 bands away)"))
        for name, r in out.items():
            print("%-5s %12d %13d %8.2f %14.1f %12d %10.1f / %5.1f" % (
                name, r["block_delay"], r["system_delay"], r["system_delay_ms"], r["reconstruction_snr_db"],
                r["filter_length"], r["stopband_att_2bands_db"], r["stopband_att_5bands_db"]))
    return out


# ------------------------------------------------------------------ coder files

CODER_FILES = ("audio_encoder_ldfb.py", "audio_decoder_ldfb.py", "ldfb_filebased.py")


def install(coder_dir):
    """Copy the LDFB coder files into your copy of Python-Audio-Coder (coder_dir):
    audio_encoder_ldfb.py, audio_decoder_ldfb.py, ldfb_filebased.py and the (fixed)
    pyrecplayfastLDFB_classes.py. Your original audio_encoder.py/audio_decoder.py stay as they are."""
    for f in CODER_FILES:
        shutil.copy(os.path.join(HERE, f), os.path.join(coder_dir, f))
    src = block_classes().__file__
    shutil.copy(src, os.path.join(coder_dir, "pyrecplayfastLDFB_classes.py"))
    return coder_dir


if __name__ == "__main__":
    for n in (512, 1024):
        filterbank_report(n)
        print()
