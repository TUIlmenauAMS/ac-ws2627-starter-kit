"""Checks for the kit's per-topic helpers: pre-echo measure (ac_kit), baseline.py, topic3/, topic5/.

Registered with run_checks.py; run them with --only preecho,baseline,topic3,topic5.
"""

import csv
import io
import os
import shutil
from contextlib import redirect_stdout

from run_checks import Skip, check, fixed_coder_copy, last_lines, read_csv  # noqa: F401


def coder_with_n(ctx, n):
    """Copy of the fixed coder with N=n in encoder and decoder (in the work folder, made once)."""
    d = os.path.join(ctx.workdir, "coder_N%d" % n)
    if n == 1024:
        return ctx.coder
    if not os.path.isfile(os.path.join(d, "audio_encoder.py")):
        fixed_coder_copy(ctx, d, edits=[(f, "N=1024", "N=%d" % n) for f in ("audio_encoder.py", "audio_decoder.py")])
    return d


def transient_clip(ctx):
    """The castanets clip of SQAM, or the dense snare clip of the fallback set."""
    clips = ctx.clips(classes=["transient"])
    for key in ("castanet", "snare_dense"):
        for c in clips:
            if key in c["file"]:
                return c
    return clips[0]


# ------------------------------------------------------------------ WP4: pre-echo

@check("preecho.onsets", "find_onsets: synthetic clicks within +-2 ms, close onsets merged; onsets of the transient clips")
def pe_onsets(ctx):
    import numpy as np
    k = ctx.kit_module()
    fs = 44100
    rng = np.random.default_rng(0)
    true = [5000, 30000, 60000, 100000, 150000]
    x = rng.standard_normal(200000) * 1e-3
    for p in true + [31000]:                     # 31000: 23 ms after 30000, must be merged into it
        x[p:p + 4000] += rng.standard_normal(4000) * np.exp(-np.arange(4000) / 600)
    on = k.find_onsets(x, fs)
    assert len(on) == len(true), "found %s, expected %s" % (on, true)
    err_ms = np.abs(on - np.array(true)) / fs * 1000
    assert err_ms.max() <= 2.0, "onset error up to %.2f ms" % err_ms.max()
    st = np.stack([x, 0.5 * x], axis=1)           # stereo input works the same way
    assert np.array_equal(k.find_onsets(st, fs), on)
    assert len(k.find_onsets(np.zeros(44100), fs)) == 0, "onsets found in digital silence"
    notes = ["clicks: max error %.2f ms" % err_ms.max()]
    for c in ctx.clips(classes=["transient"]):
        fs, y = k.read_wav(c["path"])
        o = k.find_onsets(y, fs)
        notes.append("%s: %d onsets" % (c["file"], len(o)))
    return "; ".join(notes)


@check("preecho.definition", "pre_echo_db: error only BEFORE the onsets scores higher than error only AFTER them")
def pe_definition(ctx):
    import numpy as np
    k = ctx.kit_module()
    fs = 44100
    rng = np.random.default_rng(1)
    x = rng.standard_normal(200000) * 1e-3
    pos = [20000, 70000, 120000]
    for p in pos:
        x[p:p + 4000] += rng.standard_normal(4000) * np.exp(-np.arange(4000) / 600)
    w = int(0.02 * fs)
    before, after = x.copy(), x.copy()
    for p in pos:
        before[p - w:p - 100] += 0.01 * rng.standard_normal(w - 100)
        after[p + 100:p + w] += 0.01 * rng.standard_normal(w - 100)
    delay = 1024                                   # as if decoded by the course coder
    dec_b = np.concatenate([np.zeros(delay), before])
    dec_a = np.concatenate([np.zeros(delay), after])
    rb = k.pre_echo_db(x, dec_b, fs, delay)
    ra = k.pre_echo_db(x, dec_a, fs, delay)
    assert rb["pre_echo_onsets"] == ra["pre_echo_onsets"] == 3, (rb, ra)
    assert rb["pre_echo_db"] > ra["pre_echo_db"] + 20, (rb["pre_echo_db"], ra["pre_echo_db"])
    st = k.pre_echo_db(np.stack([x, x], 1), np.stack([dec_b, dec_b], 1), fs, delay)
    assert abs(st["pre_echo_db"] - rb["pre_echo_db"]) < 1e-9, "stereo with equal channels must give the same value"
    return "noise before: %.1f dB, noise after: %.1f dB (3 onsets)" % (rb["pre_echo_db"], ra["pre_echo_db"])


@check("preecho.coder_n", "coder at N=2048 gives more pre-echo than at N=256 at the same SNR (interpolated)")
def pe_coder(ctx):
    import numpy as np
    k = ctx.kit_module()
    clip = transient_clip(ctx)
    fs, x = k.read_wav(clip["path"])

    def point(n, q):
        o = k.run_coder(clip["path"], q, coder_dir=coder_with_n(ctx, n), workdir=os.path.join(ctx.workdir, "pe_tmp"))
        _, y = k.read_wav(o["decoded"])
        with redirect_stdout(io.StringIO()):
            snr = k.evaluate(x, y, fs, delay=n, block=n)["snr_db"]
        return snr, k.pre_echo_db(x, y, fs, delay=n)

    s256, p256 = point(256, 200)
    pts = sorted(point(2048, q) for q in (100, 200, 400))       # pre-echo of N=2048 vs. SNR
    snrs = [s for s, _ in pts]
    assert snrs[0] <= s256 <= snrs[-1], "N=256 SNR %.2f dB outside the N=2048 range %s" % (s256, snrs)
    pe2048 = float(np.interp(s256, snrs, [p["pre_echo_db"] for _, p in pts]))
    msg = "%s at SNR %.2f dB: N=256 pre-echo %.2f dB, N=2048 %.2f dB (interpolated from quality 100/200/400; %d onsets)" % (
        clip["file"], s256, p256["pre_echo_db"], pe2048, p256["pre_echo_onsets"])
    assert p256["pre_echo_onsets"] >= 3, msg
    assert pe2048 > p256["pre_echo_db"] + 2, msg
    return msg


@check("preecho.sweep_columns", "quality_sweep(extra_metrics=[pre_echo_db]): extra columns at the end, standard columns unchanged")
def pe_sweep(ctx):
    k = ctx.kit_module()
    clip = transient_clip(ctx)
    d = os.path.join(ctx.workdir, "pe_sweep")
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    plain, extra = os.path.join(d, "plain.csv"), os.path.join(d, "extra.csv")
    with redirect_stdout(io.StringIO()):
        k.quality_sweep([clip], (100,), out_csv=plain, append=False, workdir=os.path.join(d, "tmp"))
        k.quality_sweep([clip], (100,), out_csv=extra, append=False, workdir=os.path.join(d, "tmp"),
                        extra_metrics=[k.pre_echo_db])
        # appending works with and without the extra metric; an unknown extra column is refused
        k.quality_sweep([clip], (200,), out_csv=extra, workdir=os.path.join(d, "tmp"), extra_metrics=[k.pre_echo_db])
        k.quality_sweep([clip], (400,), out_csv=extra, workdir=os.path.join(d, "tmp"))
    with open(plain, newline="") as f:
        h_plain = next(csv.reader(f))
    with open(extra, newline="") as f:
        h_extra = next(csv.reader(f))
    assert h_plain == k.FIELDS, h_plain
    assert h_extra == k.FIELDS + ["pre_echo_db", "pre_echo_onsets"], h_extra
    a, b = read_csv(plain)[0], read_csv(extra)
    skip = ("encode_s", "decode_s")              # run times differ from run to run
    assert all(a[c] == b[0][c] for c in k.FIELDS if c not in skip), "standard columns differ"
    assert b[0]["pre_echo_db"] and b[1]["pre_echo_db"] and b[2]["pre_echo_db"] == "", b
    try:
        def other(o, dd, fs, delay):
            return 1.0
        with redirect_stdout(io.StringIO()):
            k.quality_sweep([clip], (100,), out_csv=extra, workdir=os.path.join(d, "tmp"), extra_metrics=[other])
        raise AssertionError("a new extra column was appended to a file without it")
    except ValueError:
        pass
    import matplotlib
    matplotlib.use("Agg")
    k.plot_rate_quality(extra)
    return "header: ... compressed_bytes, %s; pre-echo at q=100: %s dB" % (", ".join(h_extra[len(k.FIELDS):]), b[0]["pre_echo_db"])


# ------------------------------------------------------------------ WP3: topic3/ (low-delay filter bank)

def _topic3(ctx):
    """Import topic3/ldfb_filebased.py of the kit; clone the filter bank repository (AC_FB_DIR)."""
    import importlib.util
    from topic_smoke import FBC, need_repo
    os.environ["AC_FB_DIR"] = need_repo(ctx, FBC)
    path = os.path.join(ctx.kit, "topic3", "ldfb_filebased.py")
    if not os.path.isfile(path):
        raise Skip("topic3/ldfb_filebased.py not in the kit")
    spec = importlib.util.spec_from_file_location("ldfb_filebased", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@check("topic3.filterbank", "LDFBanafb/LDFBsynfb: N=128..2048 perfect reconstruction, delay N, MDCTanafb conventions")
def t3_fb(ctx):
    import sys
    import numpy as np
    L = _topic3(ctx)
    notes = []
    for N in L.SUPPORTED_N:
        c = L.ldfb_coefficients(N)
        assert len(c) == 2.5 * N
        ana = lambda x, n: L.LDFBanafb(x, n, c)          # noqa: E731
        syn = lambda y: L.LDFBsynfb(y, c)                # noqa: E731
        d = L.block_delay(ana, syn, N)
        snr = L.reconstruction_snr_db(ana, syn, N, d)
        assert d == N and snr > 250, "N=%d: delay %d, SNR %.1f dB" % (N, d, snr)
        notes.append("N=%d %.0f dB" % (N, snr))
    # same shapes as the coder's MDCTanafb/MDCTsynfb; the block-wise MDCT equals MDCTanafb
    sys.path.insert(0, ctx.repo("Python-Audio-Coder"))
    try:
        from MDCTfb import MDCTanafb, MDCTsynfb
    finally:
        sys.path.pop(0)
    N = 256
    x = np.random.default_rng(0).standard_normal(10 * N + 7)
    fb = L.sine_window(N)
    y_ref, y_ldfb, y_blk = MDCTanafb(x, N, fb), L.LDFBanafb(x, N, L.ldfb_coefficients(N)), L.MDCTanafb_blocks(x, N)
    assert y_ref.shape == y_ldfb.shape == y_blk.shape, (y_ref.shape, y_ldfb.shape)
    assert np.allclose(y_ref, y_blk, atol=1e-10), "MDCTclass differs from MDCTanafb"
    assert MDCTsynfb(y_ref, fb).shape == L.LDFBsynfb(y_ldfb, L.ldfb_coefficients(N)).shape
    # one call does not change the next one (no memory kept between files or channels)
    a = L.LDFBanafb(x, N, L.ldfb_coefficients(N))
    assert np.array_equal(a, y_ldfb)
    return "reconstruction SNR: " + ", ".join(notes) + "; shapes as MDCTanafb/MDCTsynfb"


@check("topic3.report", "filterbank_report(512, 1024): delays measured with impulses, SNR, stopband attenuation")
def t3_report(ctx):
    L = _topic3(ctx)
    lines = []
    for N in (512, 1024):
        with redirect_stdout(io.StringIO()):
            r = L.filterbank_report(N)
        for name in ("LDFB", "MDCT"):
            v = r[name]
            assert v["block_delay"] == N and v["system_delay"] == 2 * N - 1, v
            assert v["reconstruction_snr_db"] > 250, v
            lines.append("%s N=%d: system delay %d (%.1f ms), SNR %.0f dB, %d taps, stopband %.1f/%.1f dB" % (
                name, N, v["system_delay"], v["system_delay_ms"], v["reconstruction_snr_db"], v["filter_length"],
                v["stopband_att_2bands_db"], v["stopband_att_5bands_db"]))
        assert r["LDFB"]["filter_length"] == 2 * r["MDCT"]["filter_length"]
    return "\n".join(lines)


@check("topic3.coder", "audio_encoder_ldfb.py/audio_decoder_ldfb.py: byte-identical with mdct; LDFB N=512 in the coder")
def t3_coder(ctx):
    import filecmp
    k = ctx.kit_module()
    L = _topic3(ctx)
    d = os.path.join(ctx.workdir, "topic3_coder")
    shutil.rmtree(d, ignore_errors=True)
    shutil.copytree(ctx.coder, d)
    L.install(d)
    clip = ctx.clips(split="tuning", classes=["speech"])[0]["path"]
    fs, x = k.read_wav(clip)
    w = os.path.join(ctx.workdir, "topic3_runs")
    args = dict(coder_dir=d, encoder="audio_encoder_ldfb.py", decoder="audio_decoder_ldfb.py")
    a = k.run_coder(clip, 100, coder_dir=d, workdir=os.path.join(w, "orig"))
    b = k.run_coder(clip, 100, workdir=os.path.join(w, "copy"), **args)
    c = k.run_coder(clip, 100, workdir=os.path.join(w, "copy_mdct"), extra_args=["--filterbank", "mdct"], **args)
    for o in (b, c):
        assert filecmp.cmp(a["compressed"], o["compressed"], shallow=False), "compressed file differs"
        assert filecmp.cmp(a["decoded"], o["decoded"], shallow=False), "decoded file differs"
    notes = ["defaults byte-identical"]
    for fbank in ("mdct", "ldfb"):
        o = k.run_coder(clip, 100, workdir=os.path.join(w, fbank), extra_args=["--filterbank", fbank, "--N", "512"], **args)
        _, y = k.read_wav(o["decoded"])
        dl = k.find_delay(x, y)
        with redirect_stdout(io.StringIO()):
            r = k.evaluate(x, y, fs, delay=512, block=512)
        assert dl == 512 and r["snr_db"] > 5, (fbank, dl, r)
        notes.append("%s N=512 q=100: delay %d, %.3f bit/sample, SNR %.2f dB" % (
            fbank, dl, k.bits_per_sample(o["compressed"], k.num_samples(x)), r["snr_db"]))
    return "; ".join(notes)


# ------------------------------------------------------------------ WP2: topic5/ (window optimization)

def _topic5(ctx):
    import importlib
    import sys
    d = os.path.join(ctx.kit, "topic5")
    if not os.path.isfile(os.path.join(d, "window_design.py")):
        raise Skip("topic5/window_design.py not in the kit")
    if d not in sys.path:
        sys.path.insert(0, d)
    import window_design
    return importlib.reload(window_design)


@check("topic5.window", "window_from_params: p=0 is the sine window; random p: perfect reconstruction in MDCTanafb/MDCTsynfb")
def t5_window(ctx):
    import sys
    import numpy as np
    W = _topic5(ctx)
    sys.path.insert(0, ctx.repo("Python-Audio-Coder"))
    try:
        from MDCTfb import MDCTanafb, MDCTsynfb
    finally:
        sys.path.pop(0)
    N = 1024
    for K in (4, 8, 12):
        assert np.max(np.abs(W.window_from_params(np.zeros(K), N) - W.sine_window(N))) < 1e-12
    rng = np.random.default_rng(0)
    x = rng.standard_normal(20 * N)
    worst = 0.0
    for _ in range(6):
        p = rng.normal(0, 1.0, int(rng.integers(4, 13)))
        fb = W.window_from_params(p, N)
        assert len(fb) == 1.5 * N and np.all(fb > 0)
        xr = MDCTsynfb(MDCTanafb(x, N, fb), fb)
        worst = max(worst, np.max(np.abs(x[:-2 * N] - xr[N:len(x) - N])))
    assert worst < 1e-9, worst
    return "p=0 equals the sine window; max reconstruction error %.1e for 6 random p (K=4..12)" % worst


@check("topic5.objectives", "stopband_energy improvable by >= 3 dB (L-BFGS-B); ldfb_objective: perfect reconstruction, improvable")
def t5_objectives(ctx):
    import time
    import numpy as np
    from scipy.optimize import minimize
    from topic_smoke import FBC, need_repo
    W = _topic5(ctx)
    N = 1024
    f = W.objective("stopband", N=N)
    t0 = time.perf_counter()
    r = minimize(f, np.zeros(8), method="L-BFGS-B")
    sec = (time.perf_counter() - t0) / r.nfev
    assert r.fun < f(np.zeros(8)) - 3, (f(np.zeros(8)), r.fun)
    notes = ["stopband: sine %.2f dB -> %.2f dB (L-BFGS-B, %d evaluations, %.4f s each)" % (f(np.zeros(8)), r.fun, r.nfev, sec)]
    os.environ["AC_FB_DIR"] = need_repo(ctx, FBC)
    import importlib.util
    spec = importlib.util.spec_from_file_location("ldfb_filebased_t5", os.path.join(ctx.kit, "topic3", "ldfb_filebased.py"))
    L = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(L)
    rng = np.random.default_rng(1)
    for _ in range(3):
        c = W.ldfb_coeff_from_params(rng.normal(0, 0.5, 8), 512)
        ana = lambda x, n: L.LDFBanafb(x, n, c)       # noqa: E731
        snr = L.reconstruction_snr_db(ana, lambda y: L.LDFBsynfb(y, c), 512, 512)
        assert snr > 250, snr
    g = W.objective("ldfb", N=N)
    r2 = minimize(g, np.zeros(8), method="L-BFGS-B", options={"maxfun": 600})
    assert r2.fun < g(np.zeros(8)) - 1, (g(np.zeros(8)), r2.fun)
    notes.append("ldfb: random p reconstruct perfectly; %.2f dB -> %.2f dB (L-BFGS-B, %d evaluations)" % (g(np.zeros(8)), r2.fun, r2.nfev))
    return "\n".join(notes)


@check("topic5.counted", "counted(): exact count across Random Directions' worker processes, budget stop, L-BFGS-B count")
def t5_counted(ctx):
    import sys
    import contextlib as cl
    import numpy as np
    from scipy.optimize import minimize
    from topic_smoke import RD, need_repo
    W = _topic5(ctx)
    sys.path.insert(0, need_repo(ctx, RD))
    try:
        from optimrandomdir_parallel_linesearch import optimrandomdir
    finally:
        sys.path.pop(0)
    d = os.path.join(ctx.workdir, "topic5_counted")
    os.makedirs(d, exist_ok=True)
    f = W.counted(W.objective("stopband", N=1024), os.path.join(d, "rd.log"), budget=300)
    stopped = False
    try:
        with cl.redirect_stdout(io.StringIO()):
            optimrandomdir(f, np.zeros(8), iterations=5000)
    except W.BudgetExhausted:
        stopped = True
    with open(f.log_file) as fh:
        lines = fh.readlines()
    pids = {l.split(",")[2] for l in lines}
    assert stopped, "the budget did not stop Random Directions"
    assert len(f.history()) == 300 and len(lines) >= 300, (len(f.history()), len(lines))
    assert len(lines) - 300 < 64, "too many evaluations after the budget: %d" % (len(lines) - 300)
    assert len(pids) > 1, "all evaluations ran in one process: the parallel case was not tested"
    best, p = f.best()
    assert best < f.f(np.zeros(8)) and abs(f.f(p) - best) < 1e-9
    f.write_csv(os.path.join(d, "rd.csv"), optimizer="random_directions", run=0)
    assert len(read_csv(os.path.join(d, "rd.csv"))) == 300
    g = W.counted(W.objective("stopband", N=1024), os.path.join(d, "lbfgsb.log"))
    r = minimize(g, np.zeros(8), method="L-BFGS-B")
    assert g.count() == r.nfev, (g.count(), r.nfev)
    return "Random Directions: %d evaluations logged from %d processes, stopped at the budget of 300 (%d finished after it); L-BFGS-B: %d = nfev" % (
        len(lines), len(pids), len(lines) - 300, g.count())


@check("topic5.coder", "window coder copies: byte-identical without AC_WINDOW_FILE; coder_objective runs (seconds per evaluation)")
def t5_coder(ctx):
    import filecmp
    import time
    import numpy as np
    k = ctx.kit_module()
    W = _topic5(ctx)
    d = os.path.join(ctx.workdir, "topic5_coder")
    shutil.rmtree(d, ignore_errors=True)
    shutil.copytree(ctx.coder, d)
    W.install(d)
    clip = ctx.clips(split="tuning", classes=["speech"])[0]["path"]
    w = os.path.join(ctx.workdir, "topic5_runs")
    args = dict(coder_dir=d, encoder="audio_encoder_window.py", decoder="audio_decoder_window.py")
    a = k.run_coder(clip, 100, coder_dir=d, workdir=os.path.join(w, "orig"))
    b = k.run_coder(clip, 100, workdir=os.path.join(w, "copy"), **args)
    sine = os.path.join(w, "sine.txt")
    np.savetxt(sine, W.sine_window(1024))
    c = k.run_coder(clip, 100, workdir=os.path.join(w, "sinefile"), env={"AC_WINDOW_FILE": sine}, **args)
    for o in (b, c):
        assert filecmp.cmp(a["compressed"], o["compressed"], shallow=False) and \
            filecmp.cmp(a["decoded"], o["decoded"], shallow=False), "output differs from the original coder"
    with redirect_stdout(io.StringIO()):
        base = W.coder_run(None, 1024, clip, 100, d)
        t0 = time.perf_counter()
        v0 = W.coder_objective(np.zeros(8), 1024, clip, 100, d, baseline=base)
        p = np.array([-0.9, 0, 0, 0, 0, 0, 0, 0])
        v1 = W.coder_objective(p, 1024, clip, 100, d, baseline=base)
        sec = (time.perf_counter() - t0) / 2
    assert abs(v0 - base["bits_per_sample"]) < 1e-9, (v0, base)
    assert v1 != v0, "the window from AC_WINDOW_FILE had no effect"
    return "byte-identical; coder_objective: sine %.4f, p1=-0.9: %.4f (%.1f s per evaluation, %s)" % (
        v0, v1, sec, os.path.basename(clip))


@check("topic5.example", "compare_optimizers.py (2 runs, 500 evaluations): CSV and figure, both optimizers improve")
def t5_example(ctx):
    from topic_smoke import RD, need_repo
    out = os.path.join(ctx.workdir, "topic5_example")
    shutil.rmtree(out, ignore_errors=True)
    log = ctx.run_script([os.path.join(ctx.kit, "topic5", "compare_optimizers.py"), "--runs", "2", "--budget", "500",
                          "--out", out, "--rd-dir", need_repo(ctx, RD)], ctx.workdir, timeout=1800)
    rows = read_csv(os.path.join(out, "best_windows.csv"))
    for r in rows:
        assert float(r["best_value"]) < float(r["start_value"]) - 3, r
    assert os.path.getsize(os.path.join(out, "convergence.png")) > 10000
    return last_lines(log, 1)


# ------------------------------------------------------------------ WP1: baseline.py

def _baseline(ctx, topic, variant):
    out = os.path.join(ctx.workdir, "baseline", "t%d%s.csv" % (topic, variant))
    if os.path.exists(out):
        os.remove(out)
    log = ctx.run_script([os.path.join(ctx.kit, "baseline.py"), "--topic", str(topic), "--variant", variant,
                          "--out", out, "--coder-dir", ctx.coder, "--testset-dir", ctx.testset_dir,
                          "--workdir", os.path.join(ctx.workdir, "baseline", "tmp%d" % topic)], ctx.workdir, timeout=1800)
    assert "git add" in log and "git commit" in log, "no git commands printed"
    rows = read_csv(out)
    assert rows and list(rows[0])[:3] == ["testset", "topic", "variant"], list(rows[0])[:3]
    assert all(r["testset"] == ctx.testset_id and r["topic"] == str(topic) and r["variant"] == variant for r in rows)
    return rows


def _lossy(ctx, topic, variant):
    rows = _baseline(ctx, topic, variant)
    ref = {(r["clip"], r["quality"]): r for r in read_csv(os.path.join(ctx.kit, "reference_results_%s.csv" % ctx.testset_id))}
    classes = sorted({r["class"] for r in rows})
    for r in rows:
        e = ref[(r["clip"], r["quality"])]
        assert abs(float(r["bits_per_sample"]) - float(e["bits_per_sample"])) < 1e-3 and \
            abs(float(r["snr_db"]) - float(e["snr_db"])) < 0.01, "baseline differs from the reference: %s" % r
    assert len(rows) == 4 * len(classes)
    return "%d rows (%s x 4 qualities), match the reference" % (len(rows), ", ".join(classes))


@check("baseline.t2", "baseline.py --topic 2 --variant a (lossy: speech + transient tuning clips)")
def b2(ctx):
    return _lossy(ctx, 2, "a")


@check("baseline.t12", "baseline.py --topic 12 --variant b (lossy, no named classes: speech, transient, polyphonic)")
def b12(ctx):
    return _lossy(ctx, 12, "b")


@check("baseline.t7", "baseline.py --topic 7 --variant a (predictive lossless coder on all tuning clips)")
def b7(ctx):
    rows = _baseline(ctx, 7, "a")
    assert {r["coder"] for r in rows} == {"predictive"} and all(r["exact"] == "True" for r in rows), rows
    return "%d clips, all exact up to the documented end loss" % len(rows)


@check("baseline.t9", "baseline.py --topic 9 --variant a (predictive and IntMDCT lossless coders)")
def b9(ctx):
    rows = _baseline(ctx, 9, "a")
    coders = {r["coder"].split()[0] for r in rows}
    assert coders == {"predictive", "intmdct"} and all(r["exact"] == "True" for r in rows), rows
    return "%d rows, both coders exact up to the documented end loss" % len(rows)


@check("baseline.t15", "baseline.py --topic 15 (Opus on 5 dev-clean utterances; only with RUN_T15=1)")
def b15(ctx):
    if os.environ.get("RUN_T15") != "1":
        raise Skip("set RUN_T15=1 to run it (downloads dev-clean, about 0.35 GB; needs pesq, pystoi, ffmpeg)")
    out = os.path.join(ctx.workdir, "baseline", "t15a.csv")
    ctx.run_script([os.path.join(ctx.kit, "baseline.py"), "--topic", "15", "--variant", "a", "--out", out,
                    "--librispeech-root", os.path.join(ctx.workdir, "LibriSpeech_data")], ctx.workdir, timeout=3600)
    rows = read_csv(out)
    assert len(rows) == 15 and list(rows[0])[:3] == ["testset", "topic", "variant"]
    return "%d rows" % len(rows)
