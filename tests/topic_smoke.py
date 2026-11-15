"""Topic smoke tests: the starting code of each topic runs on current Python/NumPy with the kit's fixes.

Ported from work_packages/reference_tests/run_audit.py and run_audit2.py (absolute paths replaced by
the work folder; the checks that only showed the bugs of the unfixed files are left out, the fixed
versions are kept). Registered with run_checks.py; run them with --only topics.
"""

import os
import shutil

from run_checks import check, fixed_coder_copy, last_lines, pinned_clone

FBC = "FilterBanks_FastPythonImplementation_Classes"
RD = "LowDelayMultichannelSourceSeparation_Random-Directions_Demo"
BBO = "BlackBoxOptimizerSPcomparison"


def need_repo(ctx, name):
    """Clone a course repository into the work folder on first use, at the kit's pinned commit
    (ADSP_Tutorials: only fspeech.wav)."""
    dest = ctx.repo(name)
    pinned_clone(name, dest)
    return dest


SINE = "np.sin(np.pi/(2*N)*(np.arange(int(1.5*N))+0.5))"


# ------------------------------------------------------------------ Topic 1

@check("topics.t1_kbd_pr", "T1 KBD window (first 1.5N of a 2N KBD window): perfect reconstruction in MDCTanafb/MDCTsynfb")
def t1_pr(ctx):
    out = ctx.run_py("""
        import numpy as np, scipy.signal.windows as W
        from MDCTfb import MDCTanafb, MDCTsynfb
        N=1024
        for beta in (4.0, 8.0):
            fb=W.kaiser_bessel_derived(2*N, beta)[:int(1.5*N)]
            x=np.random.default_rng(0).standard_normal(20*N)
            xr=MDCTsynfb(MDCTanafb(x,N,fb),fb)
            err=np.max(np.abs(x[:len(x)-2*N]-xr[N:len(x)-N]))
            print('beta',beta,'max reconstruction error %.1e' % err)
            assert err<1e-9
        """, ctx.repo("Python-Audio-Coder"))
    return last_lines(out)


@check("topics.t1_window_used", "T1 fixed MDCT_psayac_quant_enc() uses the window passed to it")
def t1_window(ctx):
    out = ctx.run_py("""
        import numpy as np, scipy.signal.windows as W, io, contextlib
        from psyac_quantization import MDCT_psayac_quant_enc
        N=1024; fs=44100
        x=np.random.default_rng(1).standard_normal(20*N)*1000
        sine=$SINE
        kbd=W.kaiser_bessel_derived(2*N,4.0)[:int(1.5*N)]
        with contextlib.redirect_stdout(io.StringIO()):
            _,y1,_=MDCT_psayac_quant_enc(x,fs,sine,N); _,y2,_=MDCT_psayac_quant_enc(x,fs,kbd,N)
        assert not np.allclose(y1,y2), 'the encoder ignores its window argument'
        print('MDCT output differs for sine and KBD window: ok')
        """, os.path.join(ctx.coder, "PythonPsychoacoustics"), SINE=SINE)
    return last_lines(out, 1)


@check("topics.t1_regression", "fixed psyac_quantization.py: coder output byte-identical to the original (sine window)")
def t1_regression(ctx):
    clip = ctx.clips(split="tuning")[0]["path"]
    out = ctx.run_py("""
        import sys, filecmp; sys.path.insert(0,'$KIT'); import ac_kit as k
        a=k.run_coder('$CLIP',100,coder_dir='$CODER',workdir='$WORK/reg_fixed')
        b=k.run_coder('$CLIP',100,coder_dir='$REPOS/Python-Audio-Coder',workdir='$WORK/reg_orig')
        c=filecmp.cmp(a['compressed'],b['compressed'],shallow=False); d=filecmp.cmp(a['decoded'],b['decoded'],shallow=False)
        print('compressed identical', c, '| decoded identical', d); assert c and d
        """, ctx.workdir, CLIP=clip)
    return last_lines(out, 1)


@check("topics.t1_kbd_coder", "T1 KBD window end to end in the coder (encoder and decoder changed)", quick=False)
def t1_kbd_coder(ctx):
    d = fixed_coder_copy(ctx, os.path.join(ctx.workdir, "t1_kbd"), edits=[
        (f, "fb=" + SINE, "import scipy.signal.windows as W; fb=W.kaiser_bessel_derived(2*N,4.0)[:int(1.5*N)]")
        for f in ("audio_encoder.py", "audio_decoder.py")])
    clip = ctx.clips(classes=["tonal"], split="tuning")[0]["path"]
    out = ctx.run_py("""
        import sys; sys.path.insert(0,'$KIT'); import ac_kit as k
        fs,x=k.read_wav('$CLIP')
        for q in (100, 100000):
            o=k.run_coder('$CLIP',q,coder_dir='.'); _,y=k.read_wav(o['decoded']); r=k.evaluate(x,y,fs)
            o2=k.run_coder('$CLIP',q,coder_dir='$CODER'); _,y2=k.read_wav(o2['decoded']); r2=k.evaluate(x,y2,fs)
            print('q=%g  KBD: %.3f bit/sample, SNR %.2f dB | sine: %.3f bit/sample, SNR %.2f dB' % (q,
              k.bits_per_sample(o['compressed'],k.num_samples(x)), r['snr_db'], k.bits_per_sample(o2['compressed'],k.num_samples(x)), r2['snr_db']))
            assert abs(r['snr_db']-r2['snr_db']) < 3
        """, d, CLIP=clip)
    return last_lines(out, 2)


# ------------------------------------------------------------------ Topic 2

def _coder_with_n(ctx, n):
    return fixed_coder_copy(ctx, os.path.join(ctx.workdir, "t2_N%d" % n),
                            edits=[(f, "N=1024", "N=%d" % n) for f in ("audio_encoder.py", "audio_decoder.py")])


for _n in (256, 2048):
    def _t2(ctx, n=_n):
        d = _coder_with_n(ctx, n)
        clip = ctx.clips(classes=["polyphonic"], split="tuning")[0]["path"]
        out = ctx.run_py("""
            import sys; sys.path.insert(0,'$KIT'); import ac_kit as k
            fs,x=k.read_wav('$CLIP')
            o=k.run_coder('$CLIP',100,coder_dir='.'); _,y=k.read_wav(o['decoded']); D=k.find_delay(x,y)
            r=k.evaluate(x,y,fs,delay=$N,block=$N)
            print('N=$N: delay',D,'bits/sample %.3f'%k.bits_per_sample(o['compressed'],k.num_samples(x)),'SNR %.2f dB'%r['snr_db'],'percloss %.3g'%r['percloss'])
            assert D==$N
            """, d, CLIP=clip, N=n)
        return last_lines(out, 1)
    check("topics.t2_N%d" % _n, "T2 coder with N=%d: runs, measured delay = N" % _n)(_t2)


# ------------------------------------------------------------------ Topics 3 and 4

def _fbc_fixed(ctx):
    src = need_repo(ctx, FBC)
    d = os.path.join(ctx.workdir, "fbc_fixed")
    shutil.rmtree(d, ignore_errors=True)
    shutil.copytree(src, d, ignore=shutil.ignore_patterns(".git"))
    return d


@check("topics.t3_ldfbclass", "T3 fixed LDFBclass (N=512, interpolated N=1024): perfect reconstruction, block delay")
def t3(ctx):
    out = ctx.run_py("""
        import numpy as np, io, contextlib
        import pyrecplayfastLDFB_classes as C, fbinterp2
        def measure(fbank,N,M=40):
            x=np.random.default_rng(0).standard_normal(M*N); xr=np.zeros_like(x)
            for m in range(M): xr[m*N:(m+1)*N]=fbank.inverse(fbank.forward(x[m*N:(m+1)*N]))
            L=len(x)-4*N; best=min(range(0,4*N),key=lambda d: np.mean((x[:L]-xr[d:d+L])**2))
            e=x[:L]-xr[best:best+L]; return best, 10*np.log10(np.mean(x[:L]**2)/np.mean(e**2))
        fb=np.loadtxt('fb2048t1023d512bbitcs.mat'); fbsym=np.append(fb[:768],fb[1024:])
        for N in (512,1024):
            c=fbsym
            if N!=512:
                with contextlib.redirect_stdout(io.StringIO()): c=fbinterp2.fbinterp(fb=fbsym,n=N/512,N=512)
            d,s=measure(C.LDFBclass(N,c),N); print('LDFB N=%d: delay %d, SNR %.1f dB'%(N,d,s)); assert s>200
        N=1024; d,s=measure(C.MDCTclass(N,np.sin(np.pi/(2*N)*(np.arange(int(1.5*N))+0.5))),N)
        print('MDCT N=1024: delay %d, SNR %.1f dB'%(d,s)); assert s>200
        """, _fbc_fixed(ctx))
    return last_lines(out, 3)


@check("topics.t4_mdctclass", "T4 MDCTclass (original file): block-wise perfect reconstruction")
def t4(ctx):
    out = ctx.run_py("""
        import numpy as np
        import pyrecplayfastLDFB_classes as C
        N=1024; fb=np.sin(np.pi/(2*N)*(np.arange(int(1.5*N))+0.5))
        Mc=C.MDCTclass(N,fb); M=30
        x=np.random.default_rng(0).standard_normal(M*N); xr=np.zeros_like(x)
        for m in range(M):
            xr[m*N:(m+1)*N]=Mc.inverse(Mc.forward(x[m*N:(m+1)*N]))
        best=min(range(0,3*N),key=lambda d: np.mean((x[:len(x)-3*N]-xr[d:d+len(x)-3*N])**2))
        e=x[:len(x)-3*N]-xr[best:best+len(x)-3*N]
        print('delay',best,'max error %.1e' % np.max(np.abs(e)))
        assert np.max(np.abs(e))<1e-9
        """, need_repo(ctx, FBC))
    return last_lines(out, 1)


# ------------------------------------------------------------------ Topic 5

def _t5_dir(ctx):
    return fixed_coder_copy(ctx, os.path.join(ctx.workdir, "t5"))


@check("topics.t5_examples", "T5 fixed optimization_example.py and optimization_exampleLDFB.py run", quick=False)
def t5_examples(ctx):
    d = _t5_dir(ctx)
    a = ctx.run_script(["optimization_example.py"], d, timeout=1200)
    b = ctx.run_script(["optimization_exampleLDFB.py"], d, timeout=1200)
    return "MDCT: %s || LDFB: %s" % (last_lines(a, 1)[:150], last_lines(b, 1)[:150])


@check("topics.t5_randomdir", "T5 optimrandomdir() (Random Directions repo, joblib) on optimfuncMDCT, N=4")
def t5_rd(ctx):
    need_repo(ctx, RD)
    out = ctx.run_py("""
        import sys, numpy as np, io, contextlib
        sys.path.insert(0,'$REPOS/$RD')
        from optimrandomdir_parallel_linesearch import optimrandomdir
        from optimfuncMDCT import optimfuncMDCT
        x0=np.random.default_rng(0).random(6)
        with contextlib.redirect_stdout(io.StringIO()):
            xo=optimrandomdir(optimfuncMDCT, x0, args=(4,), iterations=200)
        print('objective start %.4g -> after 200 RD iterations %.4g'%(optimfuncMDCT(x0,4),optimfuncMDCT(xo,4)))
        assert optimfuncMDCT(xo,4) < optimfuncMDCT(x0,4)
        """, ctx.repo("Python-Audio-Coder"), RD=RD)
    return last_lines(out, 1)


@check("topics.t5_bbo", "T5 bbo_demo_rd_vs_cmaes.py (BlackBoxOptimizerSPcomparison) runs", quick=False)
def t5_bbo(ctx):
    src = need_repo(ctx, BBO)
    d = os.path.join(ctx.workdir, "t5_bbo")
    os.makedirs(d, exist_ok=True)
    shutil.copy(os.path.join(src, "bbo_demo_rd_vs_cmaes.py"), d)
    return last_lines(ctx.run_script(["bbo_demo_rd_vs_cmaes.py"], d, timeout=900), 2)


# ------------------------------------------------------------------ Topic 7

def _t7_dir(ctx):
    d = os.path.join(ctx.workdir, "t7")
    if not os.path.isfile(os.path.join(d, "fspeech.wav")):
        fixed_coder_copy(ctx, d)
        shutil.copy(os.path.join(need_repo(ctx, "ADSP_Tutorials"), "audio", "fspeech.wav"), d)
    return d


for _s, _q in (("lpcexample.py", True), ("lmslosslessexample.py", False), ("wclms_losslessexample.py", False)):
    def _t7(ctx, s=_s):
        return last_lines(ctx.run_script([s], _t7_dir(ctx), timeout=1800), 2)
    check("topics.t7_%s" % _s.split("example")[0].rstrip("_"),
          "T7 %s with the kit's sound.py (no pyaudio) and fspeech.wav from ADSP_Tutorials" % _s, quick=_q)(_t7)


# ------------------------------------------------------------------ Topic 12

@check("topics.t12_nfilts", "T12 coder with nfilts=32 Bark bands runs; kit percloss uses the unmodified model")
def t12(ctx):
    d = fixed_coder_copy(ctx, os.path.join(ctx.workdir, "t12_nfilts"),
                         edits=[(f, "nfilts=64", "nfilts=32") for f in ("audio_encoder.py", "audio_decoder.py")])
    clip = ctx.clips(classes=["speech"], split="tuning")[0]["path"]
    out = ctx.run_py("""
        import sys; sys.path.insert(0,'$KIT'); import ac_kit as k
        fs,x=k.read_wav('$CLIP')
        o=k.run_coder('$CLIP',100,coder_dir='.'); _,y=k.read_wav(o['decoded']); r=k.evaluate(x,y,fs)
        print('nfilts=32: bits/sample %.3f'%k.bits_per_sample(o['compressed'],k.num_samples(x)),'SNR %.2f'%r['snr_db'],'percloss %.3g'%r['percloss'])
        """, d, CLIP=clip)
    return last_lines(out, 1)


# ------------------------------------------------------------------ Topic 13

for _label in ("mono", "stereo"):
    def _t13(ctx, label=_label):
        d = os.path.join(ctx.workdir, "t13")
        if not os.path.isfile(os.path.join(d, "psyacpostfilterFromFile.py")) or label == "mono":
            fixed_coder_copy(ctx, d)
        ch = "1" if label == "mono" else "2"
        clip = [c for c in ctx.clips() if c["channels"] == ch][0]["path"]
        name = "t13" + label
        shutil.copy(clip, os.path.join(d, name + ".wav"))
        ctx.run_script(["psyacprefilterToFile.py", name + ".wav", "100"], d)
        # the pre-filter writes <name>pref.wav (the reference test had a typo here)
        ctx.run_script(["psyacpostfilterFromFile.py", name + "pref.wav"], d)
        made = sorted(f for f in os.listdir(d) if f.startswith(name))
        assert name + "postf.wav" in made, "no post-filtered file: %s" % made
        return "files: " + ", ".join(made)
    check("topics.t13_%s" % _label, "T13 fixed pre-/post-filter chain on a %s clip" % _label)(_t13)
