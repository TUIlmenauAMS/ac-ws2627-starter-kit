"""Check that the starter kit still works: one command, PASS/FAIL per check.

    python starter_kit/tests/run_checks.py --quick          # about 7 minutes
    python starter_kit/tests/run_checks.py                  # everything, about 9-12 minutes
    python starter_kit/tests/run_checks.py --only rice,errors --workdir ~/kit_checks

The script clones the course repositories into a work folder (default: a new temporary folder;
pass --workdir to reuse the clones next time) at the commits pinned in setup_topic.py, builds the
test set and runs the checks. It never changes the kit itself. Exit code 0 means every check
passed (SKIP does not count as a failure); see tests/README.md for what each check means.

Test set: if the environment variable SQAM_ZIP points to the EBU SQAM zip, the SQAM test set
and reference_results_sqam-v1.csv are used; otherwise the fallback set and
reference_results_librosa-v1.csv.
"""

import argparse
import csv
import io
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
import traceback
import warnings
import zipfile
from contextlib import redirect_stdout
from string import Template

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
KIT = os.path.dirname(TESTS_DIR)
GITHUB = "https://github.com/TUIlmenauAMS/"
QUALITIES = (50, 100, 200, 400)


class Skip(Exception):
    """Raised by a check that cannot run here (missing optional package, environment variable not set)."""


# ------------------------------------------------------------------ context and helpers

class Context:
    """Everything the checks share: paths, options, and results of earlier checks."""

    def __init__(self, workdir, quick):
        self.workdir = os.path.abspath(workdir)
        self.quick = quick
        self.kit = KIT
        self.py = sys.executable
        self.repos = os.path.join(self.workdir, "repos")       # untouched clones
        self.coder = os.path.join(self.workdir, "coder")       # Python-Audio-Coder with the demo's fixes
        self.sqam_zip = os.environ.get("SQAM_ZIP") or None
        self.testset_id = None
        self.testset_dir = None
        self.sweep_rows = None                                 # rows of the reference check
        self.env = dict(os.environ, MPLBACKEND="Agg", PYTHONWARNINGS="ignore",
                        MPLCONFIGDIR=os.path.join(self.workdir, ".mpl"))
        os.makedirs(self.env["MPLCONFIGDIR"], exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", self.env["MPLCONFIGDIR"])   # also for checks run in this process

    def repo(self, name):
        return os.path.join(self.repos, name)

    def kit_module(self):
        """Import ac_kit from the kit being tested (not from anywhere else on sys.path)."""
        if self.kit not in sys.path:
            sys.path.insert(0, self.kit)
        os.environ["AC_CODER_DIR"] = self.coder
        import ac_kit
        return ac_kit

    def clips(self, split=None, classes=None):
        k = self.kit_module()
        return k.load_manifest(self.testset_dir, split=split, classes=classes)

    def run_py(self, code, cwd, timeout=900, **subst):
        """Run a Python snippet in a fresh interpreter (cwd = `cwd`). $KIT, $WORK, $TESTSET, $REPOS
        and the keyword arguments are substituted. Returns the output; raises AssertionError with
        the last lines of the output if the snippet fails."""
        values = dict(KIT=self.kit, WORK=self.workdir, TESTSET=self.testset_dir or "", REPOS=self.repos,
                      CODER=self.coder, **{k: str(v) for k, v in subst.items()})
        code = Template(textwrap.dedent(code)).safe_substitute(values)
        return run_cmd([self.py, "-c", code], cwd, self.env, timeout)

    def run_script(self, args, cwd, timeout=900):
        return run_cmd([self.py] + list(args), cwd, self.env, timeout)


def run_cmd(cmd, cwd, env, timeout):
    try:
        r = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise AssertionError("timed out after %d s" % timeout)
    out = (r.stdout + r.stderr).strip()
    if r.returncode != 0:
        tail = "\n".join([l for l in out.splitlines() if l.strip()][-8:])
        raise AssertionError("exit code %d:\n%s" % (r.returncode, tail))
    return out


def last_lines(text, n=2):
    lines = [l for l in text.splitlines() if l.strip()]
    return " | ".join(lines[-n:])


def pinned_clone(name, dest):
    """Clone course repository `name` into `dest` at the commit pinned in setup_topic.py, as the
    students get it (a clone at another commit is replaced). Returns 'present' or 'cloned'."""
    sys.path.insert(0, KIT)
    import setup_topic
    if setup_topic.head(dest) == setup_topic.REPOS[name]:
        return "present"
    shutil.rmtree(dest, ignore_errors=True)
    setup_topic.clone(name, dest)
    return "cloned"


def fixed_coder_copy(ctx, dest, edits=()):
    """Fresh copy of the coder (pinned commit) in `dest`, with text replacements.
    edits: (file, old, new), `old` must occur once."""
    shutil.rmtree(dest, ignore_errors=True)
    shutil.copytree(ctx.coder, dest)
    for rel, old, new in edits:
        p = os.path.join(dest, rel)
        with open(p) as f:
            t = f.read()
        assert t.count(old) == 1, "%s: expected exactly one '%s'" % (rel, old)
        with open(p, "w") as f:
            f.write(t.replace(old, new))
    return dest


# ------------------------------------------------------------------ check registry

CHECKS = []


def check(cid, title, quick=True):
    """Register a check. cid: short id used by --only; quick=False: left out with --quick."""
    def deco(fn):
        CHECKS.append({"id": cid, "title": title, "fn": fn, "quick": quick})
        return fn
    return deco


# ------------------------------------------------------------------ 1. setup

@check("setup.clone", "clone Python-Audio-Coder at the kit's pinned commit, as the demo notebook does")
def c_clone(ctx):
    os.makedirs(ctx.repos, exist_ok=True)
    state = pinned_clone("Python-Audio-Coder", ctx.repo("Python-Audio-Coder"))
    shutil.rmtree(ctx.coder, ignore_errors=True)
    shutil.copytree(ctx.repo("Python-Audio-Coder"), ctx.coder, ignore=shutil.ignore_patterns(".git"))
    sys.path.insert(0, KIT)
    import setup_topic
    src = ctx.repo("Python-Audio-Coder")
    # plain files: no nested repository in the group's GitLab repository, a .kit_source file instead
    assert not os.path.exists(os.path.join(src, ".git")), "the clone still has a .git folder"
    assert os.path.isfile(os.path.join(src, setup_topic.MARKER)), "no %s file in the clone" % setup_topic.MARKER
    rev = setup_topic.head(src)
    assert rev == setup_topic.REPOS["Python-Audio-Coder"], "marker says %s" % rev
    return "%s (commit %s, plain folder with %s)" % (state, rev[:7], setup_topic.MARKER)


@check("setup.testset", "prepare the test set (SQAM if SQAM_ZIP is set, else the fallback set)")
def c_testset(ctx):
    k = ctx.kit_module()
    if ctx.sqam_zip:
        d = os.path.join(ctx.workdir, "testset_sqam")
        buf = io.StringIO()
        with redirect_stdout(buf):
            ctx.testset_id = k.prepare_test_set(source="sqam", sqam_zip=ctx.sqam_zip, testset_dir=d)
    else:
        d = os.path.join(ctx.workdir, "testset_fallback")
        with redirect_stdout(io.StringIO()):
            ctx.testset_id = k.prepare_test_set(source="fallback", testset_dir=d)
    ctx.testset_dir = d
    n = len(k.load_manifest(d))
    return "%s, %d clips in %s" % (ctx.testset_id, n, d)


# ------------------------------------------------------------------ 2. reference and monotonic checks

def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


@check("reference", "quality_sweep reproduces the reference results (bits 1e-3, SNR 0.01 dB, percloss 1 %)")
def c_reference(ctx):
    k = ctx.kit_module()
    ref_file = os.path.join(ctx.kit, "reference_results_%s.csv" % ctx.testset_id)
    ref = {(r["clip"], float(r["quality"])): r for r in read_csv(ref_file)}
    clips = ctx.clips(split="tuning") if ctx.quick else ctx.clips()
    out_csv = os.path.join(ctx.workdir, "reference_check_%s.csv" % ctx.testset_id)
    with redirect_stdout(io.StringIO()):
        rows = k.quality_sweep(clips, QUALITIES, out_csv=out_csv, append=False,
                               workdir=os.path.join(ctx.workdir, "sweep_tmp"))
    ctx.sweep_rows = rows
    bad = []
    for r in rows:
        key = (r["clip"], float(r["quality"]))
        assert key in ref, "%s q=%s is not in %s" % (key[0], key[1], os.path.basename(ref_file))
        e = ref[key]
        d_bits = abs(float(r["bits_per_sample"]) - float(e["bits_per_sample"]))
        d_snr = abs(float(r["snr_db"]) - float(e["snr_db"]))
        d_pl = abs(float(r["percloss"]) - float(e["percloss"])) / float(e["percloss"])
        if d_bits >= 1e-3 or d_snr >= 0.01 or d_pl >= 0.01:
            bad.append("%s q=%g: bits %.5f vs %s, SNR %.3f vs %s, percloss %s vs %s" % (
                key[0], key[1], float(r["bits_per_sample"]), e["bits_per_sample"], float(r["snr_db"]),
                e["snr_db"], r["percloss"], e["percloss"]))
    assert not bad, "differs from the reference:\n" + "\n".join(bad)
    return "%d clips x %d qualities match (%s)" % (len(clips), len(QUALITIES), os.path.basename(out_csv))


def monotonic_problems(rows):
    by_clip = {}
    for r in rows:
        by_clip.setdefault(r["clip"], []).append(r)
    bad = []
    for clip, rs in sorted(by_clip.items()):
        rs = sorted(rs, key=lambda r: float(r["quality"]))
        for a, b in zip(rs, rs[1:]):
            qa, qb = float(a["quality"]), float(b["quality"])
            if not float(b["bits_per_sample"]) > float(a["bits_per_sample"]):
                bad.append("%s: bitrate does not rise from q=%g to q=%g" % (clip, qa, qb))
            if not float(b["snr_db"]) > float(a["snr_db"]):
                bad.append("%s: SNR does not rise from q=%g to q=%g" % (clip, qa, qb))
            if not float(b["percloss"]) < float(a["percloss"]):
                bad.append("%s: percloss does not fall from q=%g to q=%g" % (clip, qa, qb))
    return bad, len(by_clip)


@check("monotonic", "bitrate and SNR rise, percloss falls with quality (measured and reference files)")
def c_monotonic(ctx):
    if ctx.sweep_rows is None:
        raise AssertionError("needs the rows of the reference check, which did not run")
    bad, n = monotonic_problems(ctx.sweep_rows)
    for name in ("reference_results_sqam-v1.csv", "reference_results_librosa-v1.csv"):
        b, _ = monotonic_problems(read_csv(os.path.join(ctx.kit, name)))
        bad += ["%s: %s" % (name, x) for x in b]
    assert not bad, "\n".join(bad)
    return "%d measured clips and both reference files" % n


# ------------------------------------------------------------------ 3. Rice coder and lossless coders

@check("rice.roundtrip", "Rice coder: round trip b=0..12 signed/unsigned, empty input, non-integer input")
def c_rice(ctx):
    import numpy as np
    sys.path.insert(0, os.path.join(ctx.kit, "rice_coder"))
    try:
        from ricecoder import rice_encode, rice_decode
    finally:
        sys.path.pop(0)
    rng = np.random.default_rng(0)
    n_cases = 0
    for signed in (True, False):
        for b in range(13):
            for trial in range(5):
                x = np.round(rng.laplace(0, 2.0 ** (b + rng.uniform(-2, 2)), int(rng.integers(1, 2000)))).astype(np.int64)
                if trial == 0:
                    x[:] = 0
                if not signed:
                    x = np.abs(x)
                y = rice_decode(rice_encode(x, b, signed), b, len(x), signed)
                assert np.array_equal(x, y), "round trip failed for b=%d signed=%s" % (b, signed)
                n_cases += 1
    assert rice_encode(np.array([], dtype=np.int64), 3) == b""
    assert len(rice_decode(b"", 3, 0)) == 0
    for bad in (np.array([1.5, 2.0]), np.array([0.25])):
        try:
            rice_encode(bad, 2)
        except ValueError:
            pass
        else:
            raise AssertionError("rice_encode accepted non-integer input %s" % bad)
    try:
        rice_encode(np.array([-1, 2]), 2, signed=False)
    except ValueError:
        pass
    else:
        raise AssertionError("rice_encode accepted a negative value with signed=False")
    try:
        rice_decode(rice_encode(np.arange(5), 2), 2, 50)
    except ValueError:
        pass
    else:
        raise AssertionError("rice_decode did not report that the data ran out")
    return "%d random round trips, empty input, ValueError for bad input" % n_cases


def _lossless_coders(ctx):
    d = os.path.join(ctx.workdir, "lossless")
    shutil.rmtree(d, ignore_errors=True)
    shutil.copytree(os.path.join(ctx.kit, "rice_coder"), d, ignore=shutil.ignore_patterns("__pycache__"))
    return d


@check("rice.coders", "both lossless coders: one stereo and one mono clip, exact up to the documented end loss")
def c_lossless(ctx):
    import numpy as np
    import scipy.io.wavfile as wavfile
    k = ctx.kit_module()
    d = _lossless_coders(ctx)
    stereo = [c for c in ctx.clips() if c["channels"] == "2"][0]["path"]
    mono = [c for c in ctx.clips() if c["channels"] == "1"][0]["path"]
    notes = []
    for label, src in (("stereo", stereo), ("mono", mono)):
        fs, x = k.read_wav(src)
        # predictive coder: any channel count; loses the remainder of its 20-ms blocks plus 10 samples
        name = "pred_" + label
        shutil.copy(src, os.path.join(d, name + ".wav"))
        ctx.run_script(["lossless_predictive_audio_encoder.py", name + ".wav"], d)
        ctx.run_script(["lossless_predictive_audio_decoder.py", name + ".lacodpred"], d)
        _, y = k.read_wav(os.path.join(d, name + "larek.wav"))
        blk = int(fs * 20e-3)
        keep = (len(x) + 10) // blk * blk - 10
        r = k.lossless_check(x, y, delay=0, trim_end=len(y) - keep)
        assert r["exact"] and r["compared"] == keep, "predictive coder, %s: %s" % (label, r)
        bits = k.bits_per_sample(os.path.join(d, name + ".lacodpred"), k.num_samples(x))
        notes.append("predictive %s: exact for %d samples, end loss %d, %.2f bit/sample" % (label, keep, len(x) - keep, bits))
        # IntMDCT coder: needs stereo (duplicate a mono channel), delay 1024, loses the last 4 blocks
        name = "intmdct_" + label
        xs = x if x.ndim == 2 else np.stack([x, x], axis=1)
        wavfile.write(os.path.join(d, name + ".wav"), fs, xs.astype(np.int16))
        ctx.run_script(["lossless_rice_audio_encoder.py", name + ".wav"], d)
        ctx.run_script(["lossless_rice_audio_decoder.py", name + ".lacodrice"], d)
        _, y = k.read_wav(os.path.join(d, name + "larek.wav"))
        keep = len(xs) // 1024 * 1024 - 4 * 1024
        r = k.lossless_check(xs, y, delay=1024, trim_end=len(y) - 1024 - keep)
        assert r["exact"] and r["compared"] == keep, "IntMDCT coder, %s: %s" % (label, r)
        bits = k.bits_per_sample(os.path.join(d, name + ".lacodrice"), k.num_samples(xs))
        notes.append("IntMDCT %s: exact for %d samples, end loss %d, %.2f bit/sample" % (label, keep, len(xs) - keep, bits))
    return "; ".join(notes)


# ------------------------------------------------------------------ 4. error handling of ac_kit

def expect_error(exc_type, fn, *args, contains=None, **kwargs):
    try:
        fn(*args, **kwargs)
    except exc_type as e:
        if contains and contains not in str(e):
            raise AssertionError("%s raised, but the message lacks '%s': %s" % (exc_type.__name__, contains, e))
        return str(e)
    raise AssertionError("%s(...) did not raise %s" % (fn.__name__, exc_type.__name__))


@check("errors", "ac_kit error handling: clear errors for the documented mistakes")
def c_errors(ctx):
    import numpy as np
    k = ctx.kit_module()
    x1 = np.ones(5000)
    x2 = np.ones((5000, 2))
    expect_error(ValueError, k.align, x1, x2, contains="channel layouts")
    expect_error(ValueError, k.align, x1, np.ones(500), 1024, contains="nothing to compare")
    expect_error(ValueError, k.evaluate, x2, np.ones((500, 2)), 44100, contains="nothing to compare")
    expect_error(ValueError, k.lossless_check, x1, x1, 0, 5000, contains="leaves nothing")
    expect_error(ValueError, k.find_delay, x1, np.ones(3000), contains="shorter than max_delay")
    clip = ctx.clips(split="tuning")[0]["path"]
    expect_error(RuntimeError, k.run_coder, clip, 100, workdir=os.path.join(ctx.workdir, "err_tmp"),
                 decoded_suffix="_decoded.wav", contains="check decoded_suffix")
    expect_error(RuntimeError, k.run_coder, clip, 100, workdir=os.path.join(ctx.workdir, "err_tmp"),
                 compressed_ext=".mp3", contains="check compressed_ext")
    expect_error(FileNotFoundError, k.run_coder, clip, 100, coder_dir=ctx.workdir, contains="not found")
    other = os.path.join(ctx.workdir, "other_columns.csv")
    with open(other, "w") as f:
        f.write("clip,quality,snr\nx.wav,100,10\n")
    expect_error(ValueError, k.quality_sweep, [clip], (100,), out_csv=other, contains="another kit version")
    # broken SQAM zip: 'auto' falls back with a warning, 'sqam' raises
    broken = os.path.join(ctx.workdir, "broken_sqam.zip")
    with open(broken, "wb") as f:
        f.write(b"PK\x03\x04 this is not a complete zip file")
    d = os.path.join(ctx.workdir, "testset_broken")
    shutil.rmtree(d, ignore_errors=True)
    buf = io.StringIO()
    with redirect_stdout(buf):
        tid = k.prepare_test_set(source="auto", sqam_zip=broken, testset_dir=d)
    assert tid == k.FALLBACK_ID, "expected the fallback test set, got %s" % tid
    assert "WARNING" in buf.getvalue() and "FALLBACK" in buf.getvalue(), "no fallback warning printed"
    shutil.rmtree(d, ignore_errors=True)
    with redirect_stdout(io.StringIO()):
        expect_error(Exception, k.prepare_test_set, source="sqam", sqam_zip=broken, testset_dir=d)
    assert not os.path.exists(os.path.join(d, "manifest.csv")), "a failed SQAM build left a test set behind"
    return "10 error cases"


# ------------------------------------------------------------------ 5. topic smoke tests and 6. notebook

sys.modules.setdefault("run_checks", sys.modules[__name__])   # so that the helper modules share CHECKS
sys.path.insert(0, TESTS_DIR)
import topic_smoke  # noqa: E402  (registers its checks with @check)
import kit_extensions  # noqa: E402  (checks of baseline.py, topic3/, topic5/, pre-echo; skipped if absent)


@check("notebook", "starter_kit_demo.ipynb runs with nbconvert and ends with 'setup matches the reference'")
def c_notebook(ctx):
    try:
        import nbformat
        from nbconvert.preprocessors import ExecutePreprocessor
    except ImportError:
        raise Skip("needs nbconvert and ipykernel: pip install nbconvert ipykernel")
    d = os.path.join(ctx.workdir, "notebook")
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    # the kit as the notebook expects it: starter_kit.zip without testset/ and __pycache__
    zpath = os.path.join(d, "starter_kit.zip")
    make_kit_zip(ctx.kit, zpath)
    shutil.copy(os.path.join(ctx.kit, "starter_kit_demo.ipynb"), d)
    nb = nbformat.read(os.path.join(d, "starter_kit_demo.ipynb"), as_version=4)
    # point the test-set cell at the same test set as the other checks (no 175 MB download here)
    n_changed = 0
    for c in nb.cells:
        if c.cell_type == "code" and "SQAM_ZIP = None" in c.source:
            if ctx.sqam_zip:
                c.source = c.source.replace("SQAM_ZIP = None", "SQAM_ZIP = %r" % ctx.sqam_zip)
            else:
                c.source = c.source.replace("k.prepare_test_set(sqam_zip=SQAM_ZIP)", "k.prepare_test_set(source='fallback')")
            n_changed += 1
        if c.cell_type == "code" and "TOPIC =" in c.source:
            # the "your topic's baseline" cell (WP1): a fast topic, so the check stays short
            c.source = c.source.replace("TOPIC = 1", "TOPIC = 2", 1)
    assert n_changed == 1, "the notebook's test-set cell has changed: update this check"
    # a ready clone (with its .kit_source file, so the notebook sees the pinned commit) saves time;
    # the notebook clones itself when the folder is missing
    shutil.copytree(ctx.repo("Python-Audio-Coder"), os.path.join(d, "Python-Audio-Coder"))
    ep = ExecutePreprocessor(timeout=1800, kernel_name="python3")
    saved = {v: os.environ.get(v) for v in ("MPLBACKEND", "MPLCONFIGDIR")}
    os.environ.update(MPLBACKEND="Agg", MPLCONFIGDIR=ctx.env["MPLCONFIGDIR"])
    how = "nbconvert"
    try:
        ep.preprocess(nb, {"metadata": {"path": d}})
        nbformat.write(nb, os.path.join(d, "executed.ipynb"))
        text = "\n".join(o.get("text", "") for c in nb.cells if c.cell_type == "code"
                         for o in c.get("outputs", []) if o.get("output_type") == "stream")
    except PermissionError:
        # No Jupyter kernel can start where local sockets are forbidden (some sandboxes):
        # run the code cells one after another in one plain Python process instead.
        how = "plain Python, no Jupyter kernel: local sockets are not permitted here"
        text = exec_notebook_cells(ctx, nb, d)
    finally:
        for v, val in saved.items():
            if val is None:
                os.environ.pop(v, None)
            else:
                os.environ[v] = val
    assert "setup matches the reference" in text, "the notebook did not print 'setup matches the reference'"
    line = [l for l in text.splitlines() if "setup matches the reference" in l][0]
    extra = ""
    if any(c.cell_type == "code" and "TOPIC =" in c.source for c in nb.cells):
        assert os.path.isfile(os.path.join(d, "results", "baseline.csv")), "the baseline cell wrote no results/baseline.csv"
        extra = "; baseline cell (topic 2) wrote results/baseline.csv"
    return "%s%s (executed with %s; in %s)" % (line.strip(), extra, how, d)


GITLAB_SNIPPET = """
import os, sys
sys.path.insert(0, os.path.join($PROJECT, 'starter_kit'))
os.environ['AC_CODER_DIR'] = os.path.join($PROJECT, 'Python-Audio-Coder')
import gitlab_repo
k = gitlab_repo.use_repository($URL, None, 'Kit Check', 'kit.check@example.org', project=$PARENT)
assert os.getcwd() == os.path.join($PARENT, $NAME), os.getcwd()
assert os.path.abspath(k.__file__).startswith(os.path.join($PARENT, $NAME)), k.__file__
assert os.environ['AC_CODER_DIR'] == os.path.join($PARENT, $NAME, 'Python-Audio-Coder')
assert len(k.load_manifest()) > 0
print('OK', os.getcwd())
"""


def _git_out(*args, cwd=None):
    return subprocess.run(["git"] + list(args), cwd=cwd, capture_output=True, text=True, check=True).stdout.strip()


@check("gitlab", "notebook's GitLab cell (gitlab_repo.py): first push, later session, save()")
def c_gitlab(ctx):
    base = os.path.join(ctx.workdir, "gitlab")
    shutil.rmtree(base, ignore_errors=True)
    proj = os.path.join(base, "colab_session1")          # what /content looks like after the demo notebook
    os.makedirs(os.path.join(proj, "results"))
    shutil.copytree(ctx.kit, os.path.join(proj, "starter_kit"),
                    ignore=shutil.ignore_patterns(".git", "testset", "__pycache__", ".ipynb_checkpoints"))
    shutil.copytree(ctx.testset_dir, os.path.join(proj, "starter_kit", "testset"))   # built in session 1
    shutil.copytree(ctx.repo("Python-Audio-Coder"), os.path.join(proj, "Python-Audio-Coder"))
    with open(os.path.join(proj, "results", "baseline.csv"), "w") as f:
        f.write("testset,clip\nx,y\n")
    report = []
    for variant in ("empty", "with_readme"):              # GitLab project created empty, or with a README
        bare = os.path.join(base, "remote_%s" % variant, "ac-ws2627-p99a.git")
        os.makedirs(bare)
        _git_out("init", "-q", "--bare", cwd=bare)
        url = "file://" + bare
        if variant == "with_readme":
            seed = os.path.join(base, "seed")
            _git_out("init", "-q", "-b", "main", seed)
            open(os.path.join(seed, "README.md"), "w").write("# group repository\n")
            _git_out("add", "-A", cwd=seed)
            _git_out("-c", "user.name=x", "-c", "user.email=x@example.org", "commit", "-q", "-m", "init", cwd=seed)
            _git_out("push", "-q", url, "main", cwd=seed)
        q = lambda v: repr(v)
        # session 1: copy the setup in, commit, push, then work from the repository
        out = ctx.run_py(GITLAB_SNIPPET, proj, PROJECT=q(proj), PARENT=q(proj), URL=q(url),
                         NAME=q("ac-ws2627-p99a"))
        assert "Baseline commit" in out and "OK" in out, out[-600:]
        files = _git_out("ls-tree", "-r", "--name-only", "main", cwd=bare).splitlines()
        modes = _git_out("ls-tree", "-r", "main", cwd=bare)
        for f in ("starter_kit/ac_kit.py", "starter_kit/gitlab_repo.py", "Python-Audio-Coder/audio_encoder.py",
                  "Python-Audio-Coder/.kit_source", "results/baseline.csv", ".gitignore"):
            assert f in files, "%s missing in the pushed commit" % f
        assert not any(f.startswith("starter_kit/testset/") for f in files), "test recordings were committed"
        assert not any(l.startswith("160000") for l in modes.splitlines()), "a nested repository (gitlink) was committed"
        gi = _git_out("show", "main:.gitignore", cwd=bare)
        assert "testset/" in gi and "TECH3253_SQAM_FLAC.zip" in gi, gi
        if variant == "with_readme":
            assert "README.md" in files, "the existing README was lost"
        # later session, another member: fresh folder, only the kit from GitHub, nothing to copy
        later = os.path.join(base, "colab_session2_%s" % variant)
        shutil.copytree(os.path.join(proj, "starter_kit"), os.path.join(later, "starter_kit"),
                        ignore=shutil.ignore_patterns("testset"))
        shutil.copytree(ctx.testset_dir, os.path.join(later, "starter_kit", "testset"))
        n_before = _git_out("rev-list", "--count", "main", cwd=bare)
        out2 = ctx.run_py(GITLAB_SNIPPET, later, PROJECT=q(later), PARENT=q(later), URL=q(url),
                          NAME=q("ac-ws2627-p99a"))
        assert "cloned" in out2 and "OK" in out2 and "Baseline commit" not in out2, out2[-600:]
        assert _git_out("rev-list", "--count", "main", cwd=bare) == n_before, "the later session committed"
        if variant == "empty":
            report.append(_check_save(ctx, bare, os.path.join(proj, "ac-ws2627-p99a"),
                                      os.path.join(later, "ac-ws2627-p99a")))
        shutil.rmtree(os.path.join(proj, "ac-ws2627-p99a"))
        report.append("%s: %d files pushed" % (variant, len(files)))
    return "; ".join(report)


SAVE_SNIPPET = """
import sys
sys.path.insert(0, 'starter_kit')
import gitlab_repo
gitlab_repo.save($MSG)
"""


def _check_save(ctx, bare, a, b):
    """gitlab_repo.save(): two members' sessions a and b (clones of the same repository)."""
    def edit(repo, rel, old, new):
        p = os.path.join(repo, rel)
        t = open(p).read()
        assert t.count(old) >= 1, "%s: '%s' not found" % (rel, old)
        open(p, "w").write(t.replace(old, new, 1))

    def save(repo, msg):
        return ctx.run_py(SAVE_SNIPPET, repo, MSG=repr(msg))

    def remote_file(rel, branch="main"):
        return _git_out("show", "%s:%s" % (branch, rel), cwd=bare)

    out = save(a, "nothing changed")
    assert "Nothing new to save" in out, out[-300:]
    # b changes the encoder and saves; a changes the decoder and saves: both end up on main
    edit(b, "Python-Audio-Coder/audio_encoder.py", "import", "import  ")
    assert "Saved in GitLab" in save(b, "member B: encoder")
    edit(a, "Python-Audio-Coder/audio_decoder.py", "import", "import  ")
    out = save(a, "member A: decoder")
    assert "Saved in GitLab" in out, out[-300:]
    assert "import  " in remote_file("Python-Audio-Coder/audio_encoder.py"), "B's change lost"
    assert "import  " in remote_file("Python-Audio-Coder/audio_decoder.py"), "A's change lost"
    # both change the same line: the first one wins on main, the second goes to a branch of its own
    head_before = None
    _git_out("-C", b, "pull", "-q")
    edit(b, "starter_kit/START_HERE.md", "# Start here", "# Start here (B)")
    assert "Saved in GitLab" in save(b, "member B: title")
    head_before = _git_out("rev-parse", "main", cwd=bare)
    edit(a, "starter_kit/START_HERE.md", "# Start here", "# Start here (A)")
    out = save(a, "member A: title")
    assert "NOT on main" in out and "START_HERE.md" in out, out[-400:]
    assert _git_out("rev-parse", "main", cwd=bare) == head_before, "main was changed by the conflicting save"
    branches = _git_out("branch", "--list", "save-*", cwd=bare)
    assert branches, "no save-* branch"
    assert "(A)" in remote_file("starter_kit/START_HERE.md", branches.split()[-1]), "A's version not on its branch"
    assert not os.path.exists(os.path.join(a, ".git", "MERGE_HEAD")), "merge left half-done"
    return "save(): nothing to save, two members' changes merged, same-line change kept on %s" % branches.split()[-1]


def exec_notebook_cells(ctx, nb, d):
    """Fallback for check 'notebook': run all code cells in order in one Python process
    (cwd = d), like 'Run all' in a kernel. Raises AssertionError if a cell fails."""
    code = ["import sys", "__cells = []"]
    for i, c in enumerate(nb.cells):
        if c.cell_type == "code":
            code.append("print('### cell %d', flush=True)" % i)
            code.append("exec(compile(%r, 'cell %d', 'exec'))" % (c.source, i))
    script = os.path.join(d, "notebook_cells.py")
    with open(script, "w") as f:
        f.write("\n".join(code) + "\n")
    return run_cmd([ctx.py, script], d, ctx.env, 1800)


def make_kit_zip(kit, zpath):
    """Zip the kit as it goes to the students: folder starter_kit/, without testset/, __pycache__
    and hidden files or folders (.DS_Store, .ipynb_checkpoints, ...)."""
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for root, dirs, files in os.walk(kit):
            rel_root = os.path.relpath(root, kit)
            dirs[:] = [x for x in dirs if x != "__pycache__" and not x.startswith(".")
                       and not (rel_root == "." and x == "testset")]
            for fn in files:
                if fn.startswith(".") or fn.endswith(".pyc"):
                    continue
                full = os.path.join(root, fn)
                z.write(full, os.path.join("starter_kit", os.path.relpath(full, kit)))


# ------------------------------------------------------------------ main

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workdir", help="work folder for clones and outputs (default: a new temporary folder)")
    ap.add_argument("--quick", action="store_true", help="reference check on the tuning clips only; skip slow checks")
    ap.add_argument("--only", help="comma-separated check ids or id prefixes (setup checks always run)")
    ap.add_argument("--list", action="store_true", help="list the checks and exit")
    a = ap.parse_args(argv)
    if a.list:
        for c in CHECKS:
            print("%-24s %s%s" % (c["id"], c["title"], "" if c["quick"] else "   [not in --quick]"))
        return 0
    workdir = a.workdir or tempfile.mkdtemp(prefix="kit_checks_")
    os.makedirs(workdir, exist_ok=True)
    ctx = Context(workdir, a.quick)
    only = [s.strip() for s in a.only.split(",")] if a.only else None
    print("kit:      %s" % KIT)
    print("workdir:  %s" % ctx.workdir)
    print("python:   %s (%s)" % (sys.version.split()[0], sys.executable))
    print("test set: %s" % ("SQAM from SQAM_ZIP=%s" % ctx.sqam_zip if ctx.sqam_zip else "fallback (SQAM_ZIP not set)"))
    print("mode:     %s\n" % ("quick" if a.quick else "full"))
    results = []
    t_all = time.perf_counter()
    setup_ok = True
    for c in CHECKS:
        is_setup = c["id"].startswith("setup.")
        if only and not is_setup and not any(c["id"] == o or c["id"].startswith(o + ".") or c["id"].startswith(o)
                                             for o in only):
            continue
        if a.quick and not c["quick"]:
            results.append((c, "SKIP", 0.0, "not in --quick"))
            print("SKIP  %-24s %s (not in --quick)" % (c["id"], c["title"]))
            continue
        if not setup_ok and not is_setup:
            results.append((c, "FAIL", 0.0, "setup failed"))
            print("FAIL  %-24s setup failed" % c["id"])
            continue
        t0 = time.perf_counter()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                info = c["fn"](ctx) or ""
            status = "PASS"
        except Skip as e:
            status, info = "SKIP", str(e)
        except Exception as e:
            status = "FAIL"
            info = str(e) if isinstance(e, AssertionError) and str(e) else traceback.format_exc(limit=3)
            if is_setup:
                setup_ok = False
        dt = time.perf_counter() - t0
        results.append((c, status, dt, info))
        print("%-5s %-24s %6.1f s  %s" % (status, c["id"], dt, c["title"]))
        if info:
            for line in str(info).strip().splitlines()[:25]:
                print("      " + line)
        sys.stdout.flush()
    n_fail = sum(1 for r in results if r[1] == "FAIL")
    n_pass = sum(1 for r in results if r[1] == "PASS")
    n_skip = sum(1 for r in results if r[1] == "SKIP")
    total = time.perf_counter() - t_all
    print("\n%d passed, %d failed, %d skipped in %.0f s (%.1f min)" % (n_pass, n_fail, n_skip, total, total / 60))
    with open(os.path.join(ctx.workdir, "check_results.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["check", "status", "seconds", "info"])
        for c, s, dt, info in results:
            w.writerow([c["id"], s, "%.1f" % dt, str(info).replace("\n", " | ")[:500]])
    print("results:  %s" % os.path.join(ctx.workdir, "check_results.csv"))
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
