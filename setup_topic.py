"""Set up the course code your topic needs, at the versions the starter kit was tested with.

    python starter_kit/setup_topic.py --topic 7          (in Colab: !python starter_kit/setup_topic.py --topic 7)

Run it in your project folder (the folder that contains starter_kit/). For your topic it
  * clones each course repository it needs into this folder and checks out the commit the kit was
    tested with (so everyone works with the same code, also if a repository changes later), and
  * copies the kit files the topic needs into Python-Audio-Coder (START_HERE.md says which).
The cloned folders get no .git folder of their own (a file .kit_source records the repository and
commit instead), so they can go straight into your group's GitLab repository with your changes.
Running it again is safe: folders that are already there are checked, not cloned again.
Options: --list (only print what it would do), --dest FOLDER (instead of the current folder).
"""

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys

KIT = os.path.dirname(os.path.abspath(__file__))
GITHUB = "https://github.com/TUIlmenauAMS/"

PAC = "Python-Audio-Coder"
FBC = "FilterBanks_FastPythonImplementation_Classes"
RD = "LowDelayMultichannelSourceSeparation_Random-Directions_Demo"
BBO = "BlackBoxOptimizerSPcomparison"
ADSP = "ADSP_Tutorials"
APPF = "AdaptivePrePostFilters"

# The commits the kit was tested with (tests/run_checks.py). Change them only together with a new
# run of the kit checks. Python-Audio-Coder and the filter bank classes include the kit's fixes
# (Nov 2026: Colab-safe sound.py, window fix in psyac_quantization.py, psyacloss.py import, ...).
REPOS = {
    PAC: "e012a85f038cb1aaab99c366247021597a26ed5e",
    FBC: "2dc9757fbd847f6256a48ce25f11cee8c26cf0cb",
    RD: "0cf02a27852eaae2e9e21e4f3051f5e60bf42953",
    BBO: "43ad5019d66b7b6f42cad300704776ba3c0c50d8",
    ADSP: "d50a942839b714e550af860006ba2026826b727c",
    APPF: "17452ea32a27637e1bbd4e4bec943adfd0cf4b8e",
}
SPARSE = {ADSP: "audio/fspeech.wav"}     # only this file is needed from ADSP_Tutorials

RICE_LOSSLESS = ["rice_coder/ricecoder.py", "rice_coder/lossless_*.py"]
# per topic: repositories to clone, and (kit files, folder relative to Python-Audio-Coder) to copy
TOPICS = {
    1: ([PAC], []),
    2: ([PAC], []),
    3: ([PAC, FBC], []),
    4: ([PAC, FBC], []),
    5: ([PAC, RD, BBO, FBC], []),           # FBC only for variant b, BBO optional (CMA-ES demo)
    6: ([PAC], [(RICE_LOSSLESS, ".")]),
    7: ([PAC, ADSP], [(RICE_LOSSLESS, ".")]),
    8: ([PAC, ADSP], [(RICE_LOSSLESS, ".")]),
    9: ([PAC], [(["rice_coder/*.py"], ".")]),
    10: ([PAC], []),
    11: ([PAC], []),
    12: ([PAC], []),
    13: ([PAC, APPF], [(["rice_coder/ricecoder.py"], ".")]),   # APPF optional (related work)
    14: ([PAC], []),
    15: ([], []),                            # see topic15/README.md
}


def _git(*args, cwd=None):
    r = subprocess.run(["git"] + list(args), cwd=cwd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("git %s failed:\n%s" % (" ".join(args), (r.stderr or r.stdout)[-1500:]))
    return r.stdout.strip()


MARKER = ".kit_source"     # written into every folder clone() creates, instead of a .git folder


def head(dest):
    """Commit of the course code in `dest`: from its .kit_source file (folders made by clone()), else
    from its .git folder (a git clone), else None."""
    marker = os.path.join(dest, MARKER)
    if os.path.isfile(marker):
        try:
            with open(marker) as f:
                return json.load(f)["commit"]
        except (ValueError, KeyError, OSError):
            return None
    if not os.path.isdir(os.path.join(dest, ".git")):
        return None
    try:
        return _git("rev-parse", "HEAD", cwd=dest)
    except RuntimeError:
        return None


def clone(name, dest=None, base=None):
    """Clone course repository `name` into `dest` (default ./name) at the commit the kit was tested with.

    Only that commit is downloaded. The clone's .git folder is then removed and replaced by a file
    .kit_source (repository, commit), so the folder can be committed to another repository as plain
    files. A folder that is already there is left as it is; a warning says so if it is not at the
    kit's commit. Returns (dest, message)."""
    dest = os.path.abspath(dest or name)
    commit = REPOS[name]
    if os.path.exists(dest):
        h = head(dest)
        if h == commit:
            return dest, "%s: already there, at the kit's commit %s" % (name, commit[:7])
        if h is None:
            return dest, ("%s: WARNING: the folder is there but was not set up by the kit; kept as it is. The kit "
                          "was tested with commit %s." % (name, commit[:7]))
        return dest, ("%s: WARNING: the folder is there at commit %s, the kit was tested with %s.\n"
                      "  To use the kit's version: git -C %s fetch origin %s && git -C %s checkout %s"
                      % (name, h[:7], commit[:7], dest, commit, dest, commit))
    url = (base or os.environ.get("AC_REPO_BASE") or GITHUB) + name
    os.makedirs(dest)
    try:
        _git("init", "-q", dest)
        _git("remote", "add", "origin", url, cwd=dest)
        fetch = ["fetch", "-q", "--depth", "1"]
        if name in SPARSE:
            fetch.append("--filter=blob:none")
            _git("sparse-checkout", "set", "--no-cone", "/" + SPARSE[name], cwd=dest)
        _git(*(fetch + ["origin", commit]), cwd=dest)
        _git("-c", "advice.detachedHead=false", "checkout", "-q", "FETCH_HEAD", cwd=dest)
    except Exception:
        shutil.rmtree(dest, ignore_errors=True)
        raise
    got = head(dest)
    if got != commit:
        shutil.rmtree(dest, ignore_errors=True)
        raise RuntimeError("%s: checked out %s instead of %s" % (name, got, commit))
    # plain files from here on: no nested repository inside the group's GitLab repository
    remove_tree(os.path.join(dest, ".git"))
    with open(os.path.join(dest, MARKER), "w") as f:
        json.dump({"repository": url, "commit": commit}, f)
        f.write("\n")
    return dest, "%s: cloned at commit %s" % (name, commit[:7])


def remove_tree(path):
    """shutil.rmtree that also removes read-only files (git's object files, on Windows). A missing path is fine."""
    import stat
    if not os.path.lexists(path):
        return

    def again(func, p, _):
        os.chmod(p, stat.S_IWRITE)
        func(p)
    if sys.version_info >= (3, 12):
        shutil.rmtree(path, onexc=again)
    else:
        shutil.rmtree(path, onerror=again)


def setup(topic, dest=".", dry_run=False, log=print):
    """Clone the repositories topic `topic` needs into `dest` and copy its kit files. Returns the messages."""
    if topic not in TOPICS:
        raise SystemExit("unknown topic %r: use 1-15" % topic)
    repos, copies = TOPICS[topic]
    msgs = []
    for name in repos:
        target = os.path.join(dest, name)
        if dry_run:
            msgs.append("would clone %s at %s into %s" % (name, REPOS[name][:7], target))
        else:
            msgs.append(clone(name, target)[1])
        log(msgs[-1])
    if name_needed(topic, ADSP) and not dry_run:
        src = os.path.join(dest, ADSP, SPARSE[ADSP])
        shutil.copy(src, os.path.join(dest, PAC, "fspeech.wav"))
        msgs.append("copied %s/%s -> %s/fspeech.wav" % (ADSP, SPARSE[ADSP], PAC)); log(msgs[-1])
    for patterns, rel in copies:
        files = sorted(f for p in patterns for f in glob.glob(os.path.join(KIT, p)))
        if not files:
            raise RuntimeError("kit files %s not found in %s" % (patterns, KIT))
        to = os.path.normpath(os.path.join(dest, PAC, rel))
        for f in files:
            if not dry_run:
                shutil.copy(f, to)
        msgs.append("%s %d kit file(s) (%s) -> %s" % ("would copy" if dry_run else "copied", len(files),
                                                       ", ".join(os.path.basename(f) for f in files), to))
        log(msgs[-1])
    return msgs


def name_needed(topic, name):
    return name in TOPICS[topic][0]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--topic", type=int, required=True, help="your topic number, 1-15")
    ap.add_argument("--dest", default=".", help="project folder (default: the current folder)")
    ap.add_argument("--list", action="store_true", help="only print what would be done")
    a = ap.parse_args(argv)
    setup(a.topic, a.dest, dry_run=a.list)
    if not a.list:
        print("done: topic %d set up in %s" % (a.topic, os.path.abspath(a.dest)))


if __name__ == "__main__":
    sys.exit(main())
