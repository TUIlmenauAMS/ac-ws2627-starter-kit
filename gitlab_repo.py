"""Put your setup into your group's GitLab repository, and work from that repository in later sessions.

In Colab this is the GitLab cell ("Your group's GitLab repository") of starter_kit_demo.ipynb:

    import getpass, gitlab_repo
    k = gitlab_repo.use_repository(GITLAB_URL, getpass.getpass('GitLab access token: '), GIT_NAME, GIT_EMAIL)

First time (your repository has no starter_kit folder yet): it copies starter_kit/, the course code the
kit set up (Python-Audio-Coder, ...) and results/ into the repository, adds a .gitignore (test
recordings and downloads are never committed), commits and pushes. It prints the commit hash: with
results/baseline.csv in it, that is the baseline commit for your proposal.
Later sessions (the repository already has starter_kit/): it only clones the repository.
Either way it then switches into the repository: the working folder, the kit (`k`) and the coder
(AC_CODER_DIR) all come from your repository, and the test set is prepared there.

Saving your work (do it before you close Colab: Colab forgets everything at the end of a session):

    gitlab_repo.save("what you changed")

It gets your teammates' latest changes, commits yours and pushes. If a teammate changed the same lines,
nothing on the main branch is overwritten: your version goes to a branch of its own and the message
says what to do.

The access token (GitLab: user settings -> Access tokens, scopes read_repository and write_repository)
is kept only for this session, in a file outside the repository: it is never committed.
On your own computer you can pass token=None to use the git login you already have.
"""

import os
import re
import shutil
import subprocess
import sys
import tempfile
from urllib.parse import quote, urlsplit

KIT = os.path.dirname(os.path.abspath(__file__))
if KIT not in sys.path:
    sys.path.insert(0, KIT)
import setup_topic  # noqa: E402  (the list of course repositories and remove_tree)

# never committed: SQAM clips (licence), downloads, caches
GITIGNORE = ["testset/", "TECH3253_SQAM_FLAC.zip", "LibriSpeech_data/", "*.tar.gz",
             "__pycache__/", "*.pyc", ".ipynb_checkpoints/", ".DS_Store"]
# course repositories that go into the group's repository when the kit has set them up
# (ADSP_Tutorials only provides fspeech.wav, which setup_topic.py copies into Python-Audio-Coder)
COURSE_CODE = [n for n in setup_topic.REPOS if n != setup_topic.ADSP]
KIT_MODULES = ("ac_kit", "setup_topic", "psyacloss", "psyacmodel_reference", "sqam_test_set", "make_test_set")

_secret = {"token": None}


def _parse(url):
    """(clean url, repository name) for an https:// GitLab URL (file:// and local paths also work, for tests)."""
    url = (url or "").strip().rstrip("/")
    if not url:
        raise SystemExit("Set GITLAB_URL to your group's repository, e.g. "
                         "'https://gitlab.tu-ilmenau.de/<user>/ac-ws2627-p05b'")
    if url.startswith("git@"):
        raise SystemExit("Use the https address of your repository (GitLab: Code -> Clone with HTTPS), not git@...")
    if url.endswith(".git"):
        url = url[:-4]
    if "://" in url and not url.startswith(("https://", "file://")):
        raise SystemExit("GITLAB_URL must start with https://")
    name = url.split("/")[-1]
    return url, name


def _run(args, cwd=None, check=True):
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0")     # fail instead of waiting for a password prompt
    r = subprocess.run(["git"] + list(args), cwd=cwd, capture_output=True, text=True, env=env)
    out = (r.stdout + r.stderr).strip()
    if _secret["token"]:
        out = out.replace(_secret["token"], "***")
    if check and r.returncode != 0:
        hint = ""
        if "could not read Username" in out:
            hint = ("\nGit got no access token. Run the cell again and paste the token when it asks (on your own "
                    "computer without a token: log in to GitLab with git first).")
        elif any(s in out for s in ("Authentication failed", "HTTP Basic", "403", "401")):
            hint = ("\nCheck the access token (scopes read_repository and write_repository, not expired) and that "
                    "you are a member of the repository.")
        elif "not found" in out.lower() or "does not appear to be a git repository" in out:
            hint = "\nCheck GITLAB_URL: copy it from GitLab (Code -> Clone with HTTPS)."
        cmd = next((a for i, a in enumerate(args) if not a.startswith("-") and (i == 0 or args[i - 1] != "-c")), "")
        raise RuntimeError("git %s failed:\n%s%s" % (cmd, out[-1500:], hint))
    return out


def _credential_args(url, token):
    """git -c options that give `token` to git for this session, from a file outside any repository."""
    if token is None or not url.startswith("https://"):
        return []                                   # your own git login (on your own computer), or a test
    # a pasted token often brings a space or line break with it; git would then ignore the whole line
    token = "".join((token or "").split())
    if not token:
        raise SystemExit("No access token entered: run the cell again and paste your GitLab access token "
                         "(GitLab: user settings -> Access tokens) when it asks.")
    _secret["token"] = token
    host = urlsplit(url).netloc
    path = os.path.join(tempfile.gettempdir(), "ac_gitlab_credentials")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write("https://oauth2:%s@%s\n" % (quote(token, safe=""), host))
    return ["-c", "credential.helper=", "-c", "credential.helper=store --file=%s" % path.replace("\\", "/")]


def connect(url, token=None, name="", email="", parent=None):
    """Clone the repository at `url` into parent/<repository name> (or update it if it is already there).

    `parent` defaults to the folder that contains starter_kit/. Returns the repository folder."""
    url, repo_name = _parse(url)
    parent = os.path.abspath(parent or os.path.dirname(KIT))
    dest = os.path.join(parent, repo_name)
    cred = _credential_args(url, token)
    if os.path.isdir(os.path.join(dest, ".git")):
        if cred:
            _run(["config", "credential.helper", cred[-1].split("=", 1)[1]], cwd=dest)
        if _run(["rev-parse", "--verify", "-q", "HEAD"], cwd=dest, check=False):
            _run(["pull", "-q", "--ff-only"], cwd=dest)
        print("repository %s: updated" % repo_name)
    else:
        if os.path.exists(dest):
            raise SystemExit("%s exists but is not a git repository: rename it, then run this again" % dest)
        _run(cred + ["clone", "-q", url, dest])
        if cred:     # this session's later pulls and pushes (the setting stays in .git, never committed)
            _run(["config", "credential.helper", cred[-1].split("=", 1)[1]], cwd=dest)
        if not _run(["rev-parse", "--verify", "-q", "HEAD"], cwd=dest, check=False):
            # the server's default branch does not exist (yet) but others do: check out main, else the first one
            remote = [b.strip() for b in _run(["branch", "-r"], cwd=dest).splitlines() if "->" not in b and b.strip()]
            if remote:
                br = "origin/main" if "origin/main" in remote else remote[0]
                _run(["checkout", "-q", "-B", br.split("/", 1)[1], br], cwd=dest)
        print("repository %s: cloned" % repo_name)
    if name:
        _run(["config", "user.name", name], cwd=dest)
    if email:
        _run(["config", "user.email", email], cwd=dest)
    return dest


def _copy(src, dst):
    shutil.copytree(src, dst, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns(".git", "testset", "__pycache__", ".ipynb_checkpoints", "*.pyc",
                                                  ".DS_Store"))


def save_setup(repo, project=None):
    """First time: copy starter_kit/, the course code the kit set up and results/ from `project` into
    `repo`, add a .gitignore, commit and push. Returns the commit hash."""
    repo = os.path.abspath(repo)
    project = os.path.abspath(project or os.path.dirname(KIT))
    if os.path.exists(os.path.join(repo, "starter_kit")):
        raise SystemExit("%s already contains starter_kit/: nothing to copy" % repo)
    if os.path.commonpath([repo, project]) == repo:
        raise SystemExit("the project folder %s is inside the repository" % project)
    if not os.path.isdir(os.path.join(project, "starter_kit")):
        raise SystemExit("no starter_kit folder in %s: run the first cell of the demo notebook first" % project)
    if not _run(["config", "user.email"], cwd=repo, check=False):
        raise SystemExit("Set GIT_NAME and GIT_EMAIL (your name and university e-mail, used for your commits) "
                         "and run this again")
    copied = ["starter_kit"]
    _copy(os.path.join(project, "starter_kit"), os.path.join(repo, "starter_kit"))
    for name in COURSE_CODE:
        if os.path.isdir(os.path.join(project, name)):
            _copy(os.path.join(project, name), os.path.join(repo, name))
            copied.append(name)
    if os.path.isdir(os.path.join(project, "results")):
        _copy(os.path.join(project, "results"), os.path.join(repo, "results"))
        copied.append("results")
    gi = os.path.join(repo, ".gitignore")
    have = open(gi).read().splitlines() if os.path.isfile(gi) else []
    missing = [line for line in GITIGNORE if line not in have]
    if missing:
        with open(gi, "a") as f:
            if have and have[-1].strip():
                f.write("\n")
            f.write("# added by the starter kit: never commit the SQAM test recordings or downloads\n")
            f.write("\n".join(missing) + "\n")
    if not _run(["rev-parse", "--verify", "-q", "HEAD"], cwd=repo, check=False):
        _run(["symbolic-ref", "HEAD", "refs/heads/main"], cwd=repo)     # empty repository: branch main
    _run(["add", "-A"], cwd=repo)
    baseline = os.path.isfile(os.path.join(repo, "results", "baseline.csv"))
    import ac_kit
    msg = "Baseline: starter kit %s, course code at the kit's commits%s" % (
        ac_kit.KIT_VERSION, " and baseline run" if baseline else "")
    _run(["commit", "-q", "-m", msg], cwd=repo)
    _run(["push", "-q", "-u", "origin", "HEAD"], cwd=repo)
    commit = _run(["rev-parse", "HEAD"], cwd=repo)
    print("copied into your repository and pushed:", ", ".join(copied))
    if baseline:
        print("Baseline commit (for your proposal: text field and 'Baseline commit' line):", commit)
    else:
        print("Commit:", commit, "- results/baseline.csv was not there yet: run the baseline cell, then commit it")
    return commit


def start(repo):
    """Work from `repo`: working folder, kit and coder from the repository; the test set prepared there.
    Returns the ac_kit module of the repository's kit."""
    repo = os.path.abspath(repo)
    kit = os.path.join(repo, "starter_kit")
    if not os.path.isdir(kit):
        raise SystemExit("%s has no starter_kit folder" % repo)
    old_testset = os.path.join(KIT, "testset")
    os.chdir(repo)
    sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != KIT]
    sys.path.insert(0, kit)
    for m in KIT_MODULES:
        sys.modules.pop(m, None)
    coder = os.path.join(repo, "Python-Audio-Coder")
    if os.path.isdir(coder):
        os.environ["AC_CODER_DIR"] = coder
    import ac_kit
    new_testset = os.path.join(kit, "testset")
    if (not os.path.exists(new_testset) and os.path.isfile(os.path.join(old_testset, "manifest.csv"))
            and os.path.abspath(old_testset) != os.path.abspath(new_testset)):
        shutil.copytree(old_testset, new_testset)     # same clips (checked by hash when built): no new download
    tid = ac_kit.prepare_test_set()
    _tidy_outside(repo, os.path.dirname(KIT))
    print("working in %s (kit %s, test set %s)." % (repo, ac_kit.KIT_VERSION, tid))
    print("Change your files there (Files panel: folder %s)." % os.path.basename(repo))
    print('Save your work before you close Colab:  gitlab_repo.save("what you changed")')
    return ac_kit


SKIP = {".git", "testset", "__pycache__", ".ipynb_checkpoints", ".DS_Store"}


def _differs(outside, inside):
    """Files under `outside` that are missing or different under `inside` (relative paths)."""
    import filecmp
    diff = []
    for root, dirs, files in os.walk(outside):
        dirs[:] = [d for d in dirs if d not in SKIP]
        for f in files:
            if f in SKIP or f.endswith(".pyc"):
                continue
            a = os.path.join(root, f)
            rel = os.path.relpath(a, outside)
            b = os.path.join(inside, rel)
            if not os.path.isfile(b) or not filecmp.cmp(a, b, shallow=False):
                diff.append(rel)
    return diff


def _tidy_outside(repo, project):
    """Clear the copies the notebook set up outside the repository (starter_kit/, course code, results/),
    so that nobody changes a file that is not in the repository. Copies identical to the repository's
    are deleted; others (e.g. the fresh coder of a later session, or a file changed outside by mistake)
    are moved to not_in_your_repository/, so nothing is lost."""
    project, repo = os.path.abspath(project), os.path.abspath(repo)
    if project == repo or os.path.commonpath([project, repo]) != project:
        return
    moved = []
    for name in ["starter_kit", "results"] + COURSE_CODE + [setup_topic.ADSP]:
        outside, inside = os.path.join(project, name), os.path.join(repo, name)
        if not os.path.isdir(outside):
            continue
        if name == setup_topic.ADSP or (os.path.isdir(inside) and not _differs(outside, inside)):
            setup_topic.remove_tree(outside)
            continue
        park = os.path.join(project, "not_in_your_repository")
        os.makedirs(park, exist_ok=True)
        setup_topic.remove_tree(os.path.join(park, name))
        shutil.move(outside, os.path.join(park, name))
        moved.append(name)
    if moved:
        print("Moved %s to not_in_your_repository/: copies outside your repository, never saved to GitLab."
              % ", ".join(moved))


def _toplevel(path=None):
    out = _run(["rev-parse", "--show-toplevel"], cwd=path or os.getcwd(), check=False)
    return out if out and os.path.isdir(out) else None


def save(message, repo=None):
    """Save your work in GitLab: get your teammates' latest changes, commit yours, push.

    `message` says what you changed (e.g. "KBD window with beta = 4 in encoder and decoder").
    If a teammate changed the same lines in the meantime, the main branch is left as it is: your version
    is pushed to a branch of its own (printed), so nothing is lost."""
    message = (message or "").strip()
    if not message:
        raise SystemExit('Say what you changed, e.g. gitlab_repo.save("KBD window in encoder and decoder")')
    repo = repo or _toplevel()
    if not repo or not os.path.isdir(os.path.join(repo, "starter_kit")):
        raise SystemExit("You are not in your group's repository: run the first cell and the GitLab cell of "
                         "the demo notebook first, then save again.")
    if not _run(["config", "user.email"], cwd=repo, check=False):
        raise SystemExit("Set GIT_NAME and GIT_EMAIL in the notebook's GitLab cell, run it, then save again.")
    _run(["add", "-A"], cwd=repo)
    if _run(["diff", "--cached", "--name-only"], cwd=repo):
        _run(["commit", "-q", "-m", message], cwd=repo)
        made = True
    else:
        made = False
    branch = _run(["rev-parse", "--abbrev-ref", "HEAD"], cwd=repo)
    for attempt in (1, 2):                      # a teammate may push in between: try once more
        _run(["fetch", "-q", "origin"], cwd=repo)
        has_remote = bool(_run(["rev-parse", "--verify", "-q", "origin/%s" % branch], cwd=repo, check=False))
        if has_remote:
            merged = subprocess.run(["git", "merge", "-q", "--no-edit", "origin/%s" % branch], cwd=repo,
                                    capture_output=True, text=True, env=dict(os.environ, GIT_TERMINAL_PROMPT="0"))
            if merged.returncode != 0:
                files = _run(["diff", "--name-only", "--diff-filter=U"], cwd=repo, check=False).split()
                _run(["merge", "--abort"], cwd=repo, check=False)
                who = re.sub(r"[^a-z0-9]", "", (_run(["config", "user.name"], cwd=repo, check=False) or "")
                             .split(" ")[0].lower()) or "member"
                side = "save-%s-%s" % (who, _run(["log", "-1", "--format=%cd", "--date=format:%m%d-%H%M"], cwd=repo))
                _run(["push", "-q", "origin", "HEAD:refs/heads/%s" % side], cwd=repo)
                print("Your work is saved in GitLab on the branch '%s', NOT on %s." % (side, branch))
                print("A teammate changed the same lines in: %s" % (", ".join(files) or "(see GitLab)"))
                print("Decide together which version to keep (GitLab shows both: Code -> Branches), then ask in the "
                      "forum if you need help bringing them together.")
                return None
        ahead = _run(["rev-list", "--count", "origin/%s..HEAD" % branch], cwd=repo, check=False) if has_remote else "1"
        if ahead in ("", "0"):
            print("Nothing new to save: GitLab already has everything." if not made else "Saved.")
            return None
        r = subprocess.run(["git", "push", "-q", "-u", "origin", "HEAD"], cwd=repo, capture_output=True, text=True,
                           env=dict(os.environ, GIT_TERMINAL_PROMPT="0"))
        if r.returncode == 0:
            commit = _run(["rev-parse", "HEAD"], cwd=repo)
            print("Saved in GitLab: %s (%s)" % (commit[:8], message if made else "teammates' changes merged"))
            return None
        if attempt == 2:
            _run(["push", "-q", "-u", "origin", "HEAD"], cwd=repo)   # raises with the git message and a hint
    return None


def use_repository(url, token=None, name="", email="", project=None):
    """Clone your group's repository; the first time also copy your setup into it and push (save_setup).
    Then work from the repository (start). Returns the kit module `k`."""
    repo = connect(url, token, name, email, parent=project)
    if not os.path.isdir(os.path.join(repo, "starter_kit")):
        save_setup(repo, project)
    return start(repo)
