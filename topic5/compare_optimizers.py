"""EXAMPLE (not your result): Random Directions vs. L-BFGS-B at the same number of evaluations.

    python starter_kit/topic5/compare_optimizers.py                      # K=8, N=1024, 2000 evaluations, 3 runs
    python starter_kit/topic5/compare_optimizers.py --runs 10 --objective ldfb

Needs a clone of LowDelayMultichannelSourceSeparation_Random-Directions_Demo (for optimrandomdir):
--rd-dir, or the environment variable AC_RD_DIR, or the folder in the current directory.
Writes <out>/convergence.csv (optimizer, run, evaluation, value, best_so_far, seconds),
<out>/best_windows.csv and <out>/convergence.png (best value vs. evaluations, mean and range).

Run r starts both optimizers from the same point: r = 0 from p = 0 (the sine window), the others
from small random parameters (seed r, standard deviation --spread). Every evaluation is counted with window_design.counted,
also the parallel trials of Random Directions and the numerical-gradient evaluations of L-BFGS-B.
L-BFGS-B stops by itself when it has converged (often after a few hundred evaluations); its curve
then stays flat until the budget.
"""

import argparse
import contextlib
import csv
import io
import os
import sys
import time

import numpy as np
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import window_design as W  # noqa: E402

RD_REPO = "LowDelayMultichannelSourceSeparation_Random-Directions_Demo"


def load_optimrandomdir(rd_dir=None):
    for d in (rd_dir, os.environ.get("AC_RD_DIR"), RD_REPO, os.path.join("..", RD_REPO)):
        if d and os.path.isfile(os.path.join(d, "optimrandomdir_parallel_linesearch.py")):
            sys.path.insert(0, os.path.abspath(d))
            from optimrandomdir_parallel_linesearch import optimrandomdir
            return optimrandomdir
    raise FileNotFoundError("optimrandomdir_parallel_linesearch.py not found: clone %s with\n"
                            "  python starter_kit/setup_topic.py --topic 5\n"
                            "and pass --rd-dir or set AC_RD_DIR" % RD_REPO)


def run_random_directions(f, p0, budget, optimrandomdir):
    try:
        with contextlib.redirect_stdout(io.StringIO()):          # optimrandomdir prints a lot
            optimrandomdir(f, p0.copy(), iterations=budget)
    except W.BudgetExhausted:
        pass


def run_lbfgsb(f, p0, budget):
    try:
        minimize(f, p0.copy(), method="L-BFGS-B", options={"maxfun": budget, "maxiter": budget})
    except W.BudgetExhausted:
        pass


def plot(rows, path, objective, budget):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for opt, color in (("random_directions", "C0"), ("lbfgsb", "C1")):
        runs = sorted({r["run"] for r in rows if r["optimizer"] == opt})
        curves = np.full((len(runs), budget), np.nan)
        for i, run in enumerate(runs):
            for r in rows:
                if r["optimizer"] == opt and r["run"] == run:
                    curves[i, r["evaluation"] - 1] = r["best_so_far"]
            # an optimizer that stopped early keeps its best value until the budget
            last = np.where(~np.isnan(curves[i]))[0][-1]
            curves[i, last + 1:] = curves[i, last]
        ev = np.arange(1, budget + 1)
        ax.plot(ev, curves.mean(axis=0), color=color, label="%s (mean of %d runs)" % (
            "Random Directions" if opt == "random_directions" else "L-BFGS-B", len(runs)))
        ax.fill_between(ev, curves.min(axis=0), curves.max(axis=0), color=color, alpha=0.2)
    ax.set_xscale("log")
    ax.set_xlabel("objective evaluations")
    ax.set_ylabel("best %s so far (dB, lower is better)" % objective)
    ax.set_title("Example: convergence (shaded: best and worst run)", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--objective", choices=("stopband", "ldfb"), default="stopband")
    ap.add_argument("--K", type=int, default=8, help="number of window parameters")
    ap.add_argument("--N", type=int, default=1024, help="number of subbands (the coder's N)")
    ap.add_argument("--budget", type=int, default=2000, help="objective evaluations per run")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--spread", type=float, default=0.3,
                    help="standard deviation of the random start parameters of runs 1, 2, ...")
    ap.add_argument("--out", default="topic5_example")
    ap.add_argument("--rd-dir", help="clone of %s" % RD_REPO)
    a = ap.parse_args(argv)
    optimrandomdir = load_optimrandomdir(a.rd_dir)
    os.makedirs(a.out, exist_ok=True)
    obj = W.objective(a.objective, N=a.N)
    rows, best = [], []
    t_all = time.perf_counter()
    for run in range(a.runs):
        p0 = np.zeros(a.K) if run == 0 else np.random.default_rng(run).normal(0, a.spread, a.K)
        for name in ("random_directions", "lbfgsb"):
            f = W.counted(obj, os.path.join(a.out, "log_%s_run%d.txt" % (name, run)), budget=a.budget)
            t0 = time.perf_counter()
            if name == "random_directions":
                run_random_directions(f, p0, a.budget, optimrandomdir)
            else:
                run_lbfgsb(f, p0, a.budget)
            hist = f.history()
            for h in hist:
                rows.append(dict(optimizer=name, run=run, **h))
            val, p = f.best()
            best.append([name, run, len(hist), "%.4f" % obj(p0), "%.4f" % val] + ["%.6f" % v for v in p])
            print("run %d %-18s start %8.3f dB  best %8.3f dB  %5d evaluations  %6.1f s" % (
                run, name, obj(p0), val, len(hist), time.perf_counter() - t0))
    with open(os.path.join(a.out, "convergence.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["optimizer", "run", "evaluation", "value", "best_so_far", "seconds"])
        for r in rows:
            w.writerow([r["optimizer"], r["run"], r["evaluation"], "%.10g" % r["value"],
                        "%.10g" % r["best_so_far"], r["seconds"]])
    with open(os.path.join(a.out, "best_windows.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["optimizer", "run", "evaluations", "start_value", "best_value"] + ["p%d" % (i + 1) for i in range(a.K)])
        w.writerows(best)
    plot(rows, os.path.join(a.out, "convergence.png"), a.objective, a.budget)
    print("wrote %s/convergence.csv, best_windows.csv, convergence.png (%.0f s in total)"
          % (a.out, time.perf_counter() - t_all))


if __name__ == "__main__":
    main()
