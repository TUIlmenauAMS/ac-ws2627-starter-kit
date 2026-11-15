"""Your topic's baseline for the 1-page proposal, in one command.

    python starter_kit/baseline.py --topic 7 --variant a --out results/baseline.csv

It runs the UNMODIFIED course coder (with the kit's repo fixes, as the demo notebook sets it up)
on the tuning clips that your topic and variant need, and writes one CSV:

  Topics 1-5, 10-14 (lossy): quality_sweep at quality 50, 100, 200, 400 on the tuning clip of each
      class in CLASSES below. Columns: testset, topic, variant, then the columns of quality_sweep.
  Topics 6-9 (lossless): the predictive lossless coder (topic 9: also the IntMDCT coder, mono clips
      duplicated to stereo) on all tuning clips. Columns: testset, topic, variant, clip, coder,
      bits_per_sample, exact, not_reconstructed_samples, encode_s, decode_s, coded_channels.
  Topic 15: Opus (restricted low delay, 10-ms frames) at 8, 12, 16 kbit/s on 5 dev-clean
      utterances of LibriSpeech (downloaded if needed, about 0.35 GB). Columns: testset, topic,
      variant, utterance, target_kbps, kbps, pesq, stoi, delay_ms. Needs pesq, pystoi and ffmpeg.

Options: --coder-dir (default: $AC_CODER_DIR or ./Python-Audio-Coder), --testset-dir (default:
starter_kit/testset; built with prepare_test_set() if missing, from --sqam-zip if given),
--librispeech-root (topic 15),
--workdir (temporary files). At the end it prints the git commands to commit the CSV.
"""

import argparse
import csv
import os
import shutil
import subprocess
import sys
import tempfile
import time

KIT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, KIT)
import ac_kit as k  # noqa: E402

QUALITIES = (50, 100, 200, 400)
DEFAULT_CLASSES = ("speech", "transient", "polyphonic")

# Classes per topic and variant (from the topic descriptions). Topics whose variants name no
# signal classes use DEFAULT_CLASSES. Topic 14 compares stereo coding methods, so it uses the
# two classes with stereo clips (the speech clips are mono).
CLASSES = {
    (1, "a"): ("tonal", "polyphonic"), (1, "b"): ("speech", "transient"),
    (2, "a"): ("speech", "transient"), (2, "b"): ("tonal", "polyphonic"),
    (11, "a"): ("speech", "transient"), (11, "b"): ("tonal", "polyphonic"),
    (14, "a"): ("stereo", "polyphonic"), (14, "b"): ("stereo", "polyphonic"),
}
LOSSY = (1, 2, 3, 4, 5, 10, 11, 12, 13, 14)
LOSSLESS = (6, 7, 8, 9)


def classes_for(topic, variant):
    return CLASSES.get((topic, variant), DEFAULT_CLASSES)


def _write(out, header, rows):
    d = os.path.dirname(os.path.abspath(out))
    os.makedirs(d, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def lossy_baseline(topic, variant, out, testset_dir, coder_dir, workdir):
    clips = k.load_manifest(testset_dir, split="tuning", classes=classes_for(topic, variant))
    if not clips:
        raise SystemExit("no tuning clips of the classes %s in the test set" % (classes_for(topic, variant),))
    print("topic %d%s: quality_sweep on %s at quality %s" % (topic, variant, ", ".join(c["file"] for c in clips),
                                                           ", ".join(map(str, QUALITIES))))
    tmp = os.path.join(workdir, "sweep.csv")
    rows = k.quality_sweep(clips, QUALITIES, out_csv=tmp, append=False, method="baseline",
                           coder_dir=coder_dir, workdir=os.path.join(workdir, "coder_files"))
    header = ["testset", "topic", "variant"] + [c for c in k.FIELDS if c != "testset"]
    _write(out, header, [dict(r, topic=topic, variant=variant) for r in rows])
    return rows[0]["testset"]


def _run(script, args, cwd):
    t0 = time.perf_counter()
    r = subprocess.run([sys.executable, os.path.join(KIT, "rice_coder", script)] + args, cwd=cwd,
                       capture_output=True, text=True, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
    if r.returncode != 0:
        raise RuntimeError("%s failed:\n%s" % (script, r.stderr[-2000:]))
    return time.perf_counter() - t0


def _lossless_row(x, y, fs, delay, expected_loss, compressed, t_enc, t_dec):
    """exact: every sample before the coder's documented end loss is reconstructed exactly.
    not_reconstructed_samples: samples from the first mismatch (or the end of the decoded file) on."""
    r = k.lossless_check(x, y, delay=delay)
    n = r["first_mismatch"] if r["first_mismatch"] is not None else r["compared"]
    keep = len(x) - expected_loss                 # samples the coder should reconstruct exactly
    trim = r["compared"] - keep
    if trim > 0:
        exact = k.lossless_check(x, y, delay=delay, trim_end=trim)["exact"]
    else:                                         # trim == 0: all compared; trim < 0: more lost than documented
        exact = r["exact"] and trim == 0
    return {"bits_per_sample": round(k.bits_per_sample(compressed, k.num_samples(x)), 5), "exact": bool(exact),
            "not_reconstructed_samples": len(x) - n, "encode_s": round(t_enc, 2), "decode_s": round(t_dec, 2),
            "coded_channels": 1 if x.ndim == 1 else x.shape[1]}


def lossless_baseline(topic, variant, out, testset_dir, workdir):
    import numpy as np
    import scipy.io.wavfile as wavfile
    clips = k.load_manifest(testset_dir, split="tuning")
    rows = []
    for c in clips:
        fs, x = k.read_wav(c["path"])
        name = os.path.splitext(c["file"])[0]
        # predictive coder (NLMS + Rice): loses the remainder of its 20-ms blocks plus 10 samples
        src = os.path.join(workdir, name + "_pred.wav")
        shutil.copy(c["path"], src)
        te = _run("lossless_predictive_audio_encoder.py", [src], workdir)
        td = _run("lossless_predictive_audio_decoder.py", [src[:-4] + ".lacodpred"], workdir)
        _, y = k.read_wav(src[:-4] + "larek.wav")
        blk = int(fs * 20e-3)
        loss = len(x) - ((len(x) + 10) // blk * blk - 10)
        row = _lossless_row(x, y, fs, 0, loss, src[:-4] + ".lacodpred", te, td)
        rows.append(dict(row, clip=c["file"], coder="predictive"))
        print("%-26s predictive  %6.3f bit/sample  exact=%s  not reconstructed: %d" % (
            c["file"], row["bits_per_sample"], row["exact"], row["not_reconstructed_samples"]))
        if topic == 9:
            # IntMDCT coder: needs stereo (a mono clip is duplicated); delay 1024; loses the last 4 blocks
            src = os.path.join(workdir, name + "_intmdct.wav")
            xs = x if x.ndim == 2 else np.stack([x, x], axis=1)
            wavfile.write(src, fs, xs.astype(np.int16))
            te = _run("lossless_rice_audio_encoder.py", [src], workdir)
            td = _run("lossless_rice_audio_decoder.py", [src[:-4] + ".lacodrice"], workdir)
            _, y = k.read_wav(src[:-4] + "larek.wav")
            loss = len(xs) - (len(xs) // 1024 * 1024 - 4 * 1024)
            row = _lossless_row(xs, y, fs, 1024, loss, src[:-4] + ".lacodrice", te, td)
            rows.append(dict(row, clip=c["file"], coder="intmdct" + ("" if x.ndim == 2 else " (mono duplicated)")))
            print("%-26s IntMDCT     %6.3f bit/sample  exact=%s  not reconstructed: %d" % (
                c["file"], row["bits_per_sample"], row["exact"], row["not_reconstructed_samples"]))
    tid = clips[0]["testset"]
    header = ["testset", "topic", "variant", "clip", "coder", "bits_per_sample", "exact", "not_reconstructed_samples",
              "encode_s", "decode_s", "coded_channels"]
    _write(out, header, [dict(r, testset=tid, topic=topic, variant=variant) for r in rows])
    return tid


def speech_baseline(topic, variant, out, root):
    sys.path.insert(0, os.path.join(KIT, "topic15"))
    import speech_kit as s
    base = s.download_librispeech(subsets=("dev-clean",), root=root)
    utts = s.list_utterances(base, "dev-clean")[:5]
    rows = []
    for target in (8, 12, 16):
        for path, _ in utts:
            x = s.read_audio(path)
            o = s.opus_roundtrip(x, kbps=target, frame_ms=10)
            rows.append({"testset": "librispeech-dev-clean", "topic": topic, "variant": variant,
                         "utterance": os.path.basename(path), "target_kbps": target, "kbps": round(o["kbps"], 3),
                         "pesq": round(s.pesq_wb(x, o["decoded"]), 3), "stoi": round(s.stoi_score(x, o["decoded"]), 4),
                         "delay_ms": o["algorithmic_delay_ms"]})
            print("Opus %2d kbit/s  %s  measured %.2f kbit/s  PESQ %.2f  STOI %.3f" % (
                target, rows[-1]["utterance"], rows[-1]["kbps"], rows[-1]["pesq"], rows[-1]["stoi"]))
    _write(out, ["testset", "topic", "variant", "utterance", "target_kbps", "kbps", "pesq", "stoi", "delay_ms"], rows)
    return "librispeech-dev-clean"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--topic", type=int, required=True, choices=range(1, 16), metavar="1..15")
    ap.add_argument("--variant", required=True, choices=("a", "b"))
    ap.add_argument("--out", default="results/baseline.csv")
    ap.add_argument("--coder-dir", help="course coder folder (default: $AC_CODER_DIR or ./Python-Audio-Coder)")
    ap.add_argument("--testset-dir", help="test set folder (default: starter_kit/testset)")
    ap.add_argument("--sqam-zip", help="path of the EBU SQAM zip, if you have it (skips the download)")
    ap.add_argument("--librispeech-root", default="LibriSpeech_data", help="topic 15: where LibriSpeech is (or goes)")
    ap.add_argument("--workdir", help="folder for temporary files (default: a new temporary folder)")
    a = ap.parse_args(argv)
    workdir = os.path.abspath(a.workdir or tempfile.mkdtemp(prefix="baseline_"))
    os.makedirs(workdir, exist_ok=True)
    t0 = time.perf_counter()
    if a.topic == 15:
        tid = speech_baseline(a.topic, a.variant, a.out, a.librispeech_root)
    else:
        testset_dir = a.testset_dir or os.path.join(KIT, "testset")
        if not os.path.isfile(os.path.join(testset_dir, "manifest.csv")):
            k.prepare_test_set(sqam_zip=a.sqam_zip, testset_dir=testset_dir)
        if a.topic in LOSSLESS:
            tid = lossless_baseline(a.topic, a.variant, a.out, testset_dir, workdir)
        else:
            coder_dir = k._coder_dir(a.coder_dir)
            if not os.path.isfile(os.path.join(coder_dir, "audio_encoder.py")):
                raise SystemExit("course coder not found in %s: clone Python-Audio-Coder or pass --coder-dir" % coder_dir)
            tid = lossy_baseline(a.topic, a.variant, a.out, testset_dir, coder_dir, workdir)
    print("\nwrote %s (test set %s) in %.0f s" % (a.out, tid, time.perf_counter() - t0))
    print("Commit it to your GitLab repository (from the repository folder):")
    print("  git add %s" % a.out)
    print('  git commit -m "Baseline for topic %d%s (%s)"' % (a.topic, a.variant, tid))
    print("  git push")
    return 0


if __name__ == "__main__":
    sys.exit(main())
