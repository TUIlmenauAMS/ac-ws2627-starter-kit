"""Topic 15 (low-delay neural speech codec) -- data and measurement starter.

Everything a group needs around its model, so that all Topic 15 papers measure the same way:

    download_librispeech(subsets, root)      LibriSpeech (CC BY 4.0) from openslr.org, with mirrors
    list_utterances(root, subset)            (flac path, speaker id) for every utterance of a subset
    check_speakers_disjoint(root)            assert that train / dev / test share no speaker
    read_audio(path)                         16 kHz float32 audio, read with soundfile (no torchaudio)
    random_segments(utts, n, seconds, rng)   fixed-length training segments as one numpy array
    SegmentDataset                           the same as a torch Dataset (needs PyTorch)

    bitrate_kbps(n_bytes, duration_s)        bitrate from the transmitted bitstream, in kbit/s
    huffman_bytes(indices)                   actual size of Huffman-coded indices incl. code table
    entropy_bits(indices)                    empirical entropy (a lower bound, not a bitrate)
    algorithmic_delay_ms(frame, lookahead)   delay of a frame-based codec from its frame and look-ahead
    pesq_wb(ref, deg), stoi_score(ref, deg)  speech quality / intelligibility (16 kHz)
    real_time_factor(fn, x, fs)              processing time / audio duration, with the CPU name
    opus_roundtrip(x, fs, kbps, frame_ms)    Opus baseline via ffmpeg/libopus, restricted low delay

Rules for the paper (from the topic description):
  * Bitrate is counted from what is transmitted (the entropy-coded indices + side information),
    never from the bottleneck dimension.
  * Train on train-clean-100, choose settings on dev-clean, report on test-clean (disjoint speakers).
  * Report PESQ (wideband) and STOI; state the CPU used for the real-time factor.

Setup in Colab:  !pip install -q soundfile pesq pystoi dahuffman
LibriSpeech in torchaudio: torchaudio.datasets.LIBRISPEECH still exists, but since torchaudio 2.9
reading audio needs the extra package torchcodec. This module avoids that and reads the FLAC files
with soundfile.
"""

import os
import platform
import shutil
import subprocess
import tarfile
import tempfile
import time
import urllib.request

import numpy as np

FS = 16000
MIRRORS = ("https://openslr.trmal.net/resources/12/",
           "https://openslr.elda.org/resources/12/",
           "https://openslr.magicdatatech.com/resources/12/")
SPLITS = {"train": "train-clean-100", "dev": "dev-clean", "test": "test-clean"}


# ---------------------------------------------------------------- data

def download_librispeech(subsets=("dev-clean", "test-clean", "train-clean-100"), root="LibriSpeech_data"):
    """Download and unpack LibriSpeech subsets (train-clean-100: 6.3 GB, dev/test-clean: about 0.35 GB each).

    Skips subsets that are already unpacked. In Colab, keep `root` on Google Drive
    (e.g. '/content/drive/MyDrive/LibriSpeech_data') so that you download only once.
    Returns the folder that contains the subset folders.
    """
    os.makedirs(root, exist_ok=True)
    base = os.path.join(root, "LibriSpeech")
    for sub in subsets:
        if os.path.isdir(os.path.join(base, sub)):
            continue
        tgz = os.path.join(root, sub + ".tar.gz")
        last_error = None
        for mirror in MIRRORS:
            try:
                print("downloading", mirror + sub + ".tar.gz")
                urllib.request.urlretrieve(mirror + sub + ".tar.gz", tgz + ".part")
                os.replace(tgz + ".part", tgz)
                break
            except Exception as e:  # try the next mirror
                last_error = e
                print("  failed:", e)
        else:
            raise RuntimeError("all LibriSpeech mirrors failed for %s: %s" % (sub, last_error))
        with tarfile.open(tgz) as t:
            t.extractall(root, filter="data") if hasattr(tarfile, "data_filter") else t.extractall(root)
        os.remove(tgz)
    return base


def list_utterances(base, subset):
    """All utterances of a subset as a list of (flac path, speaker id), sorted (reproducible)."""
    folder = os.path.join(base, subset)
    if not os.path.isdir(folder):
        raise FileNotFoundError("%s not found; run download_librispeech first" % folder)
    out = []
    for speaker in sorted(os.listdir(folder)):
        sp = os.path.join(folder, speaker)
        if not os.path.isdir(sp):
            continue
        for chapter in sorted(os.listdir(sp)):
            ch = os.path.join(sp, chapter)
            for f in sorted(os.listdir(ch)):
                if f.endswith(".flac"):
                    out.append((os.path.join(ch, f), speaker))
    return out


def check_speakers_disjoint(base, splits=SPLITS):
    """Assert that the train, dev and test subsets have no speaker in common; returns speaker counts."""
    speakers = {name: {s for _, s in list_utterances(base, sub)} for name, sub in splits.items()
                if os.path.isdir(os.path.join(base, sub))}
    names = list(speakers)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            common = speakers[names[i]] & speakers[names[j]]
            assert not common, "speakers in both %s and %s: %s" % (names[i], names[j], sorted(common)[:5])
    return {n: len(s) for n, s in speakers.items()}


def read_audio(path, fs=FS):
    """Read an audio file as float32 in [-1, 1) (mono); checks the sampling rate."""
    import soundfile as sf
    x, rate = sf.read(path, dtype="float32", always_2d=True)
    if rate != fs:
        raise ValueError("%s has %d Hz, expected %d Hz" % (path, rate, fs))
    return x.mean(axis=1)


def random_segments(utts, n, seconds=1.0, rng=None, fs=FS):
    """n random segments of `seconds` from random utterances -> array (n, samples), float32."""
    rng = np.random.default_rng(rng)
    L = int(seconds * fs)
    out = np.zeros((n, L), dtype=np.float32)
    for i in range(n):
        x = read_audio(utts[rng.integers(len(utts))][0], fs)
        if len(x) <= L:
            out[i, :len(x)] = x
        else:
            s = rng.integers(len(x) - L)
            out[i] = x[s:s + L]
    return out


class SegmentDataset:
    """torch Dataset of random fixed-length segments (one new random segment per item)."""

    def __init__(self, utts, seconds=1.0, items=10000, seed=0, fs=FS):
        import torch  # noqa: F401  (fails early if PyTorch is missing)
        self.utts, self.seconds, self.items, self.seed, self.fs = utts, seconds, items, seed, fs

    def __len__(self):
        return self.items

    def __getitem__(self, i):
        import torch
        seg = random_segments(self.utts, 1, self.seconds, rng=self.seed * 1000003 + i, fs=self.fs)[0]
        return torch.from_numpy(seg)


# ---------------------------------------------------------------- bitrate and delay

def bitrate_kbps(n_bytes, duration_s):
    """Bitrate in kbit/s from the number of transmitted bytes and the audio duration."""
    return 8.0 * n_bytes / duration_s / 1000.0


def entropy_bits(indices):
    """Empirical zeroth-order entropy of the indices, in bits in total. A lower bound for an
    entropy coder with a perfectly known distribution, NOT a bitrate: report huffman_bytes()."""
    _, counts = np.unique(np.asarray(indices).ravel(), return_counts=True)
    p = counts / counts.sum()
    return float(-(counts * np.log2(p)).sum())


def huffman_bytes(indices, include_table=True):
    """Bytes of the Huffman-coded indices (dahuffman), plus the code table the decoder needs
    (2 bytes per table entry for symbol and length, a simple and stated assumption)."""
    from dahuffman import HuffmanCodec
    data = [int(v) for v in np.asarray(indices).ravel()]
    codec = HuffmanCodec.from_data(data)
    n = len(codec.encode(data))
    if include_table:
        n += 2 * len(codec.get_code_table())
    return n


def algorithmic_delay_ms(frame_samples, lookahead_samples=0, fs=FS):
    """Algorithmic delay of a frame-based codec: one frame plus the look-ahead, in ms."""
    return 1000.0 * (frame_samples + lookahead_samples) / fs


# ---------------------------------------------------------------- quality and speed

def pesq_wb(ref, deg, fs=FS):
    """Wideband PESQ (ITU-T P.862.2) of the decoded signal against the original, 16 kHz."""
    from pesq import pesq
    n = min(len(ref), len(deg))
    return float(pesq(fs, np.asarray(ref[:n], dtype=np.float64), np.asarray(deg[:n], dtype=np.float64), "wb"))


def stoi_score(ref, deg, fs=FS):
    """STOI (0..1) of the decoded signal against the original."""
    from pystoi import stoi
    n = min(len(ref), len(deg))
    return float(stoi(ref[:n], deg[:n], fs, extended=False))


def cpu_name():
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def real_time_factor(fn, x, fs=FS, repeats=3):
    """Processing time / audio duration of fn(x) (best of `repeats`), and the CPU it ran on.
    Below 1 means faster than real time. Run your model on the CPU for this (topic rule)."""
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(x)
        best = min(best, time.perf_counter() - t0)
    return {"rtf": best / (len(x) / fs), "cpu": cpu_name()}


# ---------------------------------------------------------------- Opus baseline

def opus_roundtrip(x, fs=FS, kbps=12, frame_ms=10, lowdelay=True):
    """Encode and decode x with Opus (ffmpeg + libopus). Returns dict with the decoded signal,
    the bitrate measured from the Opus packets, and the algorithmic delay.

    lowdelay=True uses Opus's restricted low-delay mode (CELT only): algorithmic delay =
    frame + 2.5 ms look-ahead (Valin et al., arXiv:1602.04845), e.g. 12.5 ms at 10 ms frames.
    lowdelay=False lets Opus use SILK for speech: look-ahead 5 ms + 1.5 ms resampling.
    The bitrate is measured from the packet sizes (Ogg container overhead excluded).
    """
    import soundfile as sf
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found (Colab has it; locally install ffmpeg with libopus)")
    d = tempfile.mkdtemp(prefix="opus_")
    src, enc, dec = (os.path.join(d, n) for n in ("in.wav", "enc.ogg", "dec.wav"))
    sf.write(src, np.asarray(x, dtype=np.float32), fs, subtype="PCM_16")
    app = "lowdelay" if lowdelay else "voip"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", src, "-c:a", "libopus", "-b:a", "%dk" % kbps,
                    "-vbr", "off", "-frame_duration", str(frame_ms), "-application", app, enc], check=True)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", enc, "-ar", str(fs), "-ac", "1", dec], check=True)
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "packet=size",
                        "-of", "csv=p=0", enc], capture_output=True, text=True, check=True)
    packet_bytes = sum(int(line.split(",")[0]) for line in r.stdout.split() if line.split(",")[0].isdigit())
    y, _ = sf.read(dec, dtype="float32")
    look = 2.5 if lowdelay else 6.5
    return {"decoded": y, "kbps": bitrate_kbps(packet_bytes, len(x) / fs), "file_bytes": os.path.getsize(enc),
            "algorithmic_delay_ms": frame_ms + look}
