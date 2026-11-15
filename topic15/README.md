# Topic 15 starter: low-delay neural speech codec

`speech_kit.py` gives every Topic 15 group the same data split and the same measurements. The model itself is your work.

## Setup (Colab)

```python
!pip install -q soundfile pesq pystoi dahuffman
from google.colab import drive; drive.mount('/content/drive')       # keep LibriSpeech on Drive: download once
import sys; sys.path.insert(0, 'starter_kit/topic15')
import speech_kit as s
base = s.download_librispeech(root='/content/drive/MyDrive/LibriSpeech_data')   # 6.3 GB + 0.7 GB, first time only
print(s.check_speakers_disjoint(base))                              # train / dev / test share no speaker
train = s.list_utterances(base, 'train-clean-100')
dev, test = s.list_utterances(base, 'dev-clean'), s.list_utterances(base, 'test-clean')
```

Training data: `s.random_segments(train, n, seconds=1.0, rng=seed)` (numpy) or `s.SegmentDataset(train)` (PyTorch).
Use a GPU runtime for training (Runtime → Change runtime type) and save checkpoints to Drive often; free GPU time is limited.

`torchaudio.datasets.LIBRISPEECH` also works, but since torchaudio 2.9 reading audio needs `pip install torchcodec`. `speech_kit` reads the FLAC files with `soundfile` instead.

## Measurement rules

| What | How |
| --- | --- |
| Data | train-clean-100 for training, dev-clean for choosing settings, test-clean for results (disjoint speakers, checked by `check_speakers_disjoint`) |
| Bitrate | from what is transmitted: `s.bitrate_kbps(s.huffman_bytes(indices), duration)` plus any side information; never from the bottleneck size. `s.entropy_bits()` is only a lower bound |
| Delay | `s.algorithmic_delay_ms(frame_samples, lookahead_samples)`: derive it from your frame length and look-ahead (target below 20 ms) |
| Quality | `s.pesq_wb(ref, decoded)` (wideband PESQ) and `s.stoi_score(ref, decoded)`, on test-clean |
| Speed | `s.real_time_factor(model_fn, x)` on the CPU; report the CPU name it returns |
| Opus baseline | `s.opus_roundtrip(x, kbps=12, frame_ms=10)`: restricted low-delay mode, 12.5 ms algorithmic delay |

Opus on a test utterance (11 Nov 2026, low-delay mode, 10 ms frames, 12.5 ms delay): 8 kbit/s → PESQ 1.62, STOI 0.893; 12 kbit/s → 2.63, 0.947; 16 kbit/s → 3.32, 0.971. With 20 ms frames and its speech mode (26.5 ms delay) Opus reaches PESQ 3.30 at 12 kbit/s: the delay limit costs Opus a lot, which is where a neural codec can win.

The bitrate is measured from the Opus packets; the Ogg file adds about 1 kbit/s of container overhead, which a fair comparison leaves out.

**Check notebook:** `topic15_check.ipynb` runs every step once (about 10 minutes, 0.7 GB download): LibriSpeech, speaker check, Opus baseline, and a deliberately tiny neural codec trained for one minute through to bitrate, PESQ/STOI and real-time factor. Run it on day 1 to make sure your setup works. Tested on 11 Nov 2026 with PyTorch 2.15 on a CPU, on LibriSpeech-shaped data; the real download and GPU training were not tested there.
