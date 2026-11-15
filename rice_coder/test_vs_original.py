import sys, numpy as np
# Run from this folder in an environment that has the original packages
# (bitstream + audio.coders build on Python 3.11 with NumPy < 2).
sys.path.insert(0, '.')
from ricecoder import rice_encode_bits, rice_encode, rice_decode
from audio.coders import rice
from bitstream import BitStream
rng = np.random.default_rng(0)
cases = 0
for signed in (True, False):
    for b in range(0, 16):
        for trial in range(25):
            n = int(rng.integers(1, 700))
            scale = 2.0 ** (b + rng.uniform(-3, 3))
            x = np.round(rng.laplace(0, scale, n)).astype(np.int32)
            if trial == 0: x[:] = 0
            if not signed: x = np.abs(x)
            codec = rice(b=b, signed=signed)
            st = BitStream(x, codec)
            ref_bits = str(st)
            ref_bytes_trunc = st.copy().read(bytes, int(np.floor(len(st) / 8.0)))
            mine = rice_encode_bits(x, b, signed)
            assert ''.join(map(str, mine)) == ref_bits, ('bits', signed, b, trial)
            assert rice_encode(x, b, signed, drop_partial_byte=True) == ref_bytes_trunc, ('trunc bytes', signed, b)
            full = rice_encode(x, b, signed)
            # original decoder reads our padded bytes
            s2 = BitStream(); s2.write(full)
            assert list(s2.read(codec, n)) == list(x), ('orig decodes mine', signed, b)
            # our decoder reads original bytes and roundtrips
            assert np.array_equal(rice_decode(full, b, n, signed), x), ('roundtrip', signed, b)
            k = len(ref_bytes_trunc)
            if k:
                s3 = BitStream(); s3.write(ref_bytes_trunc)
                # decode as many values as the original can from truncated bytes, compare
                m = 0; ref_vals = []
                try:
                    while True:
                        ref_vals.append(s3.read(codec)); m += 1
                except Exception: pass
                assert np.array_equal(rice_decode(ref_bytes_trunc, b, m, signed), np.array(ref_vals)), ('trunc decode', signed, b)
                try: rice_decode(ref_bytes_trunc, b, m + 1, signed); raise AssertionError('should have run out')
                except ValueError: pass
            cases += 1
print('random-data cases passed:', cases)
