"""Golomb-Rice coder in pure Python + NumPy.

Drop-in replacement for the Rice coding the Python-Audio-Coder lossless coders
used from the packages `bitstream` and `audio.coders`. Those packages do not
install on current Python versions (e.g. Google Colab), this module needs
only NumPy.

The bit format is the same as `audio.coders.rice(b, signed)`:

    [sign bit, signed codes only: 1 = negative]
    [the b lowest bits of |x|, most significant bit first]
    [|x| >> b in unary: that many 1-bits, then one 0-bit]

Bits are packed into bytes most significant bit first, as `bitstream` does.

Usage:

    from ricecoder import rice_encode, rice_decode
    data = rice_encode(x, b=3)                 # x: array of integers -> bytes
    y = rice_decode(data, b=3, count=len(x))   # bytes -> np.int64 array
"""

import numpy as np


def rice_encode_bits(values, b, signed=True):
    """Rice-encode integers; returns the code as an array of bits (uint8, 0/1)."""
    raw = np.asarray(values).ravel()
    v = raw.astype(np.int64)
    if raw.dtype.kind == "f" and not np.array_equal(raw, v):
        raise ValueError("values must be integers (round them first)")
    b = int(b)
    if b < 0:
        raise ValueError("b must be >= 0")
    if not signed and np.any(v < 0):
        raise ValueError("negative value with signed=False")
    if len(v) == 0:
        return np.zeros(0, dtype=np.uint8)
    a = np.abs(v)
    q = a >> b                        # unary part
    r = a & ((1 << b) - 1)            # fixed-width part
    s = 1 if signed else 0
    lengths = s + b + q + 1
    starts = np.concatenate(([0], np.cumsum(lengths)[:-1])).astype(np.int64)
    bits = np.zeros(int(lengths.sum()), dtype=np.uint8)
    if signed:
        bits[starts] = v < 0
    for j in range(b):                # fixed bits, MSB first
        bits[starts + s + j] = (r >> (b - 1 - j)) & 1
    nq = int(q.sum())                 # unary ones; the closing zero is already 0
    if nq:
        first_one = np.repeat(starts + s + b, q)
        offset = np.arange(nq) - np.repeat(np.cumsum(q) - q, q)
        bits[first_one + offset] = 1
    return bits


def rice_encode(values, b, signed=True, drop_partial_byte=False):
    """Rice-encode integers and return the code as bytes.

    The last byte is filled with 0-bits. With drop_partial_byte=True the
    incomplete last byte is dropped instead, as the original course coders
    did (`stream.read(bytes, floor(len(stream)/8))`); this loses the last few
    bits, so the decoder must read fewer values than were encoded.
    """
    bits = rice_encode_bits(values, b, signed)
    if drop_partial_byte:
        bits = bits[: len(bits) // 8 * 8]
    return np.packbits(bits).tobytes()


def rice_decode(data, b, count, signed=True):
    """Decode `count` Rice-coded integers from bytes; returns an np.int64 array.

    Raises ValueError if the data ends before `count` values are decoded.
    """
    bits = np.unpackbits(np.frombuffer(bytes(data), dtype=np.uint8))
    b = int(b)
    count = int(count)
    nbits = len(bits)
    zeros = np.flatnonzero(bits == 0)       # positions of 0-bits, for the unary part
    weights = 1 << np.arange(b - 1, -1, -1, dtype=np.int64)
    out = np.empty(count, dtype=np.int64)
    pos = 0
    for i in range(count):
        if signed:
            if pos >= nbits:
                raise ValueError("not enough data: decoded %d of %d values" % (i, count))
            negative = bits[pos] == 1
            pos += 1
        else:
            negative = False
        if pos + b > nbits:
            raise ValueError("not enough data: decoded %d of %d values" % (i, count))
        fixed = int(bits[pos:pos + b] @ weights) if b else 0
        pos += b
        k = np.searchsorted(zeros, pos)
        if k == len(zeros):
            raise ValueError("not enough data: decoded %d of %d values" % (i, count))
        end = int(zeros[k])
        value = ((end - pos) << b) + fixed
        out[i] = -value if negative else value
        pos = end + 1
    return out
