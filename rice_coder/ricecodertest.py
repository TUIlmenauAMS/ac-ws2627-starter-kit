#Demo of a Rice encoder (pure-Python version, no audio.coders/bitstream needed)
#after Gerald Schuller, Aug. 2018
from ricecoder import rice_encode_bits, rice_encode, rice_decode
import numpy as np

origs=np.arange(-2,6)
print("Original= ", origs)
#b: exponent of 2
riceencoded=rice_encode_bits(origs, b=1, signed=True)
print("rice encoded=", ''.join(map(str, riceencoded)))

for index in origs:
  print("Index: ", index, "Rice code: ", ''.join(map(str, rice_encode_bits([index], b=1, signed=True))))

ricedecoded=rice_decode(rice_encode(origs, b=1, signed=True), b=1, count=8, signed=True)
print("rice decoded=", ricedecoded)
