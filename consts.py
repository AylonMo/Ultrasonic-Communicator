# Sampling & timing
SAMPLE_RATE = 48_000    # samples per second when recording or playing back
BIT_DURATION = 0.15     # seconds per bit tone
LISTEN_DURATION = 14    # seconds to record for each reception

# Parallel channels: how many bits to send in parallel per symbol period
N_CHANNELS = 5  # adjust this to your desired level of parallelism

# Frequency band to carve into 2 * N_CHANNELS tones
FREQ_MIN = 21_500       # lower cutoff, below which we ignore frequency peaks
FREQ_MAX = 23_500       # upper cutoff, above which we ignore

# Dynamically generate 2*N_CHANNELS uniformly spaced frequencies
import numpy as np
_all_freqs = np.linspace(FREQ_MIN, FREQ_MAX, 2 * N_CHANNELS)
FREQ_PAIRS = list(zip(_all_freqs[0::2], _all_freqs[1::2]))
# FREQ_PAIRS[i] = (freq_for_0, freq_for_1) for channel i

# Bit definitions
ZERO = '0'              # symbol for a “0” bit
ONE = '1'               # symbol for a “1” bit

# Original single‑channel bit frequencies (for backward compatibility)
ZERO_FREQ = 22_500      # frequency used to encode a '0'
ONE_FREQ = 23_500       # frequency used to encode a '1'
BIT_FREQ_MAP = {ZERO: ZERO_FREQ, ONE: ONE_FREQ}

# Padding used to frame the message\ nBIN_PAD = "1010101110001110101011100011"
PAD = [
    [21551, 21226, 22573, 23493, 22320, 21268],
    [23931, 23200, 23775, 22212, 21484, 23231, 22355, 21239, 21607, 22988],
    [22902, 22284, 22996, 22694, 21913],
    [23126, 21501, 21968, 22945, 21140, 21980, 22041, 22261, 22361],
    [23726, 22710],
    [23737, 22366, 21114, 23099],
    [22431, 22576, 23266, 22380, 21937, 21860, 22163, 22688, 22451, 21746],
    [21073, 23432, 23457, 22297, 21617, 22939, 22298, 23011, 21812],
    [22046, 23577, 22974, 23859, 23139, 21313, 22580],
    [21606, 21683, 22946, 22726]
]

# The actual text message to send/receive
MESSAGE = "The red fox jumped"# over the white hill!"




