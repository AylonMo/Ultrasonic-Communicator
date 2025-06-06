import numpy as np

# -------------------------------
# Timing and Sampling Parameters
# -------------------------------

SAMPLE_RATE = 48_000       # Sampling rate in Hz
BIT_DURATION = 0.04        # Duration of a bit in seconds
LISTEN_DURATION = 24       # Maximum duration to record (seconds)


# -------------------------------
# Channel and Frequency Settings
# -------------------------------

N_CHANNELS = 6             # Number of parallel channels (bits per frame)

FREQ_MIN = 20_500          # Minimum frequency used (Hz)
FREQ_MAX = 23_500          # Maximum frequency used (Hz)

# Generate 2*N_CHANNELS frequencies linearly spaced in band
_all_freqs = np.linspace(FREQ_MIN, FREQ_MAX, 2 * N_CHANNELS)

# FREQ_PAIRS[i] = (freq_for_0, freq_for_1) for channel i
FREQ_PAIRS = list(zip(_all_freqs[0::2], _all_freqs[1::2]))


# -------------------------------
# Single-Channel (Legacy) Frequencies
# -------------------------------

ZERO = '0'
ONE = '1'

ZERO_FREQ = 22_500         # Frequency for binary 0
ONE_FREQ = 23_500          # Frequency for binary 1

BIT_FREQ_MAP = {
    ZERO: ZERO_FREQ,
    ONE: ONE_FREQ
}


# -------------------------------
# Padding Sequences (for framing)
# -------------------------------

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

