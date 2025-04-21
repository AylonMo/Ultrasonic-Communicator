# Sampling & timing
SAMPLE_RATE = 48_000    # samples per second when recording or playing back
BIT_DURATION = 0.04     # seconds per bit tone
LISTEN_DURATION = 12    # seconds to record for each reception

# Bit definitions
ZERO = '0'              # symbol for a “0” bit
ONE = '1'               # symbol for a “1” bit

# Frequencies (Hz) for each bit
ZERO_FREQ = 22_500      # frequency used to encode a '0'
ONE_FREQ = 23_500       # frequency used to encode a '1'
BIT_FREQ_MAP = {ZERO: ZERO_FREQ, ONE: ONE_FREQ}

# Band‑pass filter thresholds (Hz)
FREQ_MIN = 21_000       # lower cutoff, below which we ignore frequency peaks
FREQ_MAX = 24_000       # upper cutoff, above which we ignore

# Padding used to frame the message
BIN_PAD = "1010101110001110101011100011"

# Each sub‑list in PAD is a sequence of frequencies (Hz) for one pad segment
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
MESSAGE = "I buried Paul!"
