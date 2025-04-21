from datetime import datetime

import numpy as np
import sounddevice as sd

from consts import *
from utils import string_to_binary
from audio_processing import get_pad_wave


def get_bit_tone(bits_string: str):
    """
    Convert a string of '0'/'1' bits into a concatenated waveform.
    Each bit is a pure sine at ZERO_FREQ or ONE_FREQ for BIT_DURATION seconds.
    """

    full_wave = np.array([], dtype=np.float32)

    for bit in bits_string:
        bit_freq = BIT_FREQ_MAP[bit]
        # generate the sine wave for this single bit
        time = np.linspace(0, BIT_DURATION, int(SAMPLE_RATE * BIT_DURATION), endpoint=False)
        wave = 0.5 * np.sin(2 * np.pi * bit_freq * time)
        full_wave = np.concatenate((full_wave, wave.astype(np.float32)))

    return full_wave


def play_bit_tone(bits_string: str):
    """
    Play out:
      [pad] + [bits_string tones] + [pad]
    Using the system audio output at SAMPLE_RATE.
    """

    print(f"Message before padding: {bits_string}")
    pad_wave = get_pad_wave()               # build pad preamble
    bits_wave = get_bit_tone(bits_string)   # build bit tones
    full_wave = np.concatenate((pad_wave, bits_wave, pad_wave))

    print("Transmitting at", datetime.now())
    sd.play(full_wave, samplerate=SAMPLE_RATE)
    sd.wait()  # block until playback finishes


if __name__ == "__main__":
    # Option 1 – send the text MESSAGE as bits
    bin_message = string_to_binary(MESSAGE)
    play_bit_tone(bin_message)

    # Option 2 – send a raw binary string instead
    # bit_str = "0000011111010101000001111101010100000111110101010000011111010101"
    # play_bit_tone(bit_str)

    print("Transmission complete!")
