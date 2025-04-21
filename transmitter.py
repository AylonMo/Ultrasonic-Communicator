from datetime import datetime

import numpy as np
import sounddevice as sd

from consts import *
from utils import string_to_binary


def get_bit_tone(bits_string: str):
    full_wave = np.array([], dtype=np.float32)

    for bit in bits_string:
        bit_freq = BIT_FREQ_MAP[bit]
        time = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
        wave = 0.5 * np.sin((2 * np.pi * bit_freq) * time)
        full_wave = np.concatenate((full_wave, wave.astype(np.float32)))

    return full_wave


def play_bit_tone(bits_string: str):
    print(f"message before padding: {bits_string}")
    print(f"message after padding: {PAD} {bits_string} {PAD}")
    bits_string = PAD + bits_string + PAD
    bits_wave = get_bit_tone(bits_string)

    print(datetime.now())
    sd.play(bits_wave, samplerate=SAMPLE_RATE)
    sd.wait()   # wait for the full signal to finish


if __name__ == "__main__":
    # Option 1 - Transmits text
    bin_message = string_to_binary(MESSAGE)
    play_bit_tone(bin_message)

    # Option 2 - Transmits binary
    # bit_str = "0000011111010101000001111101010100000111110101010000011111010101"
    # play_bit_tone(bit_str)

    print("transmission complete!")
