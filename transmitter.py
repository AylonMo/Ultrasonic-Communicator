import numpy as np
import sounddevice as sd

from consts import *
from utils import string_to_binary


def play_bit_tone(bits_string: str):
    bits_string = PAD + bits_string + PAD + "000"
    print("transmitting...", )
    print("PAD:", PAD)
    print("transmitting:", bits_string)
    print("PAD:", PAD)
    full_wave = np.array([], dtype=np.float32)

    for bit in bits_string:
        bit_freq = BIT_FREQ_MAP[bit]
        time = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
        wave = 0.5 * np.sin((2 * np.pi * bit_freq) * time)
        full_wave = np.concatenate((full_wave, wave.astype(np.float32)))

    sd.play(full_wave, samplerate=SAMPLE_RATE)
    sd.wait()   # wait for the full signal to finish


if __name__ == "__main__":
    # Option 1 - Transmits text
    message = "Hello, Sup!!"
    bin_message = string_to_binary(message)
    play_bit_tone(bin_message)

    # Option 2 - Transmits binary
    # bit_str = "0000011111010101000001111101010100000111110101010000011111010101"
    # play_bit_tone(bit_str)

    print("transmission complete!")
