from datetime import datetime
import numpy as np
import sounddevice as sd

from consts import SAMPLE_RATE, BIT_DURATION, N_CHANNELS, FREQ_PAIRS, MESSAGE
from utils import string_to_binary
from audio_processing import get_pad_wave
from data_management import add_error_fixer, add_preamble


def split_bits(bits: str, n: int) -> list[str]:
    """
    Round-robin split of a bit string into n parallel channels.
    """
    chans = [''] * n
    for idx, b in enumerate(bits):
        chans[idx % n] += b
    return chans


def get_multitone(chans: list[str]) -> np.ndarray:
    """
    Build a time-domain waveform that, in each BIT_DURATION frame,
    transmits one bit per channel simultaneously using two tones per channel.
    Each frame has fade-in and fade-out to reduce clicking.
    """
    t = np.linspace(0, BIT_DURATION, int(SAMPLE_RATE * BIT_DURATION), endpoint=False)
    full_wave = np.array([], dtype=np.float32)
    L = max(len(c) for c in chans)

    # Precompute fade envelope
    fade_samples = int(0.002 * SAMPLE_RATE)  # 2ms fade (you can adjust)
    fade_in = np.linspace(0, 1, fade_samples)
    fade_out = np.linspace(1, 0, fade_samples)

    for symbol_idx in range(L):
        segment = np.zeros_like(t)
        for i, chan in enumerate(chans):
            bit = chan[symbol_idx] if symbol_idx < len(chan) else '0'
            f0, f1 = FREQ_PAIRS[i]
            freq = f1 if bit == '1' else f0
            segment += 0.5 * np.sin(2 * np.pi * freq * t)
        # normalize
        segment /= N_CHANNELS

        # Apply fade-in and fade-out
        segment[:fade_samples] *= fade_in
        segment[-fade_samples:] *= fade_out

        full_wave = np.concatenate((full_wave, segment.astype(np.float32)))

    return full_wave



def play_bit_tone(bits_string: str) -> None:
    """
    Play: [pad] + [parallel multi-tone data] + [pad]
    """
    print(f"Message before padding: {bits_string}")
    pad_wave = get_pad_wave()
    # split the bit string into N_CHANNELS sub-strings
    chans = split_bits(bits_string, N_CHANNELS)
    # build and concatenate the per-symbol parallel waveform
    data_wave = get_multitone(chans)
    full_wave = np.concatenate((pad_wave, data_wave, pad_wave))

    print("Transmitting at", datetime.now())
    sd.play(full_wave, samplerate=SAMPLE_RATE)
    sd.wait()  # block until playback finishes


if __name__ == "__main__":
    # convert MESSAGE to binary and send
    bin_message = add_preamble(string_to_binary(MESSAGE))
    print("Message is:", bin_message)
    print("Message length: ", len(bin_message))

    bin_message = add_error_fixer(bin_message)
    print("Error safe message length: ", len(bin_message))
    print("Final message:", bin_message)

    play_bit_tone(bin_message)
    print("Transmission complete!")
