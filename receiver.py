import numpy as np
import sounddevice as sd
from scipy.fft import fft
import matplotlib.pyplot as plt

from consts import *
from utils import binary_to_string, extract_message

LISTEN_DURATION = 18


def record_audio():
    print("Recording...")
    recording = sd.rec(int(LISTEN_DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)
    sd.wait()
    signal = recording.flatten()
    print("Recording Complete!")
    return signal


def process_audio_to_bits_str(signal):
    block_size = int(DURATION * SAMPLE_RATE)
    sub_count = 7
    sub_size = block_size // sub_count

    result = ""
    skip = True

    for i in range(0, len(signal), block_size):
        block = signal[i: i + block_size]
        if len(block) < block_size:
            block = np.pad(block, (0, block_size - len(block)))

        peak_freqs = []
        for j in range(sub_count):
            sub = block[j * sub_size:(j + 1) * sub_size]

            fft_result = fft(sub)
            magnitude = np.abs(fft_result[:sub_size // 2])
            freqs = np.fft.fftfreq(sub_size, 1 / SAMPLE_RATE)[:sub_size // 2]

            peak_freq = freqs[np.argmax(magnitude)]
            peak_freqs.append(peak_freq)

        median_peak = int(np.median(peak_freqs))
        if median_peak < FREQ_THRESHOLD and skip:
            print("skipped")
            continue

        print("used")
        skip = False
        if median_peak > (ZERO_FREQ + ONE_FREQ) / 2:
            result += ONE
        else:
            result += ZERO

    return result


def plot_spectrogram(signal):
    plt.figure(figsize=(10, 4))
    plt.specgram(signal, Fs=SAMPLE_RATE, NFFT=1024, noverlap=512, cmap='viridis')
    plt.title("Spectrogram of Recorded Audio")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.colorbar(label='Intensity (dB)')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    audio_signal = record_audio()
    result_bits = process_audio_to_bits_str(audio_signal)
    print("Bit sequence:", result_bits)

    message = extract_message(result_bits)
    print("Message in binary:", message)
    print("Decoded text:", binary_to_string(message))

    plot_spectrogram(audio_signal)
