import sys
from datetime import datetime

import numpy as np
import sounddevice as sd
from scipy.fft import fft
from scipy.signal import correlate, find_peaks
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt


from consts import *
from utils import binary_to_string, extract_message, string_to_binary
from transmitter import get_bit_tone

LISTEN_DURATION = 18


def bandpass_filter(signal, lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)


def record_audio():
    print("Recording...")
    print(datetime.now())
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
            # print("skipped")
            continue

        # print("used")
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


# def find_time_delay(signal1, signal2, fs):
#     # normalize signals
#     signal1 = (signal1 - np.mean(signal1)) / np.std(signal1)
#     signal2 = (signal2 - np.mean(signal2)) / np.std(signal2)
#
#     # perform cross-correlation
#     corr = correlate(signal2, signal1, mode='full')
#     lags = np.arange(-len(signal1) + 1, len(signal2))
#
#     # find the lag with the maximum correlation
#     max_corr_index = np.argmax(corr)
#     time_delay_samples = lags[max_corr_index]
#     time_delay = time_delay_samples / fs
#
#     # optional plot
#     plt.figure(figsize=(8, 4))
#     plt.plot(lags / fs, corr)
#     plt.title("Cross-Correlation")
#     plt.xlabel("Lag (seconds)")
#     plt.ylabel("Correlation")
#     plt.grid()
#     plt.show()
#
#     return time_delay, time_delay_samples


def find_time_delay(signal1, signal2, fs):
    # normalize signals
    signal1 = (signal1 - np.mean(signal1)) / np.std(signal1)
    signal2 = (signal2 - np.mean(signal2)) / np.std(signal2)

    # perform cross-correlation
    corr = correlate(signal2, signal1, mode='full')
    lags = np.arange(-len(signal1) + 1, len(signal2))

    # find local maxima in the correlation
    peaks, _ = find_peaks(corr, distance=3*fs)  # adjust distance as needed

    # select the top 2 local peaks by height
    top_two_indices = peaks[np.argsort(corr[peaks])[-2:]]
    top_two_indices = top_two_indices[np.argsort(corr[top_two_indices])[::-1]]  # descending order

    # extract time delays
    time_delays_samples = lags[top_two_indices]
    time_delays = time_delays_samples / fs

    # optional plot
    plt.figure(figsize=(8, 4))
    plt.plot(lags / fs, corr)
    plt.title("Cross-Correlation")
    plt.xlabel("Lag (seconds)")
    plt.ylabel("Correlation")
    plt.grid()
    for td in time_delays:
        plt.axvline(td, color='r', linestyle='--')
    plt.show()

    return list(zip(time_delays, time_delays_samples))



if __name__ == "__main__":
    print(f"padding: {PAD}")
    og_pad_signal = get_bit_tone(PAD)

     # optional filtration
    # # Band-pass filter parameters
    # LOW_CUT = 21000  # Hz
    # HIGH_CUT = 24000  # Hz
    # # Apply bandpass filter to clean the signal
    # audio_signal = bandpass_filter(audio_signal, LOW_CUT, HIGH_CUT, SAMPLE_RATE)
    # optional filtration

    audio_signal = record_audio()
    plot_spectrogram(audio_signal)

    time_delay, time_delay_samples = find_time_delay(og_pad_signal, audio_signal, SAMPLE_RATE)
    print(time_delay, time_delay_samples)

    start, end = time_delay[1], time_delay_samples[1]
    audio_signal = audio_signal[start: end]

    result_bits = process_audio_to_bits_str(audio_signal)
    print("Bit sequence:", result_bits)

    message = extract_message(result_bits)
    print("Message in binary:", message)
    print("Decoded text:", binary_to_string(message))

    plot_spectrogram(audio_signal)
