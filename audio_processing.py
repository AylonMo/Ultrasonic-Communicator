import numpy as np
import sounddevice as sd
from scipy.fft import fft
from scipy.signal import correlate, find_peaks
import matplotlib.pyplot as plt
from datetime import datetime

from consts import *


def generate_tone_from_frequencies(freqs):
    """
    Create a single waveform segment by summing sine waves at each frequency in `freqs`,
    lasting BIT_DURATION seconds at SAMPLE_RATE Hz.
    """

    # time axis for this bit segment
    time = np.linspace(0, BIT_DURATION, int(SAMPLE_RATE * BIT_DURATION), endpoint=False)
    segment = np.zeros_like(time, dtype=np.float32)

    # add each frequency component
    for freq in freqs:
        segment += 0.5 * np.sin(2 * np.pi * freq * time)

    # normalize amplitude to avoid clipping
    segment /= len(freqs)
    return segment


def get_pad_wave():
    """
    Build the preamble/postamble 'pad' by concatenating tone segments.
    Each entry in PAD is a list of frequencies for one pad segment.
    """

    pad_wave = np.array([], dtype=np.float32)
    for segment_freqs in PAD:
        segment_wave = generate_tone_from_frequencies(segment_freqs)
        pad_wave = np.concatenate((pad_wave, segment_wave))
    return pad_wave


def record_audio():
    """
    Record a mono audio signal for LISTEN_DURATION seconds
    at SAMPLE_RATE Hz, and return it as a 1D NumPy array.
    """

    print("Recording...", datetime.now())
    rec = sd.rec(int(LISTEN_DURATION * SAMPLE_RATE),
                 samplerate=SAMPLE_RATE,
                 channels=1)
    sd.wait()  # block until the recording is finished
    print("Recording complete.", datetime.now())
    return rec.flatten()  # convert shape (N,1) → (N,)


def find_time_delays(pad_signal, received_signal):
    """
    Cross‑correlate `received_signal` against `pad_signal` to locate
    two occurrences of the pad. Returns a sorted list of two tuples:
      [(lag_samples1, lag_secs1), (lag_samples2, lag_secs2)].
    """

    # Normalize both signals to zero mean, unit variance
    ps = (pad_signal - pad_signal.mean()) / pad_signal.std()
    rs = (received_signal - received_signal.mean()) / received_signal.std()

    # Full cross‑correlation and corresponding lag values
    corr = correlate(rs, ps, mode='full')
    lags = np.arange(-len(ps) + 1, len(rs))

    # Find local maxima at least ~pad_length apart
    min_dist = int(len(ps) * 0.8)
    peaks, _ = find_peaks(corr, distance=min_dist)
    if len(peaks) < 2:
        raise RuntimeError("Could not find two pad occurrences.")

    # Select the two highest peaks by correlation magnitude
    top2 = peaks[np.argsort(corr[peaks])[-2:]]
    delays = [(lags[p], lags[p] / SAMPLE_RATE) for p in top2]

    # Sort by sample index so we know which comes first
    return sorted(delays, key=lambda x: x[0])


def process_audio_to_bits(signal):
    """
    Split the incoming `signal` into consecutive BIT_DURATION‑long blocks,
    subdivide each block into 7 sub‑blocks, run an FFT on each sub‑block
    (ignoring frequencies below FREQ_MIN), take the median of those peak
    frequencies, and convert to '0' or '1' by comparing to the midpoint
    of ZERO_FREQ and ONE_FREQ.
    """

    block_size = int(BIT_DURATION * SAMPLE_RATE)
    sub_count = 7
    sub_size = block_size // sub_count

    bits = []
    skipping = True

    for start in range(0, len(signal), block_size):
        block = signal[start : start + block_size]
        if len(block) < block_size:
            block = np.pad(block, (0, block_size - len(block)))

        peak_freqs = []
        for j in range(sub_count):
            sub = block[j * sub_size : (j + 1) * sub_size]
            X = fft(sub)
            mags = np.abs(X[: sub_size // 2])
            freqs = np.fft.fftfreq(sub_size, 1 / SAMPLE_RATE)[: sub_size // 2]

            # Mask out all frequencies below FREQ_MIN
            mask = freqs >= FREQ_MIN
            if not np.any(mask):
                peak_freqs.append(0.0)
            else:
                freqs_above = freqs[mask]
                mags_above = mags[mask]
                peak_freqs.append(freqs_above[np.argmax(mags_above)])

        median_peak = np.median(peak_freqs)

        # Skip over initial silence or pad‑noise until we see a valid bit
        if median_peak < FREQ_MIN and skipping:
            continue
        skipping = False

        # Bit = '1' if above midpoint, else '0'
        midpoint = (ZERO_FREQ + ONE_FREQ) / 2
        bits.append(ONE if median_peak > midpoint else ZERO)

    return "".join(bits)


def plot_spectrogram(sig, title="Spectrogram"):
    """
    Display a time‑frequency spectrogram of `sig` at SAMPLE_RATE Hz.
    """

    plt.figure(figsize=(10, 4))
    plt.specgram(sig,
                 Fs=SAMPLE_RATE,
                 NFFT=1024,
                 noverlap=512,
                 cmap='viridis')
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.colorbar(label='Intensity (dB)')
    plt.tight_layout()
    plt.show()
