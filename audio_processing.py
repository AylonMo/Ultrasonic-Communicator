import numpy as np
from scipy.signal import butter, lfilter
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
import sounddevice as sd

from consts import (
    SAMPLE_RATE,
    BIT_DURATION,
    LISTEN_DURATION,
    N_CHANNELS,
    FREQ_PAIRS,
    FREQ_MIN,
    FREQ_MAX,
    PAD,
)


def get_pad_wave() -> np.ndarray:
    """
    Generate the known pad preamble by concatenating each multi-frequency segment.
    Each segment has fade-in and fade-out to reduce clicking.
    """
    segments = []
    fade_samples = int(0.002 * SAMPLE_RATE)  # 2 ms fade
    fade_in = np.linspace(0, 1, fade_samples)
    fade_out = np.linspace(1, 0, fade_samples)

    for freqs in PAD:
        t = np.linspace(0, BIT_DURATION, int(SAMPLE_RATE * BIT_DURATION), endpoint=False)
        segment = np.zeros_like(t)
        for f in freqs:
            segment += (0.5 / len(freqs)) * np.sin(2 * np.pi * f * t)

        # Apply fade-in and fade-out
        segment[:fade_samples] *= fade_in
        segment[-fade_samples:] *= fade_out

        segments.append(segment)

    return np.concatenate(segments).astype(np.float32)


def record_audio() -> np.ndarray:
    """
    Record audio for LISTEN_DURATION seconds at SAMPLE_RATE.
    """
    rec = sd.rec(int(LISTEN_DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)
    sd.wait()
    return rec.flatten()


def bandpass_filter(data: np.ndarray, lowcut: float, highcut: float, fs: int, order: int = 5) -> np.ndarray:
    """
    Apply a Butterworth band-pass filter.
    """
    ny = 0.5 * fs
    b, a = butter(order, [lowcut / ny, highcut / ny], btype='band')
    return lfilter(b, a, data)


def plot_spectrogram(sig: np.ndarray, title: str = "Spectrogram") -> None:
    """
    Display a time‑frequency spectrogram of `sig` at SAMPLE_RATE Hz,
    limited to the band 20000–24000 Hz.
    """
    plt.figure(figsize=(10, 4))
    Pxx, freqs, bins, im = plt.specgram(
        sig,
        Fs=SAMPLE_RATE,
        NFFT=1024,
        noverlap=512,
    )
    plt.ylim(FREQ_MIN-500, 24000)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.colorbar(label='Intensity (dB)')
    plt.tight_layout()
    plt.show()

def find_time_delays(pad_wave: np.ndarray,
                     rec: np.ndarray,
                     min_distance: int | None = None
                    ) -> tuple[tuple[int, float], tuple[int, float]]:
    """
    Cross-correlate pad_wave with rec, plot the correlation and mark
    the two highest peaks (at least `min_distance` apart), and return
    their indices + values in chronological order.
    """
    corr = np.correlate(rec, pad_wave, mode='valid')

    # default min distance to one pad‑length if not specified
    if min_distance is None:
        min_distance = len(pad_wave)

    # 1) find the global maximum
    p1 = int(np.argmax(corr))

    # 2) mask out neighborhood around p1
    corr_masked = corr.copy()
    start = max(0, p1 - min_distance)
    end   = min(len(corr), p1 + min_distance + 1)
    corr_masked[start:end] = -np.inf

    # 3) next highest peak
    p2 = int(np.argmax(corr_masked))

    # sort so the earlier peak comes first
    p_early, p_late = sorted([p1, p2])
    v_early, v_late = corr[p_early], corr[p_late]

    # --- plot the full correlation and annotate peaks ---
    plt.figure(figsize=(10, 4))
    plt.plot(corr, label='cross-correlation')
    plt.scatter([p_early, p_late], [v_early, v_late],
                color='red', zorder=5,
                label=f'peaks at {p_early}, {p_late}')
    plt.axvline(p_early, color='red', linestyle='--', alpha=0.5)
    plt.axvline(p_late, color='red', linestyle='--', alpha=0.5)
    plt.title("Cross‑correlation between recording and pad")
    plt.xlabel("Lag (samples)")
    plt.ylabel("Correlation amplitude")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return (p_early, float(v_early)), (p_late, float(v_late))


def process_audio_to_bits(msg_sig: np.ndarray) -> str:

    frame_len = int(BIT_DURATION * SAMPLE_RATE)
    num_frames = len(msg_sig) // frame_len

    # collect bits per channel
    channel_bits = [[] for _ in range(N_CHANNELS)]

    for i in range(num_frames):
        frame = msg_sig[i * frame_len : (i + 1) * frame_len]
        for ch in range(N_CHANNELS):
            f0, f1 = FREQ_PAIRS[ch]
            X = fft(frame)
            mags = np.abs(X)
            freqs = fftfreq(len(frame), 1 / SAMPLE_RATE)
            idx0 = np.argmin(np.abs(freqs - f0))
            idx1 = np.argmin(np.abs(freqs - f1))
            bit = '1' if mags[idx1] > mags[idx0] / 2 else '0'
            channel_bits[ch].append(bit)

    # reassemble bits in round-robin
    bits = []
    max_len = max(len(cb) for cb in channel_bits)
    for k in range(max_len):
        for ch in range(N_CHANNELS):
            if k < len(channel_bits[ch]):
                bits.append(channel_bits[ch][k])
    return ''.join(bits)
