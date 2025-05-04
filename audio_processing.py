import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
from scipy.signal import butter, lfilter
from scipy.fft import fft, fftfreq
from datetime import datetime

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


# -------------------------------
# Audio Playback and Recording
# -------------------------------

def get_pad_wave() -> np.ndarray:
    """
    Generate the pad waveform for synchronization using PAD frequencies.
    Each segment fades in/out to reduce clicks.
    """
    segments = []
    fade_samples = int(0.002 * SAMPLE_RATE)
    fade_in = np.linspace(0, 1, fade_samples)
    fade_out = np.linspace(1, 0, fade_samples)

    for freqs in PAD:
        t = np.linspace(0, BIT_DURATION, int(SAMPLE_RATE * BIT_DURATION), endpoint=False)
        segment = np.zeros_like(t)
        for f in freqs:
            segment += (0.5 / len(freqs)) * np.sin(2 * np.pi * f * t)
        segment[:fade_samples] *= fade_in
        segment[-fade_samples:] *= fade_out
        segments.append(segment)

    return np.concatenate(segments).astype(np.float32)


def record_audio() -> np.ndarray:
    """Record mono audio for LISTEN_DURATION seconds."""
    rec = sd.rec(int(LISTEN_DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)
    sd.wait()
    return rec.flatten()


def play_bit_tone(bits: list[int]) -> None:
    """
    Encode and play a bit sequence using multitone modulation with pad markers.
    """
    print(f"Message before padding: {bits}")
    pad_wave = get_pad_wave()
    chans = split_bits(bits, N_CHANNELS)
    data_wave = get_multitone(chans)
    full_wave = np.concatenate((pad_wave, data_wave, pad_wave))

    print("Started transmitting at", datetime.now())
    sd.play(full_wave, samplerate=SAMPLE_RATE)
    sd.wait()
    print("Finished transmitting at", datetime.now())


# -------------------------------
# Signal Processing Utilities
# -------------------------------

def bandpass_filter(data: np.ndarray, lowcut: float, highcut: float, fs: int, order: int = 5) -> np.ndarray:
    """Apply a Butterworth band-pass filter to the signal."""
    ny = 0.5 * fs
    b, a = butter(order, [lowcut / ny, highcut / ny], btype='band')
    return lfilter(b, a, data)


def quadratic_peak(mags: np.ndarray, idx: int) -> float:
    """
    Parabolic interpolation around a spectral peak.
    Returns a fractional index near the original peak.
    """
    if idx <= 0 or idx >= len(mags) - 1:
        return idx
    alpha, beta, gamma = mags[idx - 1], mags[idx], mags[idx + 1]
    return idx + 0.5 * (alpha - gamma) / (alpha - 2 * beta + gamma)


# -------------------------------
# Spectrogram and Sync Detection
# -------------------------------

def plot_spectrogram(sig: np.ndarray, title: str = "Spectrogram") -> None:
    """Plot the spectrogram of the input signal."""
    plt.figure(figsize=(10, 4))
    plt.specgram(sig, Fs=SAMPLE_RATE, NFFT=1024, noverlap=512)
    plt.ylim(FREQ_MIN - 500, 24000)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.colorbar(label='Intensity (dB)')
    plt.tight_layout()
    plt.show()


def find_time_delays(pad_wave: np.ndarray, rec: np.ndarray, min_distance: int | None = None) -> tuple[tuple[int, float], tuple[int, float]]:
    """
    Locate pad positions in the recording via cross-correlation.
    Returns two peak indices and their correlation strengths.
    """
    corr = np.correlate(rec, pad_wave, mode='valid')
    if min_distance is None:
        min_distance = len(pad_wave)

    p1 = int(np.argmax(corr))

    corr_masked = corr.copy()
    corr_masked[max(0, p1 - min_distance):min(len(corr), p1 + min_distance + 1)] = -np.inf
    p2 = int(np.argmax(corr_masked))

    p_early, p_late = sorted([p1, p2])
    v_early, v_late = corr[p_early], corr[p_late]

    # Plot annotated correlation
    plt.figure(figsize=(10, 4))
    plt.plot(corr, label='Cross-correlation')
    plt.scatter([p_early, p_late], [v_early, v_late], color='red', label='Pad Peaks')
    plt.axvline(p_early, color='red', linestyle='--', alpha=0.5)
    plt.axvline(p_late, color='red', linestyle='--', alpha=0.5)
    plt.title("Cross-correlation with pad signal")
    plt.xlabel("Lag (samples)")
    plt.ylabel("Correlation")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return (p_early, float(v_early)), (p_late, float(v_late))


# -------------------------------
# Bitstream Decoding
# -------------------------------

def process_audio_to_bits(msg_sig: np.ndarray) -> list[int]:
    """
    Convert a multichannel audio signal into a list of bits.
    Uses FFT and magnitude comparison to detect each channel's bit.
    """
    frame_len = int(BIT_DURATION * SAMPLE_RATE)
    num_frames = len(msg_sig) // frame_len

    channel_bits = [[] for _ in range(N_CHANNELS)]

    for i in range(num_frames):
        frame = msg_sig[i * frame_len : (i + 1) * frame_len]
        windowed = frame * np.hanning(len(frame))
        padded = np.pad(windowed, (0, len(frame)), mode='constant')

        X = fft(padded)
        mags = np.abs(X)[:len(X)//2]
        freqs = fftfreq(len(padded), 1 / SAMPLE_RATE)[:len(X)//2]

        for ch in range(N_CHANNELS):
            f0, f1 = FREQ_PAIRS[ch]
            idx0 = np.argmin(np.abs(freqs - f0))
            idx1 = np.argmin(np.abs(freqs - f1))

            mag0 = mags[int(round(np.clip(quadratic_peak(mags, idx0), 0, len(mags) - 1)))]
            mag1 = mags[int(round(np.clip(quadratic_peak(mags, idx1), 0, len(mags) - 1)))]

            bit = 1 if mag1 > mag0 / 2 else 0
            channel_bits[ch].append(bit)

    # Interleave bits in round-robin order
    bits = []
    max_len = max(len(ch) for ch in channel_bits)
    for i in range(max_len):
        for ch in range(N_CHANNELS):
            if i < len(channel_bits[ch]):
                bits.append(channel_bits[ch][i])

    return bits


# -------------------------------
# Transmission Helper Functions
# -------------------------------

def split_bits(bits: list[int], n: int) -> list[list[int]]:
    """
    Round-robin split of a bit list into `n` parallel channels.
    """
    chans = [[] for _ in range(n)]
    for idx, b in enumerate(bits):
        chans[idx % n].append(b)
    return chans


def get_multitone(chans: list[list[int]]) -> np.ndarray:
    """
    Generate a multitone waveform where each BIT_DURATION segment encodes one bit per channel.
    Adds fade-in/out to reduce artifacts.
    """
    t = np.linspace(0, BIT_DURATION, int(SAMPLE_RATE * BIT_DURATION), endpoint=False)
    fade_samples = int(0.002 * SAMPLE_RATE)
    fade_in = np.linspace(0, 1, fade_samples)
    fade_out = np.linspace(1, 0, fade_samples)
    full_wave = np.array([], dtype=np.float32)

    L = max(len(c) for c in chans)

    for symbol_idx in range(L):
        segment = np.zeros_like(t)
        for i, chan in enumerate(chans):
            bit = chan[symbol_idx] if symbol_idx < len(chan) else 0
            f0, f1 = FREQ_PAIRS[i]
            freq = f1 if bit == 1 else f0
            segment += 0.5 * np.sin(2 * np.pi * freq * t)

        segment /= N_CHANNELS
        segment[:fade_samples] *= fade_in
        segment[-fade_samples:] *= fade_out
        full_wave = np.concatenate((full_wave, segment.astype(np.float32)))

    return full_wave
