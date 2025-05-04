from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

from audio_processing import (
    get_pad_wave,
    record_audio,
    find_time_delays,
    process_audio_to_bits,
    plot_spectrogram
)
from data_management import remove_preamble, fix_errors
from utils import (
    bits_to_binary_image,
    bits_to_image,
    binary_to_string
)


def receive_audio_segment() -> np.ndarray:
    """
    Records audio, finds pad markers, and extracts the raw message segment.
    """
    print("Recording...")
    pad_wave = get_pad_wave()
    rec = record_audio()

    plot_spectrogram(rec, "Full Recording")

    (start_idx, _), (end_idx, _) = find_time_delays(pad_wave, rec)
    pad_len = len(pad_wave)
    msg_sig = rec[start_idx + pad_len : end_idx]

    plot_spectrogram(msg_sig, "Extracted Message Only")
    return msg_sig


def decode_bits(msg_sig: np.ndarray) -> list[int]:
    """
    Demodulate, remove preamble, and correct errors to produce the final bitstream.
    """
    raw_bits = process_audio_to_bits(msg_sig)
    print(f"Raw bits: {len(raw_bits)}")

    no_preamble = remove_preamble(raw_bits)
    fixed = fix_errors(no_preamble)
    print(f"Fixed bits: {len(fixed)}")
    return fixed


def receive_image(msg_sig: np.ndarray):
    """
    Convert bits to a black-and-white image, save, and display it.
    """
    bits = decode_bits(msg_sig)

    img_array = bits_to_binary_image(bits)
    height, width = img_array.shape

    bits_to_image(
        img_array.flatten().tolist(),
        (width, height),
        "./images/received.jpg"
    )
    print("Image saved to ./images/received.jpg")

    plt.imshow(img_array, cmap='gray')
    plt.title("Reconstructed Image")
    plt.axis('off')
    plt.show()


def receive_text(msg_sig: np.ndarray):
    """
    Convert bits to an ASCII string and print it.
    """
    bits = decode_bits(msg_sig)
    message = binary_to_string(bits)
    print("Received text message:")
    print(message)


def main():
    msg_sig = receive_audio_segment()

    # === Choose one ===
    receive_image(msg_sig)
    # receive_text(msg_sig)
    # ===================


if __name__ == "__main__":
    main()
