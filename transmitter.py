from datetime import datetime
from pathlib import Path

from audio_processing import play_bit_tone
from data_management import add_error_fixer, add_preamble
from utils import image_to_bits, string_to_binary


def load_image_bits(path: str) -> list[int]:
    """
    Load a black-and-white image and return its pixels as a flat bit list.
    """
    bits = image_to_bits(path)
    print(f"Image '{path}' → {len(bits)} bits")
    return bits


def load_text_bits(text: str) -> list[int]:
    """
    Convert a text string into an 8-bit ASCII bit list.
    """
    bits = string_to_binary(text)
    print(f"Text ({len(text)} chars) → {len(bits)} bits")
    return bits


def prepare_bits(bits: list[int]) -> list[int]:
    """
    Apply convolutional ECC (rate-1/2) and prepend the framing preamble.
    """
    ecc_bits = add_error_fixer(bits)
    framed = add_preamble(ecc_bits)
    print(f"Prepared (ECC + preamble) → {len(framed)} bits")
    return framed


def send_bits(loader, src, label="data"):
    """
    Generalized sending function.
    loader: function to load bits (e.g., load_image_bits or load_text_bits)
    src: argument to loader (image path or text)
    label: label for reporting
    """
    start = datetime.now()
    raw = loader(src)
    tx = prepare_bits(raw)
    play_bit_tone(tx)
    duration = (datetime.now() - start).total_seconds()
    print(f"[{label}] Transmission complete in {duration:.2f}s")


if __name__ == "__main__":
    # === Choose transmission type ===

    # To send an image:
    img_path = Path("./images/bunny.jpg")
    send_bits(load_image_bits, str(img_path), label="image")

    # To send text:
    # message = "This is just another message, nothing special. But tell me, can you read it???"
    # send_bits(load_text_bits, message, label="text")

    # ================================
