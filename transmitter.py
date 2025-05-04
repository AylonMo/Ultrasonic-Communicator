from datetime import datetime

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


def send_image(path: str):
    """Load, prepare, and transmit an image file over audio."""
    raw = load_image_bits(path)
    tx = prepare_bits(raw)
    print("Starting image transmission at", datetime.now())
    play_bit_tone(tx)
    print("Finished image transmission at", datetime.now())


def send_text(text: str):
    """Load, prepare, and transmit a text message over audio."""
    raw = load_text_bits(text)
    tx = prepare_bits(raw)
    print("Starting text transmission at", datetime.now())
    play_bit_tone(tx)
    print("Finished text transmission at", datetime.now())


def main():
    # === Choose one of the following ===

    # 1) Transmit an image:
    send_image("./images/eye3.jpg")

    # 2) Transmit text:
    # send_text("This is just another message, nothing special. But tell me, can you read it???")

    # ================================


if __name__ == "__main__":
    main()
