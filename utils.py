import math
import random
import numpy as np
from PIL import Image
from consts import FREQ_MIN, FREQ_MAX


# -------------------------------
# Padding + Bit Utilities
# -------------------------------

def pad_maker() -> list[list[int]]:
    """Generates 10 random frequency sublists for framing pad tones."""
    return [
        [random.randint(FREQ_MIN, FREQ_MAX) for _ in range(random.randint(2, 10))]
        for _ in range(10)
    ]

def pad_to_square(bits):
    length = len(bits)
    side = math.ceil(math.sqrt(length))
    padded_length = side * side
    padding = padded_length - length
    return bits + [0]*padding, side


# -------------------------------
# Image <-> Bits Conversion
# -------------------------------

def image_to_bits(image_path: str) -> list[int]:
    """Converts a black-and-white image to a list of bits (0 = black, 1 = white)."""
    img = Image.open(image_path).convert('1')
    return [0 if p == 0 else 1 for p in list(img.getdata())]

def bits_to_image(bits, size: tuple[int, int], output_path: str):
    """
    Save a black-and-white image from flat bits.
    """
    expected_len = size[0] * size[1]
    if len(bits) != expected_len:
        print(f"Warning: got {len(bits)} bits, expected {expected_len} for shape {size}")
        bits = bits[:expected_len] + [0] * max(0, expected_len - len(bits))

    img = Image.new('1', size)
    img.putdata([0 if b == 0 else 255 for b in bits])
    img.save(output_path)



# -------------------------------
# JPEG Binary <-> Bits
# -------------------------------

def jpg_to_bits(jpg_binary: bytes) -> list[int]:
    """Converts raw JPEG binary data into a list of bits."""
    return [int(b) for byte in jpg_binary for b in f"{byte:08b}"]

def bits_to_jpg(bits: list[int]) -> bytes:
    """Converts a list of bits back into JPEG binary data."""
    if len(bits) % 8 != 0:
        bits += [0] * (8 - len(bits) % 8)
    return bytes(int("".join(map(str, bits[i:i+8])), 2) for i in range(0, len(bits), 8))


# -------------------------------
# String <-> Bits
# -------------------------------

def string_to_binary(text: str) -> list[int]:
    """Converts a string to a list of binary digits (8 bits per char)."""
    return [int(b) for char in text for b in format(ord(char), '08b')]

def binary_to_string(bits: list[int]) -> str:
    """Converts a list of bits into a string (ignores excess bits)."""
    n = len(bits) - (len(bits) % 8)
    return ''.join(chr(int(''.join(str(b) for b in bits[i:i+8]), 2)) for i in range(0, n, 8))


# -------------------------------
# Visualization & Validation
# -------------------------------

def bits_to_binary_image(bits: list[int], width: int | None = None) -> np.ndarray:
    """
    Converts a bit list to a 2D binary image (as numpy array).
    Pads to fit dimensions if needed.
    """
    n = len(bits)
    if width is None:
        width = int(np.ceil(np.sqrt(n)))
    height = int(np.ceil(n / width))
    padded_bits = bits + [0] * (width * height - n)
    return np.array(padded_bits, dtype=np.uint8).reshape((height, width))

def error_count(msg1: list[int], msg2: list[int]) -> int:
    """Returns number of differing bits; -1 if lengths mismatch."""
    if len(msg1) != len(msg2):
        return -1
    return sum(b1 != b2 for b1, b2 in zip(msg1, msg2))
