import random

from consts import *


def pad_maker():
    """Generates a list of 10 frequency sublist (for transmitting pads)."""

    random_lists = []
    for _ in range(10):
        length = random.randint(2, 10)
        sublist = [random.randint(FREQ_MIN, FREQ_MAX) for _ in range(length)]
        random_lists.append(sublist)
    return random_lists


def string_to_binary(text: str):
    return ''.join(format(ord(char), '08b') for char in text)


def binary_to_string(binary_text: str):
    if len(binary_text) % 8 != 0:
        binary_text = binary_text[:len(binary_text) - (len(binary_text) % 8)]
    return ''.join(chr(int(binary_text[i:i + 8], 2)) for i in range(0, len(binary_text), 8))


def error_count(msg1: str, msg2: str) -> int:
    """
    Count the number of differing bits between two binary strings.
    Returns -1 if the lengths are different.
    """
    if len(msg1) != len(msg2):
        return -1  # or raise ValueError("Messages must be the same length")

    return sum(b1 != b2 for b1, b2 in zip(msg1, msg2))
