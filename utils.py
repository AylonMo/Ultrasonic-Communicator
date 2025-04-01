from consts import *


def extract_message(binary_string: str):
    """
    Extracts the message between two identical binary pads
    that appear at the start and end of the string.
    """

    try:
        start = binary_string.index(PAD)
        finish = binary_string.index(PAD, start + len(PAD))
        message = binary_string[start + len(PAD): finish]
        return message

    except ValueError as ve:
        print("ValueError")
        return binary_string


def string_to_binary(text: str):
    return ''.join(format(ord(char), '08b') for char in text)


def binary_to_string(binary_text: str):
    if len(binary_text) % 8 != 0:
        binary_text = binary_text[:len(binary_text) - (len(binary_text) % 8)]
    return ''.join(chr(int(binary_text[i:i + 8], 2)) for i in range(0, len(binary_text), 8))
