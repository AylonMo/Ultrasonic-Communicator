import numpy as np
from collections import Counter
from consts import N_CHANNELS


# -------------------------------
# Preamble Handling
# -------------------------------

def add_preamble(message: list[int]) -> list[int]:
    """Prepends a preamble of N_CHANNELS 1s after optional padding to align channels."""
    padding = [0] * (N_CHANNELS - len(message) % N_CHANNELS) if len(message) % N_CHANNELS != 0 else []
    return padding + [1] * N_CHANNELS + message

def remove_preamble(message: list[int]) -> list[int]:
    """Removes the first N_CHANNELS bits following the first 1 (used for sync)."""
    try:
        start = message.index(1)
        return message[start + N_CHANNELS:]
    except ValueError:
        return message  # If no '1' found, return unchanged


# -------------------------------
# Convolutional Error Correction
# -------------------------------

def add_error_fixer(bitlist: list[int]) -> list[int]:
    """Applies rate-1/2 convolutional encoding with G = [111, 101]."""
    G = [0b111, 0b101]
    K = 3
    state = 0
    encoded = []
    for bit in bitlist:
        reg = (state << 1) | bit
        out0 = bin(reg & G[0]).count('1') % 2
        out1 = bin(reg & G[1]).count('1') % 2
        encoded += [out0, out1]
        state = reg & ((1 << (K - 1)) - 1)
    return encoded

def fix_errors(received: list[int]) -> list[int]:
    """Decodes bits using Viterbi decoding for the above convolutional code."""
    G = [0b111, 0b101]
    K = 3
    num_states = 1 << (K - 1)

    next_state = {}
    output = {}
    for s in range(num_states):
        for bit in [0, 1]:
            reg = (s << 1) | bit
            ns = reg & (num_states - 1)
            out = [bin(reg & g).count('1') % 2 for g in G]
            next_state[(s, bit)] = ns
            output[(s, bit)] = out

    path_metric = {s: float('inf') for s in range(num_states)}
    path_metric[0] = 0
    survivors = {s: [] for s in range(num_states)}

    for i in range(0, len(received), 2):
        pair = received[i:i+2]
        if len(pair) < 2:
            break
        new_metric = {s: float('inf') for s in range(num_states)}
        new_surv = {s: [] for s in range(num_states)}
        for s in range(num_states):
            if path_metric[s] == float('inf'):
                continue
            for bit in [0, 1]:
                ns = next_state[(s, bit)]
                expected = output[(s, bit)]
                cost = abs(pair[0] - expected[0]) + abs(pair[1] - expected[1])
                metric = path_metric[s] + cost
                if metric < new_metric[ns]:
                    new_metric[ns] = metric
                    new_surv[ns] = survivors[s] + [bit]
        path_metric = new_metric
        survivors = new_surv

    best_state = min(path_metric, key=path_metric.get)
    return survivors[best_state]


# -------------------------------
# Hybrid RLE Compression
# -------------------------------

def hybrid_compress(bits: list[int]) -> list[int]:
    """
    Hybrid compression:
    - If uniform: 1 + bit + 6-bit run length (max 63)
    - If mixed:   0 + 7 bits raw
    """
    i = 0
    compressed = []
    while i < len(bits):
        chunk = bits[i:i+7]
        if len(set(chunk)) == 1:
            run_bit = chunk[0]
            run_len = 0
            while i + run_len < len(bits) and bits[i + run_len] == run_bit and run_len < 63:
                run_len += 1
            compressed += [1, run_bit] + [int(b) for b in f"{run_len:06b}"]
            i += run_len
        else:
            compressed += [0] + [bits[i + j] if i + j < len(bits) else 0 for j in range(7)]
            i += 7
    return compressed

def hybrid_decompress(compressed: list[int]) -> list[int]:
    """Decompress hybrid RLE format."""
    i = 0
    bits = []
    while i < len(compressed):
        flag = compressed[i]
        if flag == 0:
            bits += compressed[i+1:i+8]
        else:
            run_bit = compressed[i+1]
            run_len = int(''.join(map(str, compressed[i+2:i+8])), 2)
            bits += [run_bit] * run_len
        i += 8
    return bits


# -------------------------------
# Block-Based Compression
# -------------------------------

def block_compress(bits: list[int], image_shape: tuple[int, int], block_size: int = 36) -> list[int]:
    """
    Compresses image in blocks.
    Uniform blocks use 4-bit headers, repeated 3×.
    Mixed blocks store full data.
    """
    height, width = image_shape
    arr = np.array(bits).reshape((height, width))
    compressed = [int(b) for b in format(height, '016b')] + [int(b) for b in format(width, '016b')]

    full_h = (height // block_size) * block_size
    full_w = (width  // block_size) * block_size

    def robust_header(hdr): return hdr * 3

    for row in range(0, full_h, block_size):
        for col in range(0, full_w, block_size):
            block = arr[row:row+block_size, col:col+block_size].flatten()
            if block.std() == 0:
                hdr = [0, 0, 0, block[0]]
                compressed += robust_header(hdr)
            else:
                compressed += robust_header([1, 1, 1, 1]) + block.tolist()

    # Tail blocks (right/bottom edges)
    tail_slices = []
    if full_w < width:  tail_slices.append(arr[:full_h, full_w:])
    if full_h < height: tail_slices.append(arr[full_h:, :])
    if tail_slices:
        tail_bits = np.concatenate([t.flatten() for t in tail_slices]).tolist()
        compressed += robust_header([1, 0, 1, 0]) + tail_bits

    return compressed


def block_decompress(bitstream: list[int], block_size: int = 16) -> tuple[list[int], tuple[int, int]]:
    """
    Decompresses block-based encoded bitstream into flat bit list and shape.
    Uses 3× replicated headers with majority vote for robustness.
    """
    def majority_vote(bits):
        votes = [tuple(bits[i:i+4]) for i in range(0, 12, 4)]
        count = Counter(votes)
        most_common, num = count.most_common(1)[0]
        if num < 2:
            print(f"Header severely corrupted {bits}. Using best guess: {votes[0]}")
            return list(votes[0])
        return list(most_common)

    height = int(''.join(map(str, bitstream[:16])), 2)
    width = int(''.join(map(str, bitstream[16:32])), 2)
    i = 32

    full_h = (height // block_size) * block_size
    full_w = (width  // block_size) * block_size
    blocks_per_row = full_w // block_size
    total_blocks = (full_h // block_size) * blocks_per_row

    arr = np.zeros((height, width), dtype=int)
    row_b = col_b = 0

    for _ in range(total_blocks):
        header_bits = bitstream[i:i+12]
        if len(header_bits) < 12:
            print("Incomplete header. Padding remaining blocks with zeros.")
            break
        i += 12
        header = majority_vote(header_bits)

        if header == [0, 0, 0, 0]: val = 0
        elif header == [0, 0, 0, 1]: val = 1
        elif header == [1, 1, 1, 1]:
            size = block_size**2
            block_data = bitstream[i:i+size]
            if len(block_data) != size:
                print("Incomplete mixed block data. Padding zeros.")
                patch = np.zeros((block_size, block_size), dtype=int)
            else:
                patch = np.array(block_data).reshape((block_size, block_size))
                i += size
            arr[row_b*block_size:(row_b+1)*block_size, col_b*block_size:(col_b+1)*block_size] = patch
            col_b = (col_b + 1) % blocks_per_row
            if col_b == 0: row_b += 1
            continue
        else:
            print(f"Unknown header {header}, padding block with zeros.")
            val = 0

        arr[row_b*block_size:(row_b+1)*block_size, col_b*block_size:(col_b+1)*block_size] = val
        col_b = (col_b + 1) % blocks_per_row
        if col_b == 0: row_b += 1

    # Tail recovery
    if i + 12 <= len(bitstream):
        tail_header = majority_vote(bitstream[i:i+12])
        if tail_header == [1, 0, 1, 0]:
            i += 12
            tail_bits = bitstream[i:]
            tail_i = 0

            if full_w < width:
                rows, cols = full_h, width - full_w
                size = rows * cols
                part = tail_bits[tail_i:tail_i+size]
                if len(part) == size:
                    arr[:rows, full_w:] = np.array(part).reshape((rows, cols))
                tail_i += size

            if full_h < height:
                rows, cols = height - full_h, width
                size = rows * cols
                part = tail_bits[tail_i:tail_i+size]
                if len(part) == size:
                    arr[full_h:, :] = np.array(part).reshape((rows, cols))
        else:
            print("Tail header missing or corrupted. Padding tail with zeros.")

    return arr.flatten().tolist(), (height, width)


# -------------------------------
# Optional Testing
# -------------------------------

def test_with_padding():
    height, width = 81, 81
    arr = np.zeros((height, width), dtype=int)
    arr[:16, :16] = 1
    arr[16:32, :16] = np.random.randint(0, 2, (16, 16))
    arr[:16, 16:32] = 0
    arr[32:, :] = 1
    arr[:, 32:] = 1

    bits = arr.flatten().tolist()
    compressed = block_compress(bits, (height, width))
    restored_bits, (rh, rw) = block_decompress(compressed)
    match = (restored_bits == bits) and (rh == height and rw == width)

    print("Original:", len(bits))
    print("Compressed:", len(compressed))
    print("Decompressed:", len(restored_bits))
    print("Restored dims:", rh, rw)
    print("Restoration correct:", match)
