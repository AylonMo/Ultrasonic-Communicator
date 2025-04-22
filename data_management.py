import numpy as np

def add_error_fixer(bitstring):
    """
    Rate-1/2 convolutional encoder with constraint length 3,
    Generator polynomials: G0=111 (7), G1=101 (5).

    bitstring: str or list of '0'/'1' characters
    returns:  str of '0'/'1' characters (encoded)
    """
    G = [0b111, 0b101]
    K = 3
    state = 0  # holds last K-1 bits
    encoded_chars = []

    for ch in bitstring:
        bit = int(ch)                # convert '0'/'1' → 0/1
        reg = (state << 1) | bit
        out0 = bin(reg & G[0]).count('1') % 2
        out1 = bin(reg & G[1]).count('1') % 2
        # append as chars so result is a string
        encoded_chars.append(str(out0))
        encoded_chars.append(str(out1))
        # update state to keep only last K-1 bits
        state = reg & ((1 << (K - 1)) - 1)

    return ''.join(encoded_chars)




def fix_errors(received):
    """
    Viterbi decoder for the above convolutional code (hard-decision).

    received: str or list of '0'/'1' characters (the encoded bitstream)
    returns: str of decoded '0'/'1' characters
    """
    # Convert input bits to integers
    rec_bits = [int(ch) for ch in received]

    # Convolutional code parameters
    G = [0b111, 0b101]
    K = 3
    num_states = 1 << (K - 1)

    # Precompute state transitions and expected output bits
    next_state = {}
    output = {}
    for s in range(num_states):
        for bit in [0, 1]:
            reg = (s << 1) | bit
            ns = reg & (num_states - 1)
            out0 = bin(reg & G[0]).count('1') % 2
            out1 = bin(reg & G[1]).count('1') % 2
            next_state[(s, bit)] = ns
            output[(s, bit)] = [out0, out1]

    # Initialize path metrics (costs) and survivor paths
    path_metric = {s: float('inf') for s in range(num_states)}
    path_metric[0] = 0
    survivors = {s: [] for s in range(num_states)}

    # Process received bits in pairs
    for i in range(0, len(rec_bits), 2):
        pair = rec_bits[i:i + 2]
        if len(pair) < 2:
            break  # incomplete pair, stop decoding

        new_metric = {s: float('inf') for s in range(num_states)}
        new_surv = {s: [] for s in range(num_states)}

        for s in range(num_states):
            if path_metric[s] == float('inf'):
                continue
            for bit in [0, 1]:
                ns = next_state[(s, bit)]
                expected = output[(s, bit)]
                # Hamming distance for hard-decision metric
                branch_cost = abs(pair[0] - expected[0]) + abs(pair[1] - expected[1])
                metric = path_metric[s] + branch_cost
                if metric < new_metric[ns]:
                    new_metric[ns] = metric
                    new_surv[ns] = survivors[s] + [bit]

        path_metric = new_metric
        survivors = new_surv

    # Select the best ending state and reconstruct the bit sequence
    end_state = min(path_metric, key=path_metric.get)
    decoded_bits = survivors[end_state]

    # Return as a string of '0'/'1'
    return ''.join(str(b) for b in decoded_bits)

# def fix_errors(received):
#     """
#     Viterbi decoder for the above convolutional code (hard-decision).
#
#     received: str or list of '0'/'1' characters (the encoded bitstream)
#     returns: str of decoded '0'/'1' characters
#     """
#     # Convert input bits to integers
#     rec_bits = [int(ch) for ch in received]
#
#     # Convolutional code parameters
#     G = [0b111, 0b101]
#     K = 3
#     num_states = 1 << (K - 1)
#
#     # Precompute state transitions and expected output bits
#     next_state = {}
#     output = {}
#     for s in range(num_states):
#         for bit in [0, 1]:
#             reg = (s << 1) | bit
#             ns = reg & (num_states - 1)
#             out0 = bin(reg & G[0]).count('1') % 2
#             out1 = bin(reg & G[1]).count('1') % 2
#             next_state[(s, bit)] = ns
#             output[(s, bit)] = [out0, out1]
#
#     # Initialize path metrics (costs) and survivor paths
#     path_metric = {s: float('inf') for s in range(num_states)}
#     path_metric[0] = 0
#     survivors = {s: [] for s in range(num_states)}
#
#     # Process received bits in pairs
#     for i in range(0, len(rec_bits), 2):
#         pair = rec_bits[i:i + 2]
#         if len(pair) < 2:
#             break  # incomplete pair, stop decoding
#
#         new_metric = {s: float('inf') for s in range(num_states)}
#         new_surv = {s: [] for s in range(num_states)}
#
#         for s in range(num_states):
#             if path_metric[s] == float('inf'):
#                 continue
#             for bit in [0, 1]:
#                 ns = next_state[(s, bit)]
#                 expected = output[(s, bit)]
#                 # Hamming distance for hard-decision metric
#                 branch_cost = abs(pair[0] - expected[0]) + abs(pair[1] - expected[1])
#                 metric = path_metric[s] + branch_cost
#                 if metric < new_metric[ns]:
#                     new_metric[ns] = metric
#
#         path_metric = new_metric
#         survivors = new_surv
#
#     # Select the best ending state and reconstruct the bit sequence
#     end_state = min(path_metric, key=path_metric.get)
#     decoded_bits = survivors[end_state]
#
#     # Return as a string of '0'/'1'
#     return ''.join(str(b) for b in decoded_bits)


