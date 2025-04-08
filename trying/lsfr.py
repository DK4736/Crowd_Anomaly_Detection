import numpy as np

def lsfr(seed, taps, length):
    """
    Generate a binary LFSR sequence based on a given seed and tap positions.

    Parameters:
    - seed (list[int]): Initial state of the LFSR (should be non-zero).
    - taps (list[int]): Indices for feedback taps (0-based).
    - length (int): Number of bits to generate.

    Returns:
    - list[int]: Generated LFSR sequence of given length.
    """
    if not any(seed):
        raise ValueError("Seed cannot be all zeros!")

    state = np.array(seed, dtype=int)  # Ensure integer type
    result = []

    for _ in range(length):
        next_bit = np.bitwise_xor.reduce(state[taps])  # Compute feedback bit
        result.append(state[-1])  # Store last bit before shifting

        # Shift state and insert new bit at the beginning
        state = np.roll(state, 1)
        state[0] = next_bit

    return result
