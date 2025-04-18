import numpy as np

def lsfr(seed, taps, length):
    """
    Generate a binary LSFR (Linear Feedback Shift Register) sequence.

    Parameters:
    - seed (list[int]): Initial non-zero state of the LSFR.
    - taps (list[int]): Feedback tap positions (0-indexed).
    - length (int): Desired length of output sequence.

    Returns:
    - list[int]: Generated LSFR sequence.
    """
    if not any(seed):
        raise ValueError("Seed cannot be all zeros.")

    state = np.array(seed, dtype=int)
    sequence = []

    for _ in range(length):
        feedback = np.bitwise_xor.reduce(state[taps])
        sequence.append(state[-1])
        state = np.roll(state, 1)
        state[0] = feedback

    return sequence
