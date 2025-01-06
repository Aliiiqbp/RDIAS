import bchlib
import numpy as np
import random

# Parameters for BCH code
BCH_POLYNOMIAL = 285  # Predefined polynomial, usually sufficient for n=511
BCH_T = 25  # Error-correcting capability "t"
K = 256  # Size of the original message in bits
P = 0.04  # Probability of bit flip in the BSC channel

# Initialize BCH encoder/decoder with (n=511, k=256, t=25)
bch = bchlib.BCH(BCH_T, BCH_POLYNOMIAL)


def generate_random_message(k):
    """Generate a random binary message of size k."""
    return bytearray(np.random.randint(0, 2, k // 8))


def bsc_channel(encoded_message, p):
    """Simulate a Binary Symmetric Channel (BSC) with flip probability p."""
    noisy_message = bytearray(encoded_message)  # Copy the encoded message
    for i in range(len(noisy_message) * 8):
        if random.random() < p:
            byte_index = i // 8
            bit_index = i % 8
            # Flip the bit at the specified position
            noisy_message[byte_index] ^= (1 << bit_index)
    return noisy_message


def main():
    # Step 1: Generate a random binary message
    original_message = generate_random_message(K)
    print("Original message:", original_message.hex())

    # Step 2: Encode the message using BCH
    ecc = bch.encode(original_message)
    encoded_message = original_message + ecc
    print(f"Encoded message (length {len(encoded_message)}): {encoded_message.hex()}")

    # Step 3: Transmit the encoded message through a noisy channel
    noisy_message = bsc_channel(encoded_message, P)
    print(f"Noisy message (length {len(noisy_message)}): {noisy_message.hex()}")

    # Step 4: Decode and correct errors using BCH
    received_message = noisy_message[:len(original_message)]
    received_ecc = noisy_message[len(original_message):]

    # Calculate ECC from received message
    calc_ecc = bch.encode(received_message)

    # Attempt to decode the message
    bitflips = bch.decode(data=received_message, recv_ecc=received_ecc, calc_ecc=calc_ecc)

    if bitflips >= 0:
        print("Decoding successful!")
        print(f"Corrected {bitflips} bit(s).")
    else:
        print("Decoding failed.")

    # Verify if the original message was recovered
    if original_message == received_message:
        print("Original message successfully recovered.")
    else:
        print("Failed to recover the original message.")


if __name__ == "__main__":
    main()
