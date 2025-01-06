from Crypto.PublicKey import RSA
from Crypto.Util.number import getPrime, inverse
import random
import time

# Step 1: Generate a random 256-bit binary message
def generate_random_256bit_message():
    return random.getrandbits(256)

# Step 2: Generate RSA public/private key pair with a 256-bit modulus
def generate_rsa_keys():
    key_size = 256
    e = 65537  # Commonly used public exponent

    # Generate two distinct 128-bit prime numbers p and q
    p = getPrime(128)
    q = getPrime(128)
    while p == q:
        q = getPrime(128)

    n = p * q
    phi_n = (p - 1) * (q - 1)
    d = inverse(e, phi_n)

    # Create RSA key objects
    private_key = RSA.construct((n, e, d, p, q))
    public_key = RSA.construct((n, e))

    return private_key, public_key

# Step 3: Encrypt the message using the private key
def encrypt_with_private_key(private_key, message):
    # Ensure message is less than modulus n
    message = message % private_key.n
    # Encrypt: c = m^d mod n
    cipher_int = pow(message, private_key.d, private_key.n)
    return cipher_int

# Step 4: Decrypt the message using the public key
def decrypt_with_public_key(public_key, cipher_int):
    # Decrypt: m = c^e mod n
    message_int = pow(cipher_int, public_key.e, public_key.n)
    return message_int

# Main function to tie everything together
def main():
    # Generate random 256-bit message
    message = generate_random_256bit_message()
    print(f"Original Message (integer): {message}")

    # Generate RSA keys
    private_key, public_key = generate_rsa_keys()
    print(f"RSA Modulus n (hex): {hex(private_key.n)}")
    print(f"Private Exponent d (hex): {hex(private_key.d)}")
    print(f"Public Exponent e (hex): {hex(public_key.e)}")

    # Encrypt the message using the private key
    tic = time.time()
    cipher_int = encrypt_with_private_key(private_key, message)
    toc = time.time()
    print("Encryption Time", 1000 * (toc - tic))
    print(f"Encrypted Message (integer): {cipher_int}")

    # Decrypt the message using the public key
    tic = time.time()
    decrypted_message = decrypt_with_public_key(public_key, cipher_int)
    toc = time.time()
    print("Decryption Time", toc - tic)
    print(f"Decrypted Message (integer): {decrypted_message}")

    # Compare the original and decrypted messages
    if message % private_key.n == decrypted_message:
        print("Success: The original and decrypted messages are the same.")
    else:
        print("Failure: The original and decrypted messages are different.")

if __name__ == "__main__":
    main()
