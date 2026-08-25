"""Generate a password hash for the .env file.

Usage:  python set_password.py
        Enter password when prompted. Prints the hash to paste into .env.
"""

import hashlib
import os
import getpass


def hash_password(password: str) -> str:
    """Hash a password with PBKDF2-SHA256 and a random salt."""
    salt = os.urandom(16)
    key = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 260000)
    return salt.hex() + ":" + key.hex()


def verify_password(password: str, stored_hash: str) -> bool:
    """Verify a password against a stored PBKDF2 hash."""
    salt_hex, key_hex = stored_hash.split(":", 1)
    salt = bytes.fromhex(salt_hex)
    key = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 260000)
    return key.hex() == key_hex


if __name__ == "__main__":
    print("Set a password for the Artwork Generator app.\n")
    pw = getpass.getpass("Enter password: ")
    pw2 = getpass.getpass("Confirm password: ")
    if pw != pw2:
        print("Passwords do not match.")
    else:
        h = hash_password(pw)
        print(f"\nAdd this to your .env file:\n")
        print(f"APP_PASSWORD_HASH={h}")
        print(f"\n(Keep APP_USERNAME set to your desired username)")
