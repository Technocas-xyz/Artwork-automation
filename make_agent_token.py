"""Generate a new agent token for a designer.

Usage:
    python make_agent_token.py "Designer Name"

Prints a token to paste into the designer's agent .env as AGENT_TOKEN.
"""
import sys

from src.agent_tokens import create_token


def main() -> None:
    name = " ".join(sys.argv[1:]).strip() or "designer"
    token = create_token(name)
    print()
    print(f"  Agent token for {name!r}:")
    print()
    print(f"    {token}")
    print()
    print("  Paste this into the designer's agent .env as:")
    print(f"    AGENT_TOKEN={token}")
    print()


if __name__ == "__main__":
    main()
