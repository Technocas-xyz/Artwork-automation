"""Create or refresh a saved ChatGPT browser session.

Usage:  python login.py acct1

Opens a real Chrome window using a persistent profile. Log in by hand,
then press Enter here. The session is reused by the automation.

NOTE: If the agent (agent.py / ArtworkAgent.exe) is running, STOP IT FIRST.
The agent holds the browser profile and login.py cannot share it. The server
does not open a browser at all any more, so it can keep running.
"""

import sys
from playwright.sync_api import sync_playwright

account = sys.argv[1] if len(sys.argv) > 1 else "acct1"
profile = f"./profiles/{account}"

print()
print("=" * 60)
print("  WARNING: Stop the agent before running this.")
print("  The agent holds the profile lock. Use the agent's")
print("  'Sign in to ChatGPT' button instead if it is running.")
print("=" * 60)
print()

with sync_playwright() as pw:
    ctx = pw.chromium.launch_persistent_context(
        user_data_dir=profile,
        headless=False,
        args=["--disable-blink-features=AutomationControlled"],
        viewport={"width": 1440, "height": 900},
    )
    page = ctx.pages[0] if ctx.pages else ctx.new_page()
    page.goto("https://chatgpt.com", wait_until="domcontentloaded")

    print(f"\nProfile: {profile}")
    print("Browser open. Log in manually, then press Enter here.")
    input("\nDone? Press Enter to save session...")

    ctx.close()
    print("Session saved.")
