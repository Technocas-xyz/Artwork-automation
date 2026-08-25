import sys
from playwright.sync_api import sync_playwright

account = sys.argv[1] if len(sys.argv) > 1 else "acct1"
profile = f"./profiles/{account}"

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
    print("Browser khul gaya. Agar login maange to kar lo.")
    input("Ho jaye to yahan Enter dabao...")

    ctx.close()
    print("Session save ho gaya.")
