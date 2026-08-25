"""Smoke test — full-loop proof that generation works end-to-end."""

import time
import traceback
from pathlib import Path

from src.browser import launch_context, is_logged_in
from src.generator import generate


def main() -> None:
    # --- Config (all hardcoded) ---
    account = "acct1"
    prompt = "Remove the background from this artwork and return it as a transparent PNG."
    input_dir = Path("./input")
    output_dir = Path("./output")
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Collect ALL images from ./input/ ---
    image_paths = sorted(
        str(p) for p in input_dir.iterdir()
        if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")
    )
    print(f"[test_run] Found {len(image_paths)} input image(s): {image_paths}")

    # --- Launch browser ---
    start = time.time()
    context = launch_context(account)
    page = context.pages[0] if context.pages else context.new_page()

    # --- Check login ---
    page.goto("https://chatgpt.com", wait_until="domcontentloaded")
    if not is_logged_in(page):
        print("[test_run] NOT LOGGED IN — session expired or missing.")
        print("           Run login.py first to authenticate, then retry.")
        input("Press Enter to close the browser...")
        context.close()
        return

    print("[test_run] Session OK — logged in.")

    # --- Generate ---
    print("[test_run] Sending prompt + images to ChatGPT...")
    images: list[bytes] = generate(
        page=page,
        image_paths=image_paths,
        prompt=prompt,
        run_id="test",
    )

    # --- Write output files ---
    written: list[Path] = []
    for idx, data in enumerate(images, start=1):
        out_path = output_dir / f"test_V{idx}.png"
        out_path.write_bytes(data)
        written.append(out_path)

    # --- Report ---
    elapsed = time.time() - start
    print(f"\n[test_run] Images returned: {len(images)}")
    for p in written:
        size_kb = p.stat().st_size / 1024
        print(f"  -> {p.name}  ({size_kb:.1f} KB)")
    print(f"[test_run] Total elapsed: {elapsed:.1f} s")

    # --- Keep browser open for inspection ---
    input("\nPress Enter to close the browser...")
    context.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        input("\n[test_run] Failed. Press Enter to exit...")
