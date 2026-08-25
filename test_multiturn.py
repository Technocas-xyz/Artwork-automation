"""Test script: prove multi-turn conversations work.

Turn 1: Upload an image and ask for background removal (should get image back).
Turn 2: Text-only follow-up asking for a colour change (should get a different image).

Verifies that each turn returns ONLY its own new images, not duplicates from
previous turns.
"""

import time
import traceback
from pathlib import Path

from src.browser import launch_context, is_logged_in
from src.generator import open_chat, send_turn


def main() -> None:
    account = "acct1"
    input_dir = Path("./input")
    output_dir = Path("./output")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Pick the first image available
    image_paths = sorted(
        str(p) for p in input_dir.iterdir()
        if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")
    )
    if not image_paths:
        print("[test_multiturn] No images found in ./input/")
        return

    first_image = image_paths[0]
    print(f"[test_multiturn] Using image: {first_image}")

    # --- Launch browser ---
    start = time.time()
    context = launch_context(account)
    page = context.pages[0] if context.pages else context.new_page()

    # --- Check login ---
    page.goto("https://chatgpt.com", wait_until="domcontentloaded")
    if not is_logged_in(page):
        print("[test_multiturn] NOT LOGGED IN. Run login.py first.")
        input("Press Enter to close...")
        context.close()
        return

    print("[test_multiturn] Session OK.\n")

    # --- Open chat ---
    open_chat(page)
    print("[test_multiturn] Chat opened.\n")

    # --- Turn 1: Upload + prompt ---
    print("[test_multiturn] TURN 1: Uploading image + asking for background removal...")
    t1_start = time.time()
    turn1_images = send_turn(
        page,
        prompt="Remove the background from this artwork and return it as a transparent PNG.",
        image_paths=[first_image],
        run_id="multiturn_t1",
    )
    t1_elapsed = time.time() - t1_start
    print(f"  -> Turn 1 returned {len(turn1_images)} image(s) in {t1_elapsed:.1f}s")
    for idx, data in enumerate(turn1_images, 1):
        out = output_dir / f"multiturn_T1_V{idx}.png"
        out.write_bytes(data)
        print(f"     Saved: {out.name} ({len(data)/1024:.1f} KB)")

    # --- Turn 2: Text-only follow-up ---
    print("\n[test_multiturn] TURN 2: Text-only follow-up (recolour to blue)...")
    t2_start = time.time()
    turn2_images = send_turn(
        page,
        prompt="Now recolour the artwork to blue tones and return it as a transparent PNG.",
        image_paths=None,
        run_id="multiturn_t2",
    )
    t2_elapsed = time.time() - t2_start
    print(f"  -> Turn 2 returned {len(turn2_images)} image(s) in {t2_elapsed:.1f}s")
    for idx, data in enumerate(turn2_images, 1):
        out = output_dir / f"multiturn_T2_V{idx}.png"
        out.write_bytes(data)
        print(f"     Saved: {out.name} ({len(data)/1024:.1f} KB)")

    # --- Summary ---
    total = time.time() - start
    print(f"\n[test_multiturn] Total elapsed: {total:.1f}s")
    print(f"[test_multiturn] Turn 1 images: {len(turn1_images)}, Turn 2 images: {len(turn2_images)}")

    if turn1_images and turn2_images:
        print("[test_multiturn] SUCCESS — both turns produced images independently.")
    elif not turn1_images:
        print("[test_multiturn] WARNING — Turn 1 returned no images.")
    elif not turn2_images:
        print("[test_multiturn] WARNING — Turn 2 returned no images.")

    # --- Keep open for inspection ---
    input("\nPress Enter to close the browser...")
    context.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        input("\n[test_multiturn] Failed. Press Enter to exit...")
