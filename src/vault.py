"""Vault storage for generated artwork outputs."""

from __future__ import annotations

from pathlib import Path


VAULT_DIR = Path("./vault")


def save_to_vault(
    images: list[bytes],
    client: str,
    task_id: str,
    originals: list[bytes] | None = None,
) -> list[Path]:
    """Save generated images into the vault with run numbering.

    Directory structure:
        vault/{client}/{task_id}/run_{n}/{task_id}_R{n}_V{i}.png
        vault/{client}/{task_id}/run_{n}/{task_id}_R{n}_V{i}_original.png

    If `originals` is provided, saves each original alongside the processed
    version so they can be compared if something goes wrong.

    Determines the run number by counting existing run_* folders for that task.
    Returns the list of written file paths (processed versions).
    """
    task_dir = VAULT_DIR / client / task_id
    task_dir.mkdir(parents=True, exist_ok=True)

    # Determine next run number
    existing_runs = [
        d for d in task_dir.iterdir()
        if d.is_dir() and d.name.startswith("run_")
    ]
    run_number = len(existing_runs) + 1

    run_dir = task_dir / f"run_{run_number}"
    run_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for i, data in enumerate(images, start=1):
        # Processed version
        filename = f"{task_id}_R{run_number}_V{i}.png"
        path = run_dir / filename
        path.write_bytes(data)
        written.append(path)

        # Original version (if provided)
        if originals and i <= len(originals):
            orig_filename = f"{task_id}_R{run_number}_V{i}_original.png"
            orig_path = run_dir / orig_filename
            orig_path.write_bytes(originals[i - 1])

    return written
