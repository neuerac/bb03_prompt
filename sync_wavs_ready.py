"""Copy approved WAVs into wavs_ready.

Run this script from inside the infer_out_v1 directory. It keeps the source
WAVs intact and overwrites only the destination IDs listed below.
"""

from pathlib import Path
import shutil
import sys

# Newly generated WAVs accepted in the latest review.
GENERATED_IDS = (
    "control_0002",
    "control_0003",
    "control_0023",
    "control_0024",
    "control_0025",
    "control_0033",
    "control_0162",
    "control_0168",
    "control_0169",
)

# New v1 ID -> source WAV ID in ../infer_out_v1_old/wavs_ready.
OLD_AUDIO_MAP = {
    "control_0001": "control_0001",
    "control_0046": "control_0057",
}

root = Path.cwd()
generated_dir = root / "wavs"
ready_dir = root / "wavs_ready"
old_ready_dir = root.parent / "infer_out_v1_old" / "wavs_ready"

copies = []
for item_id in GENERATED_IDS:
    copies.append((generated_dir / f"{item_id}.wav", ready_dir / f"{item_id}.wav"))
for new_id, old_id in OLD_AUDIO_MAP.items():
    copies.append((old_ready_dir / f"{old_id}.wav", ready_dir / f"{new_id}.wav"))

missing = [str(source) for source, _ in copies if not source.is_file()]
if missing:
    print("Missing source WAVs:", file=sys.stderr)
    print("\n".join(missing), file=sys.stderr)
    raise SystemExit(1)

ready_dir.mkdir(exist_ok=True)
for source, destination in copies:
    shutil.copy2(source, destination)
    print(f"{source} -> {destination}")

print(f"Copied or replaced {len(copies)} WAVs.")
