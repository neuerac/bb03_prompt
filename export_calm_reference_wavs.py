"""Export pure calm BB03 WAVs for manual reference-audio listening.

Run this script where both BB03_51h_cleaned.jsonl and the BB03 audio root
are accessible. It selects single-sentence, 8-40 second samples whose only
emotion tag is calmness/calmness1/calmness2/calmness3.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path


TAG_RE = re.compile(r"【([^】]+)】")
CALM_LABELS = {"calmness", "calmness1", "calmness2", "calmness3"}


def is_pure_calm(record: dict) -> bool:
    text = record.get("text", "")
    tags = TAG_RE.findall(text)
    duration = record.get("timestamp", [None])[0]
    return (
        len(tags) == 1
        and tags[0] in CALM_LABELS
        and isinstance(duration, (int, float))
        and 8 <= duration <= 40
        and "/单句/" in record.get("audio_path", "")
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bb03_jsonl", default="BB03_51h_cleaned.jsonl")
    parser.add_argument("--bb03_root", required=True, help="BB03 audio root directory")
    parser.add_argument("--out_dir", default="calm_reference_wavs")
    args = parser.parse_args()

    jsonl_path = Path(args.bb03_jsonl)
    audio_root = Path(args.bb03_root)
    out_dir = Path(args.out_dir)
    candidates = []

    with jsonl_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if is_pure_calm(record):
                record["source_emotion"] = TAG_RE.findall(record["text"])[0]
                record["duration_sec"] = record["timestamp"][0]
                candidates.append(record)

    if not candidates:
        raise RuntimeError("No pure calm candidates were found")

    missing = [str(audio_root / record["audio_path"]) for record in candidates if not (audio_root / record["audio_path"]).is_file()]
    if missing:
        print("Missing BB03 WAVs:", file=sys.stderr)
        print("\n".join(missing), file=sys.stderr)
        raise SystemExit(1)

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "calm_reference_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for index, record in enumerate(candidates, start=1):
            source = audio_root / record["audio_path"]
            name = f"{index:02d}_{record['source_emotion']}_{Path(record['audio_path']).name}"
            destination = out_dir / name
            shutil.copy2(source, destination)
            manifest_record = {
                "index": index,
                "local_wav": str(destination),
                "source_audio_path": record["audio_path"],
                "key": record.get("key", ""),
                "source_emotion": record["source_emotion"],
                "duration_sec": record["duration_sec"],
                "text": record["text"],
            }
            manifest.write(json.dumps(manifest_record, ensure_ascii=False) + "\n")

    counts = {}
    for record in candidates:
        label = record["source_emotion"]
        counts[label] = counts.get(label, 0) + 1
    print(f"Exported {len(candidates)} calm WAVs to {out_dir}")
    print("Label distribution:", counts)
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
