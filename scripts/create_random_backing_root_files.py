#!/usr/bin/env python3
"""Create 10,000 random ~1KB text files under the configured backing_root.

The script reads /home/g122622/dev/GLFS/configs/default.json, extracts
`fs.backing_root`, creates the directory if needed, and writes files named
1.txt through 10000.txt.
"""

from __future__ import annotations

import json
import secrets
import string
from pathlib import Path

CONFIG_PATH = Path("/home/g122622/dev/GLFS/configs/default.json")
FILE_COUNT = 10_000
FILE_SIZE = 1024
ALPHABET = string.ascii_letters + string.digits


def load_backing_root(config_path: Path) -> Path:
    with config_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    try:
        backing_root = data["fs"]["backing_root"]
    except KeyError as exc:
        raise KeyError("missing required config field: fs.backing_root") from exc

    if not isinstance(backing_root, str) or not backing_root.strip():
        raise ValueError("fs.backing_root must be a non-empty string")

    return Path(backing_root).expanduser()


def random_text(size: int) -> str:
    return "".join(secrets.choice(ALPHABET) for _ in range(size))


def main() -> None:
    backing_root = load_backing_root(CONFIG_PATH)
    backing_root.mkdir(parents=True, exist_ok=True)

    for i in range(1, FILE_COUNT + 1):
        file_path = backing_root / f"{i}.txt"
        file_path.write_text(random_text(FILE_SIZE), encoding="utf-8")

    print(f"created {FILE_COUNT} files in {backing_root}")


if __name__ == "__main__":
    main()
