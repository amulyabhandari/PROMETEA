from __future__ import annotations

from pathlib import Path


CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"


def load_prompt(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def load_config_prompt(file_name: str) -> str:
    return load_prompt(CONFIG_DIR / file_name)
