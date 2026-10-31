"""Shared access to validated, local domain-pack data."""

from pathlib import Path

import yaml
from pydantic import BaseModel

from app.config import get_settings

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKS_ROOT = PROJECT_ROOT / "packs"


def pack_directory() -> Path:
    """Return the configured pack directory without allowing path traversal."""
    pack_name = get_settings().pack
    if not pack_name.replace("_", "").isalnum():
        raise ValueError(f"Invalid pack name: {pack_name!r}")
    path = (PACKS_ROOT / pack_name).resolve()
    if path.parent != PACKS_ROOT.resolve():
        raise ValueError(f"Invalid pack name: {pack_name!r}")
    return path


def load_yaml_model[T: BaseModel](filename: str, model: type[T]) -> T:
    """Load a pack YAML file and validate it against its Pydantic schema."""
    path = pack_directory() / filename
    with path.open(encoding="utf-8") as file:
        data = yaml.safe_load(file)
    return model.model_validate(data)
