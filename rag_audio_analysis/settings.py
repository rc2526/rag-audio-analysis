from configparser import RawConfigParser
from functools import lru_cache
from pathlib import Path
from typing import List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SETTINGS_PATH = PROJECT_ROOT / "settings.ini"


@lru_cache(maxsize=1)
def load_settings() -> RawConfigParser:
    parser = RawConfigParser()
    if SETTINGS_PATH.exists():
        parser.read(SETTINGS_PATH, encoding="utf-8")
    return parser


def get_str(section: str, option: str, fallback: str = "") -> str:
    parser = load_settings()
    return parser.get(section, option, fallback=fallback)


def get_int(section: str, option: str, fallback: int = 0) -> int:
    parser = load_settings()
    return parser.getint(section, option, fallback=fallback)


def get_float(section: str, option: str, fallback: float = 0.0) -> float:
    parser = load_settings()
    return parser.getfloat(section, option, fallback=fallback)


def get_list(section: str, option: str, fallback: Optional[List[str]] = None) -> List[str]:
    value = get_str(section, option, "")
    if not value:
        return list(fallback or [])
    return [item.strip() for item in value.split(",") if item.strip()]
