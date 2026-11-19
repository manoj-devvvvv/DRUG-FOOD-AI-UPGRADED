"""Load `dfi/.env` for the API and CLI.

Shell variables always win over file values, so a deployment platform can inject
secrets without editing the file. `.env` is ignored by git; `.env.example` is the
committed template.
"""
from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

from common import ROOT

ENV_FILE = ROOT / ".env"


def load_env_file(path: str | Path) -> bool:
    target = Path(path)
    if not target.is_file():
        return False
    return bool(load_dotenv(target, override=False))


def load_project_env() -> bool:
    return load_env_file(ENV_FILE)
