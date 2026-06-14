"""Stable paths to backend assets regardless of module location."""

from pathlib import Path

# backend/  (parent of pathwise/)
BACKEND_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = BACKEND_ROOT / "data"
KB_DIR = BACKEND_ROOT / "knowledge_base"
MIGRATIONS_DIR = BACKEND_ROOT / "migrations"
SCRIPTS_DIR = BACKEND_ROOT / "scripts"
TESTS_DIR = BACKEND_ROOT / "tests"
