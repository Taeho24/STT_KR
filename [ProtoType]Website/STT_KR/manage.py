#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys
from pathlib import Path


def main():
    """Run administrative tasks."""
    # Ensure repository root (two levels up from [ProtoType]Website/STT_KR) is on sys.path
    # This lets Django import the latest core modules (audio_analyzer, emotion_classifier, etc.)
    try:
        repo_root = Path(__file__).resolve().parents[2]
        repo_root_str = str(repo_root)
        if repo_root_str not in sys.path:
            sys.path.insert(0, repo_root_str)
    except Exception:
        # Non-fatal: fall back to local utils
        pass
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'STT_KR.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == '__main__':
    main()
