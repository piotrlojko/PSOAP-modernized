import re
from pathlib import Path

import psoap


def test_package_version_matches_setup_py():
    repo_root = Path(__file__).resolve().parents[1]
    setup_text = (repo_root / "setup.py").read_text(encoding="utf-8")
    match = re.search(r'version="([^"]+)"', setup_text)
    assert match is not None
    assert psoap.__version__ == match.group(1)
