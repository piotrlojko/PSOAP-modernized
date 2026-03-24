from pathlib import Path
import tomllib


def test_build_system_requires_numpy_and_cython():
    repo_root = Path(__file__).resolve().parents[1]
    pyproject = repo_root / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    requires = data["build-system"]["requires"]

    assert any(req.startswith("numpy") for req in requires)
    assert any(req.startswith("cython") for req in requires)
