"""
Pytest configuration and fixtures for PyAntiGen tests.
"""
import os
import pytest


def _find_silk_project_path():
    """Resolve path to SILK project: local tests/silk_fixtures first, then env or sibling Elbert_2022_SILK."""
    this_dir = os.path.dirname(os.path.abspath(__file__))
    # Prefer local copy in tests/silk_fixtures (scripts + modules)
    local_fixtures = os.path.join(this_dir, "silk_fixtures")
    if os.path.isdir(local_fixtures) and os.path.isdir(os.path.join(local_fixtures, "scripts")):
        return os.path.abspath(local_fixtures)
    env_path = os.environ.get("SILK_PROJECT_PATH", "").strip()
    if env_path and os.path.isdir(env_path):
        return os.path.abspath(env_path)
    pyantigen_root = os.path.dirname(this_dir)
    parent = os.path.dirname(pyantigen_root)
    sibling = os.path.join(parent, "Elbert_2022_SILK")
    if os.path.isdir(sibling):
        return sibling
    return None


@pytest.fixture(scope="session")
def silk_project_path():
    """Path to the SILK project (scripts, modules, generated). Prefers tests/silk_fixtures when present."""
    path = _find_silk_project_path()
    if path is None:
        pytest.skip(
            "SILK project not found. Use tests/silk_fixtures (in-repo copy), set SILK_PROJECT_PATH, or place Elbert_2022_SILK next to PyAntiGen."
        )
    return path


@pytest.fixture(scope="session")
def pyantigen_root():
    """Root directory of the PyAntiGen package."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
