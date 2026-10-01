"""PyAntiGen: a declarative framework for building compartmental Antimony models."""

# Single source of truth for the version is the git tag, via setuptools-scm.
# _version.py is written at build time and ships inside the wheel; it is absent
# in a plain source checkout, so fall back to the installed distribution
# metadata, and finally to a clearly-invalid marker.
try:
    from ._version import __version__
except ImportError:  # pragma: no cover - depends on how the package was obtained
    try:
        from importlib.metadata import PackageNotFoundError, version as _version

        try:
            __version__ = _version("pyantigen")
        except PackageNotFoundError:
            __version__ = "0+unknown"
    except ImportError:
        __version__ = "0+unknown"

__all__ = ["__version__"]
