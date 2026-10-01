"""Callables stored in a design JSON as "module:qualname" references.

A study is authored in Python, where loops and functions are available, but
its record is JSON. Functions (an event generator, a solver-settings factory,
a data loader) cannot be written into JSON, so they are written as a
reference to where they are defined and imported again on load. That keeps
the JSON small and exact: the function is stored once, in its module, never
copied into rows.

Only module-level functions can be referenced. A lambda or a closure has no
importable name; ``to_ref`` refuses it rather than writing something that
cannot be loaded back.
"""
import importlib


def to_ref(fn):
    """Return "module:qualname" for a module-level callable."""
    if fn is None:
        return None
    if isinstance(fn, str):
        return fn
    mod = getattr(fn, "__module__", None)
    qual = getattr(fn, "__qualname__", None)
    if not mod or not qual or "<" in qual:
        raise ValueError(
            f"{fn!r} is not a module-level function, so a design JSON cannot "
            "refer to it. Define it at the top level of a module.")
    return f"{mod}:{qual}"


def from_ref(ref):
    """Import and return the callable named by a "module:qualname" string."""
    if ref is None or callable(ref):
        return ref
    mod_name, _, qual = ref.partition(":")
    if not qual:
        raise ValueError(f"{ref!r} is not a 'module:qualname' reference")
    obj = importlib.import_module(mod_name)
    for part in qual.split("."):
        obj = getattr(obj, part)
    return obj
