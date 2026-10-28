"""A function in a fixed-sigma block key (a derived observable) must not put its memory address into the profile directory's hash."""
from pyantigen.engine.Profile_checkpoint import _stable_key, spec_fingerprint


def _observable():                       # a module-level function, as a derived observable is
    return 1.0


def test_a_function_in_a_sigma_key_does_not_change_the_hash_with_its_address():
    key = ("blk", _observable)
    assert "0x" not in _stable_key(key) and "_observable" in _stable_key(key)
    assert _stable_key(("blk", "MFL42")) == str(("blk", "MFL42"))        # a plain key is unchanged: old directories stay valid
    a = spec_fingerprint(["p"], [1.0], {"g": 1}, ["lin"], "m", fixed_sigmas={key: 0.5})
    b = spec_fingerprint(["p"], [1.0], {"g": 1}, ["lin"], "m", fixed_sigmas={("blk", _observable): 0.5})
    assert a == b
