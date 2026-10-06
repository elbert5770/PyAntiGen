"""Per-model caches, stored on the RoadRunner instance without touching ``__getattr__``.

The engine keeps a few values on each RoadRunner so they are computed once per
model rather than once per evaluation (the state keys, the available symbols,
the parameter baseline, the configured tolerance). The obvious way to read one
back -- ``try: r._x`` / ``except AttributeError``, or ``getattr(r, "_x", None)`` --
is a trap, because RoadRunner defines::

    def __getattr__(self, name):
        if name != "this" and name in self._getIds(SelectionRecord_ALL):
            return self[name]
        raise AttributeError(name)

``__getattr__`` runs only after normal lookup has failed, so every *miss* builds
the list of every selection id the model has, to learn that the name is not one
of them. That list is roughly quadratic in model size: about a millisecond on a
25-species model, over a second on 400 species, and on a model with ~9,000 ODEs
~70 million ids, about 18 minutes and 16 GB per miss. A model pays it once per
cache slot per instance, which is why small models never notice and a large one
spends an hour before its first result.

Reading through the instance ``__dict__`` never reaches ``__getattr__``, so a
miss costs a dictionary lookup whatever the model size. Writing needs no care:
RoadRunner defines no ``__setattr__``, so ``r._x = v`` is an ordinary
instance-dict write.

Use :func:`get` and :func:`put` for every engine-private attribute on a
RoadRunner. ``tests/engine/test_model_cache.py`` fails if any of the engine's own
cache reads causes a ``_getIds`` call.
"""


def _target(r):
    """The object whose ``__dict__`` holds the caches.

    ``OptRoadRunnerProxy`` wraps a RoadRunner in ``_r`` and forwards attribute
    writes to it, so a cache written through the proxy lives on the RoadRunner.
    Reading ``vars(proxy)`` would look in the proxy's own dict instead, which is
    never where the value is.
    """
    try:
        d = vars(r)
    except TypeError:               # no __dict__ (slots, builtins): nothing cached
        return None
    inner = d.get("_r")
    if inner is not None and "_opt_param_names" in d:
        return inner
    return r


def get(r, name, default=None):
    """The value cached on *r* under *name*, or *default*. Never calls ``__getattr__``."""
    t = _target(r)
    if t is None:
        return default
    try:
        return vars(t).get(name, default)
    except TypeError:
        return default


def put(r, name, value):
    """Cache *value* on *r* under *name*. Returns whether it was stored.

    A target that refuses attributes is tolerated, as the call sites always did:
    the value is then recomputed next time instead of reused.
    """
    t = _target(r)
    if t is None:
        return False
    try:
        vars(t)[name] = value
    except Exception:
        return False
    return True
