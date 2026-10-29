"""The array tasks a stage left unfinished, as a comma list ready for ``sbatch --array=``.

A profile stage (``Profile_amici``) and a multi-start (``Multistart_amici``) both end by saying which tasks still have to run. The list
is written to ``unfinished.txt`` in the stage's own directory, and also to the file named by ``PYANTIGEN_UNFINISHED_FILE`` when
that is set, so whatever submits the jobs can find it without knowing where the engine keeps its files. An empty file means done.
"""
import os

ENV = "PYANTIGEN_UNFINISHED_FILE"


def _write(path, text):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as fh:
        fh.write(text)
    os.replace(tmp, path)


def write(directory, tasks):
    text = ",".join(str(t) for t in sorted(tasks)) + "\n" if tasks else ""
    _write(os.path.join(directory, "unfinished.txt"), text)
    if os.environ.get(ENV):
        _write(os.environ[ENV], text)
