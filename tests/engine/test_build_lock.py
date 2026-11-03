"""_build_lock serializes one model's compile across processes and threads.

It is a POSIX lock because flock is node-local on GPFS: a Slurm array's tasks on different nodes would all hold a flock at
once and build into the same directory. This test cannot span nodes, but it checks the property the lock must have on one:
no two holders overlap, whether they are processes or threads of one process.
"""
import multiprocessing as mp
import os
import threading
import time

from pyantigen.engine.Analytic_gradient import _build_lock


def _hold(directory, log, seconds):
    with _build_lock(directory):
        start = time.time()
        time.sleep(seconds)
        log.append((start, time.time()))


def _intervals_overlap(intervals):
    intervals = sorted(intervals)
    return any(b[0] < a[1] for a, b in zip(intervals, intervals[1:]))


def test_processes_do_not_overlap(tmp_path):
    directory = str(tmp_path / "model_abc")
    with mp.get_context("spawn").Manager() as m:
        log = m.list()
        procs = [mp.get_context("spawn").Process(target=_hold, args=(directory, log, 0.3)) for _ in range(3)]
        for p in procs:
            p.start()
        for p in procs:
            p.join(30)
        assert all(p.exitcode == 0 for p in procs)
        assert len(log) == 3
        assert not _intervals_overlap(list(log))
    assert os.path.exists(directory + ".lock")


def test_threads_do_not_overlap(tmp_path):
    directory = str(tmp_path / "model_abc")
    log = []
    threads = [threading.Thread(target=_hold, args=(directory, log, 0.2)) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
    assert len(log) == 3
    assert not _intervals_overlap(log)
