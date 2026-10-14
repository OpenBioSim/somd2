from somd2._viewer._data import Simulation, _LogMonitor


def _line(message, level="INFO"):
    return (
        f"2026-10-13 21:00:00.000 | {level:<8} | somd2.runner._repex:run:1 - {message}"
    )


def _cycle(n, total, hot=None):
    lines = [
        _line(f"Running dynamics for cycle {n} of {total}"),
        _line("States: [0 1 2]"),
    ]
    if hot is not None:
        lines.append(_line(f"Propagating {len(hot)} hot replicas: {hot}"))
    return lines


def _monitor(tmp_path, lines):
    path = tmp_path / "log.txt"
    path.write_text("\n".join(lines) + "\n")
    log = _LogMonitor(path)
    log.update()
    return log


def test_adaptive_summary(tmp_path):
    """
    Validate that the rankings and hot-only cycles of an adaptive replica
    exchange run are read from the log and summarised.
    """
    lines = (
        _cycle(1, 4)
        + [
            _line("Hot replicas: [1, 2]"),
            _line("Estimated cost 0.75 and variance 1.10 relative to standard"),
        ]
        + _cycle(2, 4, hot=[1, 2])
        + _cycle(3, 4)
        + [
            _line("All replicas hot"),
            _line(
                "Estimated cost per cycle 1.00 and final variance 1.00 relative "
                "to standard"
            ),
        ]
        + _cycle(4, 4)
    )
    log = _monitor(tmp_path, lines)

    summary = Simulation._adaptive(log, 3, {"hot_replicas": None})

    assert summary["hot"] == [0, 1, 2]
    assert summary["ranked_cycle"] == 3
    assert summary["cost"] == 1.0 and summary["variance"] == 1.0
    assert summary["hot_only_fraction"] == 0.25
    assert summary["history"] == {"cycle": [1, 3], "hot": [[0, 1], [1, 1], [1, 1]]}


def test_adaptive_rankings_after_restart(tmp_path):
    """
    Validate that rankings and hot-only cycles logged after the last saved
    state are discarded on restart, since the run resumes from that state.
    """
    lines = (
        _cycle(1, 3)
        + [_line("Saving replica exchange state")]
        + _cycle(2, 3, hot=[1, 2])
        + [
            _line("Hot replicas: [1, 2]"),
            _line("Estimated cost 0.75 and variance 1.10 relative to standard"),
            _line("somd2 version: test"),
        ]
        + _cycle(2, 3)
    )
    log = _monitor(tmp_path, lines)

    assert log.rankings == []
    assert log.hot_only_cycles == []


def test_adaptive_rankings_kept_after_final_save(tmp_path):
    """
    Validate that a final save, made when the last cycle isn't a checkpoint,
    keeps the rankings and hot-only cycles before it when the run is extended.
    """
    lines = (
        _cycle(1, 2)
        + [
            _line("Hot replicas: [1, 2]"),
            _line("Estimated cost 0.75 and variance 1.10 relative to standard"),
        ]
        + _cycle(2, 2, hot=[1, 2])
        + [
            _line("Saving final replica exchange state"),
            _line("somd2 version: test"),
        ]
        + _cycle(1, 1)
    )
    log = _monitor(tmp_path, lines)

    assert len(log.rankings) == 1
    assert log.hot_only_cycles == [1]


def test_adaptive_history_is_thinned(tmp_path):
    """
    Validate that a long ranking history is thinned, keeping the latest.
    """
    lines = []
    for n in range(1, 11):
        lines += _cycle(n, 10) + [
            _line("Hot replicas: [1, 2]"),
            _line("Estimated cost 0.75 and variance 1.10 relative to standard"),
        ]
    log = _monitor(tmp_path, lines)

    summary = Simulation._adaptive(log, 3, None, max_rankings=4)

    assert len(summary["history"]["cycle"]) <= 4
    assert summary["history"]["cycle"][-1] == 10
