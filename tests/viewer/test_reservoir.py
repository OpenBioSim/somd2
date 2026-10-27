from somd2._viewer._data import Simulation, _LogMonitor


def _line(message, level="INFO"):
    return (
        f"2026-10-27 21:00:00.000 | {level:<8} | somd2.runner._repex:run:1 - {message}"
    )


def _cycle(n, total, states):
    return [
        _line(f"Running dynamics for cycle {n} of {total}"),
        _line(f"States: [{' '.join(str(s) for s in states)}]"),
    ]


def _swap(frame, delta_e):
    return _line(
        f"Reservoir frame {frame} swapped in at λ = 1.00000 (ΔE = {delta_e} kJ/mol)"
    )


def _monitor(tmp_path, lines):
    path = tmp_path / "log.txt"
    path.write_text("\n".join(lines) + "\n")
    log = _LogMonitor(path)
    log.update()
    return log


def test_reservoir_summary(tmp_path):
    """
    Validate that reservoir swaps are read from the log, and that each
    swapped-in configuration is followed until a later swap replaces it.
    """
    lines = (
        [_line("Using a reservoir of 5 frames from apo.dcd")]
        + _cycle(1, 3, [0, 1, 2])
        + [_swap(3, "10.00")]
        + _cycle(2, 3, [0, 2, 1])
        + [
            _swap(4, "-2.00"),
            _line(
                "Reservoir frame 7 was rejected at λ = 1.00000 because its energy "
                "isn't finite. Check that the reservoir atoms match the system.",
                level="WARNING",
            ),
        ]
        + _cycle(3, 3, [1, 0, 2])
    )
    log = _monitor(tmp_path, lines)

    summary = Simulation._reservoir(log, [0.0, 0.5, 1.0], False)

    assert summary["window"] == 2
    assert summary["swaps"] == 2 and summary["rejected"] == 1
    assert summary["frames_used"] == 2 and summary["num_frames"] == 5
    assert summary["delta_e_mean"] == 4.0
    assert summary["delta_e_min"] == -2.0 and summary["delta_e_max"] == 10.0

    # The first configuration reaches window 0. The second stays at window 2.
    assert summary["furthest"] == [1, 0, 1]
    assert summary["reached_end"] == 1
    assert summary["swap_cycles"] == [0, 1]


def test_reservoir_swaps_dropped_on_restart(tmp_path):
    """
    Validate that swaps logged after the last saved state are dropped on
    restart, along with their cycles.
    """
    lines = (
        _cycle(1, 2, [0, 1, 2])
        + [_swap(3, "1.00"), _line("Saving replica exchange state")]
        + _cycle(2, 2, [0, 1, 2])
        + [_swap(4, "1.00"), _line("somd2 version: 2026.3.0")]
    )
    log = _monitor(tmp_path, lines)

    assert log.reservoir_swaps == [(0, 3, 1.0, True)]
