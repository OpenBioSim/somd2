import numpy as np
import pytest

from somd2.runner._base import RunnerBase
from somd2.runner._repex import DynamicsCache


def counters(num_moves=0):
    """A flat dictionary of counters, as written before the format changed."""
    return {
        "num_moves": num_moves,
        "num_accepted": num_moves,
        "num_insertions": 0,
        "num_deletions": 0,
        "num_accepted_attempts": 0,
    }


class TestLegacyDetection:
    """Tests for detecting the format of GCMC statistics."""

    def test_flat_counters_are_legacy(self):
        assert RunnerBase._is_legacy_gcmc_stats(counters(5))

    def test_keyed_by_lambda_is_current(self):
        assert not RunnerBase._is_legacy_gcmc_stats({"0.00000": counters(5)})

    @pytest.mark.parametrize("stats", [None, {}, []])
    def test_other_values_are_not_legacy(self, stats):
        assert not RunnerBase._is_legacy_gcmc_stats(stats)

    def test_conversion_keys_by_lambda(self):
        converted = RunnerBase._convert_legacy_gcmc_stats(counters(5), 0.33333)
        assert converted == {"0.33333": counters(5)}

    def test_conversion_is_a_copy(self):
        """The original must not be aliased into the converted result."""
        original = counters(5)
        converted = RunnerBase._convert_legacy_gcmc_stats(original, 0.0)
        original["num_moves"] = 99
        assert converted["0.00000"]["num_moves"] == 5


class TestLegacyRepexCheckpoint:
    """Tests for restoring a replica exchange checkpoint."""

    @staticmethod
    def make_state(gcmc_stats, lambdas=(0.0, 0.5, 1.0)):
        n = len(lambdas)
        return {
            "_lambdas": list(lambdas),
            "_rest2_scale_factors": [1.0] * n,
            "_states": np.arange(n),
            "_time": None,
            "_openmm_states": [None] * n,
            "_gcmc_samplers": [None] * n,
            "_gcmc_states": [None] * n,
            "_gcmc_stats": gcmc_stats,
            "_terminal_flip_stats": [[0, 0]] * n,
            "_num_proposed": np.zeros((n, n)),
            "_num_accepted": np.zeros((n, n)),
            "_num_swaps": np.zeros((n, n)),
        }

    def restore(self, gcmc_stats, **kwargs):
        cache = object.__new__(DynamicsCache)
        cache.__setstate__(self.make_state(gcmc_stats, **kwargs))
        return cache

    def test_per_replica_list_is_converted(self):
        """A list of counters per replica becomes a map keyed by lambda."""
        cache = self.restore([counters(i) for i in range(3)])

        assert cache._gcmc_stats == {
            "0.00000": counters(0),
            "0.50000": counters(1),
            "1.00000": counters(2),
        }

    def test_no_gcmc_gives_none(self):
        """A checkpoint from a run without GCMC has no statistics."""
        assert self.restore([None, None, None])._gcmc_stats is None

    def test_current_format_is_untouched(self):
        """A checkpoint already in the current format is left alone."""
        stats = {"0.00000": counters(4), "1.00000": counters(9)}
        assert self.restore(dict(stats))._gcmc_stats == stats

    def test_missing_attribute_defaults_to_none(self):
        """A checkpoint predating GCMC statistics has none."""
        state = self.make_state(None)
        del state["_gcmc_stats"]

        cache = object.__new__(DynamicsCache)
        cache.__setstate__(state)

        assert cache._gcmc_stats is None
