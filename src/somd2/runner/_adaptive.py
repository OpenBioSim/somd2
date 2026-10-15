######################################################################
# SOMD2: GPU accelerated alchemical free-energy engine.
#
# Copyright: 2023-2026
#
# Authors: The OpenBioSim Team <team@openbiosim.org>
#
# SOMD2 is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# SOMD2 is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with SOMD2. If not, see <http://www.gnu.org/licenses/>.
#####################################################################

__all__ = [
    "WindowRanker",
    "cold_rate",
    "exp_estimate",
    "is_unconverged",
    "statistical_inefficiency",
]

from math import ceil as _ceil

import numpy as _np

from somd2 import _logger


def statistical_inefficiency(x):
    """
    Estimate the statistical inefficiency of a time series, i.e. the number of
    samples per independent sample.

    Parameters
    ----------

    x: np.ndarray
        The time series.

    Returns
    -------

    float
        The statistical inefficiency.
    """
    n = len(x)
    if n < 2:
        return 1.0

    dx = x - x.mean()
    var = dx.dot(dx) / n
    if var == 0.0:
        return 1.0

    # The unnormalised autocovariance at every lag.
    f = _np.fft.rfft(dx, 2 * n)
    acov = _np.fft.irfft(f * _np.conj(f))[:n]

    # Sum the autocorrelation function up to its first non-positive value.
    g = 1.0
    for t in range(1, n):
        c = acov[t] / ((n - t) * var)
        if c <= 0.0:
            break
        g += 2.0 * c * (1.0 - t / n)

    return max(1.0, g)


def exp_estimate(du):
    """
    The free energy difference from exponential averaging of reduced energy
    differences.

    Parameters
    ----------

    du: np.ndarray
        The reduced energy differences.

    Returns
    -------

    float
        The reduced free energy difference.
    """
    m = (-du).max()
    return -(m + _np.log(_np.mean(_np.exp(-du - m))))


def is_unconverged(fwd, rev, g_fwd, g_rev):
    """
    Whether the free energy difference between a pair of neighbouring windows
    shows signs of poor sampling, regardless of its uncertainty.

    Parameters
    ----------

    fwd: np.ndarray
        The reduced energy difference to the upper window, sampled in the lower
        window.

    rev: np.ndarray
        The reduced energy difference to the lower window, sampled in the upper
        window.

    g_fwd, g_rev: float
        The statistical inefficiencies of fwd and rev.

    Returns
    -------

    bool
        Whether the pair is unconverged.
    """
    # The forward and reverse estimates disagree by more than 1 kT.
    if abs(exp_estimate(fwd) + exp_estimate(rev)) > 1.0:
        return True

    # The estimates from the two halves of the data differ by more than 0.25 kT,
    # and by more than twice their combined standard error.
    estimates = []
    errors = []
    for half in range(2):
        f = _np.array_split(fwd, 2)[half]
        r = _np.array_split(rev, 2)[half]
        estimates.append(0.5 * (exp_estimate(f) - exp_estimate(r)))
        errors.append(
            0.5 * _np.sqrt(f.var() * g_fwd / len(f) + r.var() * g_rev / len(r))
        )

    drift = abs(estimates[0] - estimates[1])
    return drift > 0.25 and drift > 2.0 * _np.hypot(*errors)


def cold_rate(first_cycle, num_cycles, checkpoint_cycles, mix_interval):
    """
    The share of the cycles left that propagate every window, and so the cold
    ones: the mix and checkpoint cycles, and the last cycle.

    Parameters
    ----------

    first_cycle: int
        The number of cycles run so far.

    num_cycles: int
        The number of cycles in the run.

    checkpoint_cycles: np.ndarray
        The checkpoint cycles, numbered from one, in increasing order.

    mix_interval: int
        The number of cycles between mix cycles.

    Returns
    -------

    float
        The share of cycles.
    """
    if first_cycle >= num_cycles:
        return 1.0

    # Mix cycles, then checkpoint cycles that aren't also mix cycles.
    full = num_cycles // mix_interval - first_cycle // mix_interval
    start, end = _np.searchsorted(
        checkpoint_cycles, [first_cycle, num_cycles], side="right"
    )
    checkpoints = checkpoint_cycles[start:end]
    full += int((checkpoints % mix_interval != 0).sum())

    # The last cycle always propagates every window.
    if num_cycles % mix_interval != 0 and (
        len(checkpoints) == 0 or checkpoints[-1] != num_cycles
    ):
        full += 1

    return full / (num_cycles - first_cycle)


class WindowRanker:
    """
    The data used to rank the lambda windows for adaptive replica exchange, and
    the ranking that chooses the hot windows.
    """

    def __init__(
        self,
        num_windows,
        min_samples,
        proposed,
        accepted,
        is_ring_break=False,
        is_cached=False,
    ):
        """
        Constructor.

        Parameters
        ----------

        num_windows: int
            The number of lambda windows.

        min_samples: int
            The number of independent samples a window needs before it can be
            made cold.

        proposed: np.ndarray
            The number of swaps proposed between each pair of windows so far,
            from which swaps are counted for the ranking.

        accepted: np.ndarray
            The number of swaps accepted between each pair of windows so far.

        is_ring_break: bool
            Whether the distance of the bond that is broken or made is
            recorded, in which case windows where it is slow are kept hot.

        is_cached: bool
            Whether contexts are shared between replicas, in which case the hot
            windows aren't topped up to fill the GPUs.
        """
        self._num_windows = num_windows
        self._min_samples = min_samples
        self._is_ring_break = is_ring_break
        self._is_cached = is_cached
        self.reset(proposed, accepted)

    def reset(self, proposed, accepted):
        """
        Clear the recorded data.

        Parameters
        ----------

        proposed: np.ndarray
            The number of swaps proposed between each pair of windows so far.

        accepted: np.ndarray
            The number of swaps accepted between each pair of windows so far.
        """
        # The reduced energy difference to the lower and upper neighbouring
        # window, sampled in each window.
        self.lower = [[] for _ in range(self._num_windows)]
        self.upper = [[] for _ in range(self._num_windows)]

        # The distance of the bond that is broken or made, sampled in each
        # window.
        self.distance = [[] for _ in range(self._num_windows)]

        self.set_swap_baseline(proposed, accepted)

    def set_swap_baseline(self, proposed, accepted):
        """
        Set the swap counts from which swaps are counted for the ranking.

        Parameters
        ----------

        proposed: np.ndarray
            The number of swaps proposed between each pair of windows so far.

        accepted: np.ndarray
            The number of swaps accepted between each pair of windows so far.
        """
        self.proposed = proposed.copy()
        self.accepted = accepted.copy()

    def record(self, energy_matrix, windows, distances=None):
        """
        Record the reduced energy differences between neighbouring windows, and
        the distance of the bond that is broken or made, if any. Must be called
        before the replicas are mixed.

        Parameters
        ----------

        energy_matrix: np.ndarray
            The reduced energy of each replica at each window.

        windows: list of int
            The windows that were propagated.

        distances: dict
            The bond distance in each window that was propagated, which is
            required for a ring-breaking perturbation.
        """
        if self._is_ring_break and distances is None:
            raise ValueError(
                "Bond distances are required for a ring-breaking perturbation"
            )

        for k in windows:
            row = energy_matrix[k]
            if k > 0:
                self.lower[k].append(row[k - 1] - row[k])
            if k < self._num_windows - 1:
                self.upper[k].append(row[k + 1] - row[k])
            if distances is not None:
                self.distance[k].append(distances[k])

    def to_dict(self):
        """
        The recorded data, to be stored in a checkpoint.

        Returns
        -------

        dict
            The recorded data.
        """
        return {
            "lower": [_np.array(x) for x in self.lower],
            "upper": [_np.array(x) for x in self.upper],
            "distance": [_np.array(x) for x in self.distance],
            "proposed": self.proposed,
            "accepted": self.accepted,
        }

    def restore(self, data):
        """
        Restore the recorded data from a checkpoint.

        Parameters
        ----------

        data: dict
            The recorded data, as returned by to_dict(), or None.
        """
        if data is None:
            return

        self.lower = [list(x) for x in data["lower"]]
        self.upper = [list(x) for x in data["upper"]]
        self.distance = [list(x) for x in data["distance"]]
        self.set_swap_baseline(data["proposed"], data["accepted"])

    def rank(
        self,
        window_devices,
        device_capacity,
        window_samples,
        cycles_left,
        cold_rate,
        proposed,
        accepted,
        hot,
        votes,
    ):
        """
        Rank the windows by the uncertainty they contribute to the free energy
        and choose the hot windows.

        Each pair of neighbouring windows is weighted by sigma * sqrt(g) of its
        reduced energy differences, which sets the uncertainty of its free
        energy difference for a fixed amount of sampling. Pairs with signs of
        poor sampling are always hot, as are windows with fewer than the
        minimum number of independent samples of either the energy differences
        or, for a perturbation that breaks or makes a single bond, the bond
        distance. The remaining pairs are added in order of weight, with the
        windows either side of each so that exchange carries configurations
        into and out of them. With a context per replica, each set this gives
        is filled with the cold windows owning the most variance, until every
        GPU runs as many full batches as the busiest needs, since those windows
        take no extra time. A GPU's batches hold as many replicas as it can
        propagate at once. With shared contexts the set isn't filled, since an
        added window can need another pass. The set is then scored by the
        variance of the final free energy, given the samples each window has
        and will collect in the cycles left, multiplied by the compute of the
        whole run. The set with the lowest score is chosen. A window only stops
        being hot once it has been found cold at two consecutive rankings.
        Every window is hot if no set, including the windows kept hot, is
        better than that.

        The ranking uses all of the data since the burn-in, since the slowest
        windows decorrelate over far longer than a checkpoint interval.

        Parameters
        ----------

        window_devices: list of int
            The GPU that propagates each window.

        device_capacity: list of int
            The number of replicas each GPU can propagate at once.

        window_samples: list of int
            The number of energy samples each window has collected.

        cycles_left: int
            The number of cycles left in the run.

        cold_rate: float
            The share of the cycles left that propagate every window, and so
            the cold ones.

        proposed: np.ndarray
            The number of swaps proposed between each pair of windows so far.

        accepted: np.ndarray
            The number of swaps accepted between each pair of windows so far.

        hot: list of int
            The current hot windows, or None if every window is hot.

        votes: list of int
            The number of consecutive rankings at which each window has been
            found cold.

        Returns
        -------

        hot: list of int
            The hot windows, or None if every window is hot.

        votes: list of int
            The updated number of consecutive rankings at which each window
            has been found cold.
        """
        votes = list(votes)

        # There is nothing left to allocate, so keep the current hot windows.
        if cycles_left <= 0:
            _logger.info("Not ranking windows: no cycles left, keeping hot replicas")
            return hot, votes

        num_windows = self._num_windows
        lower = [_np.array(x) for x in self.lower]
        upper = [_np.array(x) for x in self.upper]

        min_samples = self._min_samples

        # Every window needs enough samples to estimate its fluctuations.
        num_samples = min(len(x) for x in lower[1:] + upper[:-1])
        if num_samples < min_samples:
            _logger.info(f"Not ranking windows: {num_samples} samples of {min_samples}")
            return hot, votes

        g_lower = [statistical_inefficiency(x) for x in lower]
        g_upper = [statistical_inefficiency(x) for x in upper]

        # Windows with too few independent samples can't be made cold.
        undersampled = []
        for k in range(num_windows):
            series = []
            if k > 0:
                series.append((lower[k], g_lower[k]))
            if k < num_windows - 1:
                series.append((upper[k], g_upper[k]))
            if min(len(x) / g for x, g in series) < min_samples:
                undersampled.append(k)

        # Nor can windows with a slow bond distance, which can bias the free
        # energy without showing in the energy differences.
        slow_distance = []
        if self._is_ring_break:
            for k, x in enumerate(self.distance):
                x = _np.array(x)
                if len(x) / statistical_inefficiency(x) < min_samples:
                    slow_distance.append(k)

        proposed = proposed - self.proposed
        accepted = accepted - self.accepted

        # The variance per sample of each pair's free energy, from the samples
        # in its lower (forward) and upper (reverse) windows.
        fwd_variance = _np.zeros(num_windows - 1)
        rev_variance = _np.zeros(num_windows - 1)
        flagged = []
        for k in range(num_windows - 1):
            fwd = upper[k]
            rev = lower[k + 1]
            fwd_variance[k] = fwd.var() * g_upper[k]
            rev_variance[k] = rev.var() * g_lower[k + 1]

            # Flag pairs that rarely exchange, or whose estimates disagree.
            num_proposed = proposed[k, k + 1]
            if (
                num_proposed > 0 and accepted[k, k + 1] / num_proposed < 0.05
            ) or is_unconverged(fwd, rev, g_upper[k], g_lower[k + 1]):
                flagged.append(k)

        weights = _np.sqrt(fwd_variance + rev_variance)

        forced = set(undersampled) | set(slow_distance)
        for k in flagged:
            forced.update((k, k + 1))

        def pad(windows):
            return windows | {
                r + d for r in windows for d in (-1, 1) if 0 <= r + d < num_windows
            }

        all_hot = set(range(num_windows))

        # The variance owned by each window: the forward side of the pair above
        # it and the reverse side of the pair below it.
        owned = _np.zeros(num_windows)
        owned[:-1] += fwd_variance
        owned[1:] += rev_variance

        def fill(windows):
            # Make the cold windows owning the most variance hot, until each GPU
            # fills as many batches as the busiest needs. With shared contexts
            # an added window can need another pass, so it isn't free.
            if self._is_cached:
                return set(windows)
            counts = {d: 0 for d in window_devices}
            for r in windows:
                counts[window_devices[r]] += 1
            batches = max(_ceil(n / device_capacity[d]) for d, n in counts.items())
            filled = set(windows)
            for r in sorted(all_hot - windows, key=lambda r: -owned[r]):
                d = window_devices[r]
                if counts[d] < batches * device_capacity[d]:
                    filled.add(r)
                    counts[d] += 1
            return filled

        total_variance = owned.sum()
        samples = _np.maximum(_np.asarray(window_samples, dtype=float), 1.0)

        def rates(windows):
            # The samples per cycle of each window for the rest of the run.
            is_hot = _np.zeros(num_windows, dtype=bool)
            is_hot[list(windows)] = True
            return _np.where(is_hot, 1.0, cold_rate)

        def score(windows):
            # The compute of the whole run, in blocks of dynamics, and the
            # variance of the final free energy, from the samples at its end.
            rate = rates(windows)
            final = samples + rate * cycles_left
            cost = samples.sum() + rate.sum() * cycles_left
            variance = (fwd_variance / final[:-1] + rev_variance / final[1:]).sum()
            return cost, variance

        # A set must beat all hot by more than rounding error.
        all_hot_cost, all_hot_variance = score(all_hot)
        threshold = all_hot_cost * all_hot_variance * (1 - 1e-9)

        # Add pairs in order of weight, keeping the set with the lowest final
        # variance multiplied by the compute of the whole run, if better than
        # all hot. Sets are scored with the windows that will fill the workers.
        candidate = set(forced)
        chosen = None
        best = threshold
        order = list(_np.argsort(-weights)) if total_variance > 0 else []
        for k in [None] + order:
            if k is not None:
                if k in candidate and k + 1 in candidate:
                    continue
                candidate.update((int(k), int(k) + 1))
            windows = pad(candidate)
            if not windows:
                continue
            efficiency = _np.prod(score(fill(windows)))
            if efficiency < best:
                chosen, best = windows, efficiency

        if chosen is None:
            new_hot = all_hot
            votes = [0] * num_windows
        else:
            # Keep windows hot until they have been found cold twice in a row.
            previous = all_hot if hot is None else set(hot)
            new_hot = set(chosen)
            for r in range(num_windows):
                if r in chosen:
                    votes[r] = 0
                elif r in previous:
                    votes[r] += 1
                    if votes[r] < 2:
                        new_hot.add(r)

            # Windows kept hot can leave no gain over all hot.
            new_hot = fill(new_hot)
            if _np.prod(score(new_hot)) >= threshold:
                new_hot = all_hot

        hot_variance = score(new_hot)[1]
        new_hot = sorted(new_hot)

        _logger.info(
            "Window pair weights (sigma * sqrt(g)): "
            + ", ".join(f"{w:.3f}" for w in weights)
        )
        if flagged:
            _logger.info(f"Unconverged window pairs: {flagged}")
        if undersampled:
            _logger.info(f"Undersampled windows: {undersampled}")
        if slow_distance:
            _logger.info(f"Windows with a slow bond distance: {slow_distance}")
        _logger.info(
            f"Hot replicas: {new_hot}"
            if len(new_hot) < num_windows
            else "All replicas hot"
        )
        # The compute per cycle from now on and the final variance, relative to
        # every window being hot for the rest of the run.
        cost = rates(new_hot).sum() / num_windows
        variance = hot_variance / all_hot_variance if all_hot_variance > 0 else 1.0
        _logger.info(
            f"Estimated cost per cycle {cost:.2f} and final variance "
            f"{variance:.2f} relative to every window being hot"
        )

        return (None if len(new_hot) == num_windows else new_hot), votes
