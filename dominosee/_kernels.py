"""Pure numerical kernels for pairwise coupling measures.

This module depends on numpy/numba only — no xarray. Every pairwise kernel
follows the same shape contract used by :func:`dominosee.engine.pairwise_apply`:
the left-side arrays are stacked as ``(n_i, ...)``, the right-side arrays as
``(n_j, ...)``, and the result is an ``(n_i, n_j)`` matrix whose ``[i, j]``
entry refers to the directed pair ``i -> j``.

Kernels are plain Python functions; compile them on demand with
:func:`compiled` (numba ``njit``, optionally parallel).
"""
import numpy as np
from numba import njit, prange

__all__ = [
    "compiled",
    "forward_window",
    "backward_window",
    "event_sync",
    "event_sync_null",
    "eca_precursor_counts",
    "eca_trigger_counts",
]

_COMPILE_CACHE = {}


def compiled(kernel, parallel=True):
    """Return the numba-compiled variant of a kernel (cached per kernel/mode)."""
    key = (kernel, bool(parallel))
    if key not in _COMPILE_CACHE:
        _COMPILE_CACHE[key] = njit(parallel=bool(parallel))(kernel)
    return _COMPILE_CACHE[key]


"""
Event Coincidence Analysis windows (pure numpy, single time series)
"""


def forward_window(time_series, delt=2, sym=True, tau=0):
    """Smear a binary event series over the ECA precursor window.

    The result is True at time ``t`` when the series has an event within the
    window around ``t`` (forward window of length ``delt``, symmetric when
    ``sym``), shifted by the delay ``tau``.
    """
    time_series = np.asarray(time_series)
    if time_series.ndim != 1:
        raise ValueError("time_series must be a 1D array")
    if tau < 0:
        raise ValueError("tau must be non-negative")
    if delt < 0:
        raise ValueError("delt must be non-negative")

    window = np.ones((1 + 1 * sym) * delt + 1)

    if delt == 0:
        result = time_series.astype(bool).copy()
    else:
        result = np.convolve(time_series, window)[sym * delt : -delt] >= 0.5

    if tau > 0:
        result = np.roll(result, tau)
        result[:tau] = False

    return result


def backward_window(time_series, delt=2, sym=True, tau=0):
    """Smear a binary event series over the ECA trigger window.

    Same as :func:`forward_window` when ``sym``; otherwise the window extends
    backward in time and the delay shifts in the opposite direction.
    """
    time_series = np.asarray(time_series)
    if time_series.ndim != 1:
        raise ValueError("time_series must be a 1D array")
    if tau < 0:
        raise ValueError("tau must be non-negative")
    if delt < 0:
        raise ValueError("delt must be non-negative")

    if sym:
        result = forward_window(time_series, delt, sym, 0)
    else:
        window = np.ones(delt + 1)
        if delt == 0:
            result = time_series.astype(bool).copy()
        else:
            result = np.convolve(time_series, window)[delt:] >= 0.5

    if tau > 0:
        result = np.roll(result, -tau)
        result[-tau:] = False

    return result


"""
Event Synchronization
"""


def event_sync(pos_a, diff_a, count_a, pos_b, diff_b, count_b, tm, output_dtype=np.uint8):
    """Count synchronized events for every pair of event-position rows.

    Parameters
    ----------
    pos_a, pos_b : (n, max_events) int arrays
        Event positions (time indices) per node, padded with -1.
    diff_a, diff_b : (n, max_events) arrays
        Time differences between consecutive events per node.
    count_a, count_b : (n,) int arrays
        Number of events per node.
    tm : int
        Maximum time interval for synchronization.
    output_dtype : numpy dtype
        Integer dtype of the output counts.

    Returns
    -------
    (n_a, n_b) array where ``[i, j]`` counts synchronizations between node i of
    side A and node j of side B.
    """
    nodes_a = pos_a.shape[0]
    nodes_b = pos_b.shape[0]
    es = np.zeros((nodes_a, nodes_b), dtype=output_dtype)

    for i in prange(nodes_a):
        if count_a[i] > 2:
            # interior events only: the first and last event of a series have no
            # complete inter-event gap pair, so they never participate
            ex = pos_a[i, 1 : count_a[i] - 1]
            ex_gapb = diff_a[i, 0 : count_a[i] - 2]
            ex_gapf = diff_a[i, 1 : count_a[i] - 1]
            ex_tau = np.minimum(ex_gapb, ex_gapf)

            for k in range(nodes_b):
                if count_b[k] > 2:
                    count = 0
                    ey = pos_b[k, 1 : count_b[k] - 1]
                    ey_gapb = diff_b[k, 0 : count_b[k] - 2]
                    ey_gapf = diff_b[k, 1 : count_b[k] - 1]
                    ey_tau = np.minimum(ey_gapb, ey_gapf)

                    for ix in range(len(ex)):
                        for iy in range(len(ey)):
                            dist = abs(ex[ix] - ey[iy])
                            tau = min(ex_tau[ix], ey_tau[iy]) / 2.0
                            if dist < tau and dist < tm:
                                count += 1

                    es[i, k] = count
        else:
            es[i, :] = 0
    return es


def event_sync_null(time_indice, noe_a, noe_b, tm, samples=2000):
    """Null distribution of event synchronization by permuting event times.

    Draws ``samples`` random placements of ``noe_a`` and ``noe_b`` events on the
    observed time indices and returns the synchronization count of each draw.
    """
    cor = np.zeros(samples, dtype="int")
    if noe_a < 3 or noe_b < 3 or len(time_indice) < 3:
        return cor

    for k in prange(samples):
        dat0_indices = np.random.choice(time_indice, size=noe_a, replace=False)
        dat1_indices = np.random.choice(time_indice, size=noe_b, replace=False)

        ex = dat0_indices[1:-1]
        ey = dat1_indices[1:-1]
        ex_diff = np.diff(dat0_indices)
        ey_diff = np.diff(dat1_indices)

        ex_tau = np.minimum(ex_diff[:-1], ex_diff[1:])
        ey_tau = np.minimum(ey_diff[:-1], ey_diff[1:])
        count = 0
        for ix in range(len(ex)):
            for iy in range(len(ey)):
                dist = abs(ex[ix] - ey[iy])
                if ix < len(ex_tau) and iy < len(ey_tau):
                    tau = min(ex_tau[ix], ey_tau[iy]) / 2.0
                    if dist < tau and dist < tm:
                        count += 1
        cor[k] = count

    return cor


"""
Event Coincidence Analysis pairwise counts

Definitions follow the validated legacy implementation (and the binomial
confidence formulas in dominosee.eca):

- precursor(i -> j): number of side-B events at node j that fall inside the
  (forward) coincidence window of side-A node i — counts B events, so the
  binomial confidence uses n = N_B.
- trigger(i -> j): number of side-A events at node i that fall inside the
  (backward) coincidence window of side-B node j — counts A events, so the
  binomial confidence uses n = N_A.
"""


def eca_precursor_counts(window_a, events_b, output_dtype=np.uint16):
    """Count precursor coincidences for every (i, j) pair.

    Parameters
    ----------
    window_a : (n_a, n_time) bool array
        Forward coincidence windows of side A (see :func:`forward_window`).
    events_b : (n_b, n_time) bool array
        Binary event series of side B.
    """
    n_a = window_a.shape[0]
    n_b = events_b.shape[0]
    result = np.zeros((n_a, n_b), dtype=output_dtype)

    for i in prange(n_a):
        for j in range(n_b):
            result[i, j] = np.sum(window_a[i, :] & events_b[j, :])
    return result


def eca_trigger_counts(events_a, window_b, output_dtype=np.uint16):
    """Count trigger coincidences for every (i, j) pair.

    Parameters
    ----------
    events_a : (n_a, n_time) bool array
        Binary event series of side A.
    window_b : (n_b, n_time) bool array
        Backward coincidence windows of side B (see :func:`backward_window`).
    """
    n_a = events_a.shape[0]
    n_b = window_b.shape[0]
    result = np.zeros((n_a, n_b), dtype=output_dtype)

    for i in prange(n_a):
        for j in range(n_b):
            result[i, j] = np.sum(events_a[i, :] & window_b[j, :])
    return result
