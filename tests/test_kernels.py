"""Brute-force reference tests for the pure kernels (no xarray involved)."""
import numpy as np
import pytest

from dominosee import _kernels as k


"""
Reference implementations: direct transcriptions of the definitions,
written for clarity instead of speed.
"""


def window_ref(ts, delt, sym, tau, kind):
    """Event-window reference by explicit loops (no convolution, no roll)."""
    T = len(ts)
    out = np.zeros(T, dtype=bool)
    for t in range(T):
        if kind == "forward":
            u = t - tau  # roll(+tau)
            if u < 0:
                continue
            lo, hi = u - delt, u + (delt if sym else 0)
        else:
            u = t + tau  # roll(-tau)
            if u >= T:
                continue
            lo, hi = u - (delt if sym else 0), u + delt
        lo, hi = max(lo, 0), min(hi, T - 1)
        out[t] = ts[lo : hi + 1].any()
    return out


def event_sync_ref(times_a, times_b, tm):
    """ES count over interior events (first and last of each series excluded)."""
    count = 0
    for ix in range(1, len(times_a) - 1):
        for iy in range(1, len(times_b) - 1):
            tau_x = min(times_a[ix] - times_a[ix - 1], times_a[ix + 1] - times_a[ix])
            tau_y = min(times_b[iy] - times_b[iy - 1], times_b[iy + 1] - times_b[iy])
            tau = min(tau_x, tau_y) / 2.0
            dist = abs(times_a[ix] - times_b[iy])
            if dist < tau and dist < tm:
                count += 1
    return count


def positions_arrays(series_list):
    """Build (positions, diffs, counts) arrays as es.get_event_positions does."""
    counts = np.array([s.sum() for s in series_list])
    max_events = counts.max()
    pos = np.full((len(series_list), max_events), -1, dtype=np.int32)
    diffs = np.full((len(series_list), max_events), np.nan)
    for i, s in enumerate(series_list):
        t = np.flatnonzero(s)
        pos[i, : len(t)] = t
        if len(t) > 1:
            diffs[i, : len(t) - 1] = np.diff(t)
    return pos, diffs, counts


@pytest.fixture
def rng():
    return np.random.default_rng(7)


"""
Windows
"""


@pytest.mark.parametrize("sym", [True, False])
@pytest.mark.parametrize("tau", [0, 1, 3])
@pytest.mark.parametrize("delt", [0, 2])
def test_windows_match_reference(rng, sym, tau, delt):
    ts = rng.random(60) < 0.15
    for kind, func in [("forward", k.forward_window), ("backward", k.backward_window)]:
        got = func(ts, delt=delt, sym=sym, tau=tau)
        want = window_ref(ts, delt, sym, tau, kind)
        np.testing.assert_array_equal(got, want, err_msg=f"{kind} delt={delt} sym={sym} tau={tau}")


def test_window_single_event():
    ts = np.zeros(20, dtype=bool)
    ts[10] = True
    fw = k.forward_window(ts, delt=2, sym=False, tau=0)
    np.testing.assert_array_equal(np.flatnonzero(fw), [10, 11, 12])
    bw = k.backward_window(ts, delt=2, sym=False, tau=0)
    np.testing.assert_array_equal(np.flatnonzero(bw), [8, 9, 10])


"""
Event Synchronization
"""


def test_event_sync_matches_reference(rng):
    series = [rng.random(80) < p for p in (0.10, 0.15, 0.20, 0.08)]
    pos, diffs, counts = positions_arrays(series)
    tm = 10

    got = k.event_sync(pos, diffs, counts, pos, diffs, counts, tm)

    for i, si in enumerate(series):
        for j, sj in enumerate(series):
            want = event_sync_ref(np.flatnonzero(si), np.flatnonzero(sj), tm)
            assert got[i, j] == want, f"pair ({i},{j})"


def test_event_sync_compiled_equals_python(rng):
    series = [rng.random(60) < 0.15 for _ in range(3)]
    pos, diffs, counts = positions_arrays(series)
    plain = k.event_sync(pos, diffs, counts, pos, diffs, counts, 8)
    jitted = k.compiled(k.event_sync, parallel=False)(pos, diffs, counts, pos, diffs, counts, 8)
    np.testing.assert_array_equal(plain, jitted)


def test_event_sync_needs_three_events():
    s_few = np.zeros(30, dtype=bool)
    s_few[[3, 10]] = True  # only 2 events
    s_many = np.zeros(30, dtype=bool)
    s_many[[3, 10, 17, 24]] = True
    pos, diffs, counts = positions_arrays([s_few, s_many])
    got = k.event_sync(pos, diffs, counts, pos, diffs, counts, 30)
    assert got[0, :].sum() == 0 and got[:, 0].sum() == 0
    assert got[1, 1] > 0


def test_event_sync_null_shape_and_guard():
    times = np.arange(50)
    out = k.event_sync_null(times, 5, 5, 10, samples=100)
    assert out.shape == (100,)
    assert (k.event_sync_null(times, 2, 5, 10, samples=10) == 0).all()


"""
ECA pairwise counts: orientation is (i, j) = (A row, B row)
"""


def test_eca_counts_match_reference(rng):
    delt, sym, tau = 2, True, 0
    events_a = np.stack([rng.random(50) < 0.2 for _ in range(2)])  # n_a = 2
    events_b = np.stack([rng.random(50) < 0.2 for _ in range(3)])  # n_b = 3

    window_a = np.stack([k.forward_window(s, delt, sym, tau) for s in events_a])
    window_b = np.stack([k.backward_window(s, delt, sym, tau) for s in events_b])

    prec = k.eca_precursor_counts(window_a, events_b)
    trig = k.eca_trigger_counts(events_a, window_b)

    assert prec.shape == (2, 3) and trig.shape == (2, 3)
    for i in range(2):
        for j in range(3):
            assert prec[i, j] == np.sum(window_a[i] & events_b[j]), f"precursor ({i},{j})"
            assert trig[i, j] == np.sum(events_a[i] & window_b[j]), f"trigger ({i},{j})"


def test_eca_orientation_directed_pair():
    """A at t=10 preceding B at t=12: edge A->B must appear at [i=A, j=B]."""
    T = 30
    a = np.zeros(T, dtype=bool)
    b = np.zeros(T, dtype=bool)
    a[10] = True
    b[12] = True
    quiet = np.zeros(T, dtype=bool)

    events_a = np.stack([a, quiet])
    events_b = np.stack([quiet, b])
    window_a = np.stack([k.forward_window(s, delt=2, sym=False, tau=0) for s in events_a])
    window_b = np.stack([k.backward_window(s, delt=2, sym=False, tau=0) for s in events_b])

    prec = k.eca_precursor_counts(window_a, events_b)
    trig = k.eca_trigger_counts(events_a, window_b)

    # Only the (A row 0 -> B row 1) entry may fire
    assert prec[0, 1] == 1 and prec.sum() == 1
    assert trig[0, 1] == 1 and trig.sum() == 1


def test_eca_counts_compiled_equals_python(rng):
    events_a = np.stack([rng.random(40) < 0.25 for _ in range(3)])
    events_b = np.stack([rng.random(40) < 0.25 for _ in range(3)])
    window_a = np.stack([k.forward_window(s, 2, True, 0) for s in events_a])
    plain = k.eca_precursor_counts(window_a, events_b)
    jitted = k.compiled(k.eca_precursor_counts, parallel=False)(window_a, events_b)
    np.testing.assert_array_equal(plain, jitted)
