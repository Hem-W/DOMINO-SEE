"""Validate DOMINO-SEE ECA against the CoinCalc definitions.

Reference: Siegmund, Siegmund & Donner (2017), "CoinCalc - A new R package for
quantifying simultaneities of event series", Computers & Geosciences 98, 64-72.

Precursor/trigger coincidence rates (Eqs. 1-2) share one coincidence indicator
and differ only in which end is counted:

- precursor: fraction of events in one series that are matched by >=1 event in
  the other, counting the *response* series (binomial n = N of that series);
- trigger:   same indicator, counting the *driver* series.

DOMINO-SEE documents its direction as "A drives B" (A precedes B), the mirror of
the paper's "B influences A" convention. The reference below is written in
DOMINO-SEE's native convention so the mapping is explicit:

- ``get_eca_precursor_from_events(A, B)`` counts B events preceded by an A event
  (n = N_B);
- ``get_eca_trigger_from_events(A, B)`` counts A events followed by a B event
  (n = N_A).
"""
import numpy as np
import pytest
import xarray as xr
from scipy.stats import binom

from dominosee import to_node_format
from dominosee.eca import (
    get_eca_precursor_from_events,
    get_eca_trigger_from_events,
    get_eca_precursor_confidence,
    prec_confidence,
    trig_confidence,
)


# --------------------------------------------------------------------------- #
# Faithful reference implementation of Eqs. 1-2/6 (Theta-capped counts),
# expressed in DOMINO-SEE's "A drives B" convention.
# --------------------------------------------------------------------------- #
def _coincides(a, b, delt, tau, sym):
    """True if B-event at index ``b`` coincides with A-event at index ``a``.

    A drives B: B lies in [a+tau, a+tau+delt] (non-sym) or [a+tau-delt, a+tau+delt] (sym).
    """
    d = b - (a + tau)
    return (-delt <= d <= delt) if sym else (0 <= d <= delt)


def eca_reference(events_a, events_b, delt=2, tau=0, sym=False):
    """Return (K_precursor, K_trigger, N_A, N_B) per Eqs. 1-2, Theta-capped."""
    tA = np.flatnonzero(events_a)
    tB = np.flatnonzero(events_b)
    # precursor counts B events (response) matched by >=1 A event
    kp = sum(any(_coincides(a, b, delt, tau, sym) for a in tA) for b in tB)
    # trigger counts A events (driver) matched by >=1 B event
    kt = sum(any(_coincides(a, b, delt, tau, sym) for b in tB) for a in tA)
    return kp, kt, len(tA), len(tB)


def _series(rows):
    """Boolean (n_node, time) rows -> node-format DataArray."""
    rows = np.atleast_2d(np.asarray(rows, dtype=bool))
    da = xr.DataArray(
        rows,
        dims=("space", "time"),
        coords={"time": xr.date_range("2000-01-01", periods=rows.shape[1], freq="D")},
    )
    return to_node_format(da, spatial_dims=("space",))


def _pair(events_a, events_b, delt, sym, tau):
    A, B = _series(events_a), _series(events_b)
    prec = get_eca_precursor_from_events(A, B, delt=delt, sym=sym, tau=tau)
    trig = get_eca_trigger_from_events(A, B, delt=delt, sym=sym, tau=tau)
    return int(prec.isel(node_i=0, node_j=0)), int(trig.isel(node_i=0, node_j=0))


# --------------------------------------------------------------------------- #
# Paper worked example 1 (Lilac flowering A vs April temperature B)
# --------------------------------------------------------------------------- #
def test_paper_example1_symmetric_rates():
    """N_A = N_B = 6, 3 simultaneous coincidences, delT = tau = 0 -> rp = rt = 0.5."""
    A = np.zeros(30, dtype=bool)
    B = np.zeros(30, dtype=bool)
    A[[1, 5, 9, 13, 17, 21]] = True
    B[[1, 5, 9, 24, 26, 28]] = True  # 3 shared

    kp, kt, na, nb = eca_reference(A, B, delt=0, tau=0, sym=False)
    assert (kp, kt, na, nb) == (3, 3, 6, 6)
    assert kp / na == 0.5 and kt / nb == 0.5

    prec, trig = _pair(A, B, delt=0, sym=False, tau=0)
    assert prec == 3 and trig == 3


# --------------------------------------------------------------------------- #
# Discriminating case: precursor != trigger (the "four values" concern)
# --------------------------------------------------------------------------- #
def test_precursor_differs_from_trigger():
    """One A event with two following B events: precursor counts 2 B, trigger counts 1 A."""
    A = np.zeros(30, dtype=bool)
    B = np.zeros(30, dtype=bool)
    A[10] = True
    B[[11, 12]] = True

    kp, kt, na, nb = eca_reference(A, B, delt=2, tau=0, sym=False)
    assert (kp, kt) == (2, 1)  # precursor counts B events, trigger counts A events

    prec, trig = _pair(A, B, delt=2, sym=False, tau=0)
    assert (prec, trig) == (2, 1)


def test_lagged_directionality():
    """tau shifts the driver->response window forward; only A-before-B coincides."""
    A = np.zeros(60, dtype=bool)
    B = np.zeros(60, dtype=bool)
    A[[5, 25, 45]] = True
    B[[8, 28, 48]] = True  # each B is 3 after an A -> matches tau=1, delt=2

    prec, trig = _pair(A, B, delt=2, sym=False, tau=1)
    kp, kt, _, _ = eca_reference(A, B, delt=2, tau=1, sym=False)
    assert (prec, trig) == (kp, kt) == (3, 3)

    # reversed roles: B never precedes A within the window -> no coincidences
    prec_rev, trig_rev = _pair(B, A, delt=2, sym=False, tau=1)
    assert (prec_rev, trig_rev) == (0, 0)


# --------------------------------------------------------------------------- #
# DOMINO-SEE matches the reference over random inputs (both are Theta-capped)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("sym", [False, True])
@pytest.mark.parametrize("tau", [0, 1, 3])
@pytest.mark.parametrize("delt", [0, 2])
def test_matches_reference_random(sym, tau, delt):
    rng = np.random.default_rng(20240607 + tau + 10 * delt + 100 * sym)
    T = 80
    lo, hi = tau + delt + 1, T - tau - delt - 1  # keep events off the roll boundary
    for _ in range(15):
        A = np.zeros(T, dtype=bool)
        B = np.zeros(T, dtype=bool)
        A[rng.choice(np.arange(lo, hi), size=6, replace=False)] = True
        B[rng.choice(np.arange(lo, hi), size=6, replace=False)] = True

        prec, trig = _pair(A, B, delt=delt, sym=sym, tau=tau)
        kp, kt, _, _ = eca_reference(A, B, delt=delt, tau=tau, sym=sym)
        assert (prec, trig) == (kp, kt), f"delt={delt} sym={sym} tau={tau}"


# --------------------------------------------------------------------------- #
# Confidence: binomial n must match the counted series (paper Eq. 3)
# --------------------------------------------------------------------------- #
def test_confidence_binomial_n_pairing():
    """precursor confidence uses n = N_B (counted side); trigger uses n = N_A."""
    T, delt, sym, tau = 100, 2, False, 0
    TOL = delt * (1 + sym) + 1  # paper: dT+1 (non-sym ts), 2 dT+1 (sym ts)
    na, nb, kp, kt = 7, 5, 3, 2

    p_single_prec = 1 - (1 - TOL / (T - tau)) ** na
    assert prec_confidence(kp, na, nb, TOL, T, tau) == pytest.approx(
        binom.cdf(kp, n=nb, p=p_single_prec), rel=1e-5
    )
    p_single_trig = 1 - (1 - TOL / (T - tau)) ** nb
    assert trig_confidence(kt, na, nb, TOL, T, tau) == pytest.approx(
        binom.cdf(kt, n=na, p=p_single_trig), rel=1e-5
    )


def test_confidence_uses_paper_tolerance():
    """get_eca_*_confidence must use TOL = delt*(1+sym)+1, matching the paper."""
    T = 200
    A = np.zeros(T, dtype=bool)
    B = np.zeros(T, dtype=bool)
    A[[20, 60, 120]] = True
    B[[21, 61, 121]] = True
    Ada, Bda = _series(A), _series(B)

    for delt, sym in [(2, False), (3, True), (0, False)]:
        prec = get_eca_precursor_from_events(Ada, Bda, delt=delt, sym=sym, tau=0)
        conf = get_eca_precursor_confidence(prec, Ada, Bda, min_eventnum=0)

        TOL = delt * (1 + sym) + 1
        na = int(Ada.sum().item())  # 3 events per single node
        nb = int(Bda.sum().item())
        kp = int(prec.isel(node_i=0, node_j=0))
        expected = binom.cdf(kp, n=nb, p=1 - (1 - TOL / T) ** na)
        assert float(conf.isel(node_i=0, node_j=0)) == pytest.approx(expected, rel=1e-5), (
            f"delt={delt} sym={sym}"
        )
