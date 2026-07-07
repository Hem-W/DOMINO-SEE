"""End-to-end tests for the ECA pipeline through the pairwise engine."""
import numpy as np
import pytest
import xarray as xr

from dominosee import _kernels as k
from dominosee import conventions as cv
from dominosee.eca import (
    get_eca_precursor,
    get_eca_precursor_confidence,
    get_eca_precursor_from_events,
    get_eca_precursor_window,
    get_eca_trigger_confidence,
    get_eca_trigger_from_events,
)


def make_events(rows, layer=None):
    rows = np.asarray(rows, dtype=bool)
    da = xr.DataArray(
        rows,
        dims=(cv.NODE, cv.TIME),
        coords={
            cv.NODE: np.arange(rows.shape[0]),
            cv.TIME: xr.date_range("2000-01-01", periods=rows.shape[1], freq="D"),
        },
        name="event",
    )
    if layer is not None:
        da = da.assign_coords({cv.LAYER: layer})
    return da


@pytest.fixture
def events_ab():
    """Two node sets of different sizes (catches transposed outputs)."""
    rng = np.random.default_rng(3)
    events_a = make_events(rng.random((3, 90)) < 0.2, layer="drought")
    events_b = make_events(rng.random((4, 90)) < 0.2, layer="flood")
    return events_a, events_b


def test_precursor_matches_kernel(events_ab):
    events_a, events_b = events_ab
    delt, sym, tau = 2, True, 0
    prec = get_eca_precursor_from_events(events_a, events_b, delt=delt, sym=sym, tau=tau)

    assert prec.dims == (cv.NODE_I, cv.NODE_J)
    assert prec.sizes == {cv.NODE_I: 3, cv.NODE_J: 4}
    assert prec[cv.LAYER_I].item() == "drought" and prec[cv.LAYER_J].item() == "flood"

    window_a = np.stack([k.forward_window(row, delt, sym, tau) for row in events_a.values])
    want = k.eca_precursor_counts(window_a, events_b.values)
    np.testing.assert_array_equal(prec.values, want)


def test_trigger_matches_kernel(events_ab):
    events_a, events_b = events_ab
    delt, sym, tau = 3, False, 1
    trig = get_eca_trigger_from_events(events_a, events_b, delt=delt, sym=sym, tau=tau)

    assert trig.sizes == {cv.NODE_I: 3, cv.NODE_J: 4}

    # the trigger window belongs to side B (the target of the directed pair)
    window_b = np.stack([k.backward_window(row, delt, sym, tau) for row in events_b.values])
    want = k.eca_trigger_counts(events_a.values, window_b)
    np.testing.assert_array_equal(trig.values, want)


def test_directed_orientation():
    """A event at t=10 (node 0) preceding a B event at t=12 (node 1)."""
    T = 30
    a = np.zeros((2, T), dtype=bool)
    b = np.zeros((2, T), dtype=bool)
    a[0, 10] = True
    b[1, 12] = True
    events_a, events_b = make_events(a), make_events(b)

    prec = get_eca_precursor_from_events(events_a, events_b, delt=2, sym=False, tau=0)
    trig = get_eca_trigger_from_events(events_a, events_b, delt=2, sym=False, tau=0)

    assert prec.sel(node_i=0, node_j=1).item() == 1 and prec.values.sum() == 1
    assert trig.sel(node_i=0, node_j=1).item() == 1 and trig.values.sum() == 1


def test_confidence_bounds_and_min_eventnum(events_ab):
    events_a, events_b = events_ab
    prec = get_eca_precursor_from_events(events_a, events_b, delt=2, sym=True, tau=0)
    trig = get_eca_trigger_from_events(events_a, events_b, delt=2, sym=True, tau=0)

    prec_conf = get_eca_precursor_confidence(prec, events_a, events_b)
    trig_conf = get_eca_trigger_confidence(trig, events_a, events_b)

    for conf in (prec_conf, trig_conf):
        assert conf.dims == (cv.NODE_I, cv.NODE_J)
        assert float(conf.min()) >= 0.0 and float(conf.max()) <= 1.0

    # a node without events on side A zeroes its row
    quiet = events_a.copy()
    quiet.values[0, :] = False
    prec_quiet = get_eca_precursor_from_events(quiet, events_b, delt=2, sym=True, tau=0)
    conf_quiet = get_eca_precursor_confidence(prec_quiet, quiet, events_b)
    assert float(conf_quiet.sel(node_i=0).max()) == 0.0


def test_explicit_params_and_missing_attrs(events_ab):
    events_a, events_b = events_ab
    window = get_eca_precursor_window(events_a, delt=2, sym=True, tau=0)

    stripped = window.copy()
    stripped.attrs = {}
    with pytest.raises(ValueError, match="delt/sym/tau"):
        get_eca_precursor(stripped, events_b)

    explicit = get_eca_precursor(stripped, events_b, delt=2, sym=True, tau=0)
    via_attrs = get_eca_precursor(window, events_b)
    np.testing.assert_array_equal(explicit.values, via_attrs.values)


def test_rejects_grid_input():
    da = xr.DataArray(
        np.zeros((2, 2, 10), dtype=bool),
        dims=("lat", "lon", "time"),
        coords={"time": xr.date_range("2000-01-01", periods=10, freq="D")},
    )
    with pytest.raises(ValueError, match="to_node_format"):
        get_eca_precursor_from_events(da, da)
