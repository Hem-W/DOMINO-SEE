"""End-to-end tests for the ES pipeline through the pairwise engine."""
import numpy as np
import pytest
import xarray as xr

from dominosee import _kernels as k
from dominosee import conventions as cv
from dominosee import to_node_format
from dominosee.es import (
    convert_null_model_for_locations,
    create_null_model_from_indices,
    get_event_positions,
    get_event_sync_from_positions,
    get_event_time_differences,
)
from dominosee.network import get_link_from_critical_values


@pytest.fixture
def events_nodes():
    rng = np.random.default_rng(11)
    da = xr.DataArray(
        rng.normal(size=(2, 3, 120)),
        dims=("lat", "lon", "time"),
        coords={
            "lat": [10.0, 20.0],
            "lon": [100.0, 110.0, 120.0],
            "time": xr.date_range("2000-01-01", periods=120, freq="D"),
        },
        name="x",
    )
    events = (da < -0.8).assign_coords(layer="drought")
    return to_node_format(events)


def test_get_event_positions_structure(events_nodes):
    ds = get_event_positions(events_nodes)
    assert ds["event_positions"].dims == (cv.NODE, cv.EVENT)
    assert ds["event_count"].dims == (cv.NODE,)
    counts = ds["event_count"].values
    np.testing.assert_array_equal(counts, events_nodes.values.sum(axis=events_nodes.dims.index("time")))
    assert ds["event_positions"].dtype == np.int32


def test_get_event_positions_rejects_grid_input():
    da = xr.DataArray(
        np.zeros((2, 2, 10), dtype=bool),
        dims=("lat", "lon", "time"),
        coords={"time": xr.date_range("2000-01-01", periods=10, freq="D")},
    )
    with pytest.raises(ValueError, match="to_node_format"):
        get_event_positions(da)


def test_event_sync_matches_kernel_and_is_symmetric(events_nodes):
    ds = get_event_positions(events_nodes)
    positions = ds["event_positions"]
    es = get_event_sync_from_positions(positions, positions, tm=10)

    assert es.dims == (cv.NODE_I, cv.NODE_J)
    assert "lat_i" in es.coords and "lon_j" in es.coords
    assert es[cv.LAYER_I].item() == "drought" and es[cv.LAYER_J].item() == "drought"

    diffs = get_event_time_differences(positions)
    counts = ds["event_count"]
    want = k.event_sync(
        positions.values, diffs.values, counts.values,
        positions.values, diffs.values, counts.values,
        10,
    )
    np.testing.assert_array_equal(es.values, want)

    # ES between a node set and itself is symmetric
    np.testing.assert_array_equal(es.values, es.values.T)


def test_event_sync_serializes(events_nodes, tmp_path):
    ds = get_event_positions(events_nodes)
    es = get_event_sync_from_positions(ds["event_positions"], ds["event_positions"], tm=10)
    es = es.rename("event_sync")
    es.to_netcdf(tmp_path / "es.nc")
    zarr = pytest.importorskip("zarr")  # noqa: F841
    es.to_zarr(tmp_path / "es.zarr")


def test_null_model_to_links(events_nodes):
    ds = get_event_positions(events_nodes)
    es = get_event_sync_from_positions(ds["event_positions"], ds["event_positions"], tm=10)

    max_events = int(ds["event_count"].max().values)
    cvals = create_null_model_from_indices(
        events_nodes["time"], tm=10, max_events=max_events, significances=0.05, samples=200
    )
    assert set(cvals.dims) == {"noeA", "noeB", "significance"}

    null = convert_null_model_for_locations(cvals, ds["event_count"], ds["event_count"], sig=0.05)
    assert set(null.dims) == {cv.NODE_I, cv.NODE_J}

    links = get_link_from_critical_values(es, null, rule="greater", directed=False)
    assert links.dtype == bool
    assert links.dims == (cv.NODE_I, cv.NODE_J)
    assert links.attrs["directed"] is False
