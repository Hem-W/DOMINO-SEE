"""Tests for the conventions module: node format, pair dims, layers, contracts."""
import numpy as np
import pytest
import xarray as xr

from dominosee import conventions as cv


@pytest.fixture
def da_grid():
    """A small (lat, lon, time) DataArray."""
    rng = np.random.default_rng(0)
    return xr.DataArray(
        rng.normal(size=(3, 4, 5)),
        dims=("lat", "lon", "time"),
        coords={
            "lat": [10.0, 20.0, 30.0],
            "lon": [100.0, 110.0, 120.0, 130.0],
            "time": xr.date_range("2000-01-01", periods=5, freq="D"),
        },
        name="var",
    )


@pytest.fixture
def da_nodes(da_grid):
    return cv.to_node_format(da_grid)


def test_to_node_format_structure(da_nodes):
    assert cv.NODE in da_nodes.dims
    assert da_nodes.sizes[cv.NODE] == 12
    # auxiliary coordinates ride on node, integer index on node
    assert da_nodes["lat"].dims == (cv.NODE,)
    assert da_nodes["lon"].dims == (cv.NODE,)
    np.testing.assert_array_equal(da_nodes[cv.NODE].values, np.arange(12))
    # no MultiIndex left
    assert not isinstance(da_nodes.indexes.get(cv.NODE), object) or da_nodes.indexes[
        cv.NODE
    ].nlevels == 1


def test_to_node_format_serializable(da_nodes, tmp_path):
    path = tmp_path / "nodes.nc"
    da_nodes.to_netcdf(path)  # would raise NotImplementedError with a MultiIndex
    back = xr.open_dataarray(path)
    np.testing.assert_allclose(back.values, da_nodes.values)


def test_to_node_format_idempotent(da_nodes):
    again = cv.to_node_format(da_nodes)
    assert again is da_nodes


def test_to_node_format_explicit_dims(da_grid):
    renamed = da_grid.rename({"lat": "y", "lon": "x"})
    with pytest.raises(ValueError, match="spatial_dims"):
        cv.to_node_format(renamed)
    nodes = cv.to_node_format(renamed, spatial_dims=("y", "x"))
    assert nodes.sizes[cv.NODE] == 12


def test_node_format_round_trip(da_grid):
    back = cv.from_node_format(cv.to_node_format(da_grid))
    back = back.transpose(*da_grid.dims)
    np.testing.assert_allclose(back.values, da_grid.values)
    np.testing.assert_array_equal(back["lat"].values, da_grid["lat"].values)
    np.testing.assert_array_equal(back["lon"].values, da_grid["lon"].values)


def test_as_pair_renames_dims_coords_and_layer(da_nodes):
    da = da_nodes.assign_coords(layer="drought")
    left = cv.as_pair(da, "i")
    assert cv.NODE_I in left.dims
    assert "lat_i" in left.coords and "lon_i" in left.coords
    assert left["layer_i"].item() == "drought"
    assert cv.NODE not in left.dims and "lat" not in left.coords


def test_from_pair_inverse(da_nodes):
    da = da_nodes.assign_coords(layer="drought")
    back = cv.from_pair(cv.as_pair(da, "j"), "j")
    assert cv.NODE in back.dims
    assert "lat" in back.coords and back["layer"].item() == "drought"


def test_as_pair_invalid_side(da_nodes):
    with pytest.raises(ValueError, match="side"):
        cv.as_pair(da_nodes, "k")


def test_transpose_network(da_nodes):
    left = cv.as_pair(da_nodes.isel(time=0), "i")
    right = cv.as_pair(da_nodes.isel(time=0), "j")
    net = left * right  # arbitrary (node_i, node_j) array
    flipped = cv.transpose_network(net)
    a, b = 1, 7
    assert flipped.isel(node_i=a, node_j=b).item() == net.isel(node_i=b, node_j=a).item()
    assert flipped["lat_i"].size == net["lat_j"].size


def test_is_intralayer():
    da = xr.DataArray(np.zeros((2, 2)), dims=(cv.NODE_I, cv.NODE_J))
    assert cv.is_intralayer(da) is True  # no layer info: single implicit layer
    intra = da.assign_coords(layer_i="drought", layer_j="drought")
    assert cv.is_intralayer(intra) is True
    inter = da.assign_coords(layer_i="drought", layer_j="flood")
    assert cv.is_intralayer(inter) is False
    with pytest.raises(ValueError, match="inconsistent"):
        cv.is_intralayer(da.assign_coords(layer_i="drought"))


def test_require_node_format(da_grid, da_nodes):
    with pytest.raises(ValueError, match="to_node_format"):
        cv.require_node_format(da_grid, core_dims=("time",))
    cv.require_node_format(da_nodes, core_dims=("time",))  # passes
    with pytest.raises(ValueError, match="unexpected dimensions"):
        cv.require_node_format(da_nodes.expand_dims(member=2), core_dims=("time",))


def test_require_event_series(da_nodes):
    with pytest.raises(ValueError, match="boolean"):
        cv.require_event_series(da_nodes)
    cv.require_event_series(da_nodes > 0)


def test_require_pairwise():
    good = xr.DataArray(np.zeros((2, 2)), dims=(cv.NODE_I, cv.NODE_J))
    cv.require_pairwise(good)
    with pytest.raises(ValueError, match="pairwise"):
        cv.require_pairwise(good.rename({cv.NODE_J: cv.NODE}))


def test_record_provenance():
    da = xr.DataArray(np.zeros(3), dims=("time",))
    cv.record_provenance(da, "get_event", {"threshold": -1.0})
    cv.record_provenance(da, "get_event_positions", {})
    lines = da.attrs["history"].splitlines()
    assert len(lines) == 2
    assert "dominosee.get_event(" in lines[0]
