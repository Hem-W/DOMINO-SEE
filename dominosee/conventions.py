"""Naming conventions and data-model utilities for DOMINO-SEE.

This module is the single implementation point of the conventions defined in
ARCHITECTURE.md:

- Dimension names encode *role only*: every network has dims ``node_i x node_j``.
- Identity lives in coordinates: integer ``node`` index with auxiliary
  ``lat``/``lon`` coordinates; layer identity in ``layer`` (per event series)
  and ``layer_i``/``layer_j`` (per network) coordinates.
- One suffix rule: ``_i``/``_j`` applies to every sided entity
  (``node_i``, ``lat_i``, ``lon_i``, ``layer_i``).
"""
import datetime
import json

import numpy as np
import xarray as xr

__all__ = [
    "TIME",
    "EVENT",
    "NODE",
    "NODE_I",
    "NODE_J",
    "LAYER",
    "LAYER_I",
    "LAYER_J",
    "to_node_format",
    "from_node_format",
    "as_pair",
    "from_pair",
    "transpose_network",
    "is_intralayer",
    "require_node_format",
    "require_event_series",
    "require_pairwise",
    "record_provenance",
]

TIME = "time"
EVENT = "event"
NODE = "node"
LAYER = "layer"

SIDES = ("i", "j")
NODE_I = "node_i"
NODE_J = "node_j"
LAYER_I = "layer_i"
LAYER_J = "layer_j"

#: attrs key recording how the node dimension was built, for round-tripping
NODE_META_ATTR = "node_stacked_dims"

_SPATIAL_DIM_GUESSES = (("lat", "lon"), ("latitude", "longitude"))


def _side_suffix(side):
    if side not in SIDES:
        raise ValueError(f"side must be one of {SIDES}, got {side!r}")
    return f"_{side}"


"""
Node format: integer node index + auxiliary coordinates
"""


def to_node_format(obj, spatial_dims=None):
    """Flatten spatial dimensions into an integer ``node`` dimension.

    The result carries the original spatial coordinates (e.g. ``lat``, ``lon``)
    as auxiliary coordinates on ``node`` instead of a ``MultiIndex``, so it can
    be written to netCDF/zarr directly. Data already in node format is returned
    unchanged.

    Parameters
    ----------
    obj : xr.DataArray or xr.Dataset
        Object with spatial dimensions to flatten.
    spatial_dims : sequence of str, optional
        Dimensions to flatten. If None, ``("lat", "lon")`` or
        ``("latitude", "longitude")`` is detected automatically.

    Returns
    -------
    xr.DataArray or xr.Dataset
        Object with the spatial dimensions replaced by ``node``.
    """
    if NODE in obj.dims:
        return obj

    if spatial_dims is None:
        for guess in _SPATIAL_DIM_GUESSES:
            if all(dim in obj.dims for dim in guess):
                spatial_dims = guess
                break
        else:
            raise ValueError(
                "Could not detect spatial dimensions: pass spatial_dims "
                f"explicitly (available dims: {tuple(obj.dims)})"
            )
    else:
        spatial_dims = tuple(spatial_dims)
        missing = [dim for dim in spatial_dims if dim not in obj.dims]
        if missing:
            raise ValueError(f"spatial_dims {missing} not found in dims {tuple(obj.dims)}")

    meta = {"dims": list(spatial_dims), "sizes": {d: int(obj.sizes[d]) for d in spatial_dims}}

    stacked = obj.stack({NODE: spatial_dims})
    stacked = stacked.reset_index(NODE)  # drop the MultiIndex, keep aux coords
    stacked = stacked.assign_coords({NODE: np.arange(stacked.sizes[NODE])})
    stacked.attrs[NODE_META_ATTR] = json.dumps(meta)
    return stacked


def from_node_format(obj, spatial_dims=None):
    """Restore the original spatial dimensions from node format.

    Parameters
    ----------
    obj : xr.DataArray or xr.Dataset
        Object in node format (as produced by :func:`to_node_format`).
    spatial_dims : sequence of str, optional
        Spatial dimensions to restore. If None, read from the metadata that
        :func:`to_node_format` recorded in ``attrs``.

    Returns
    -------
    xr.DataArray or xr.Dataset
        Object with ``node`` unstacked back into the spatial dimensions.
    """
    if NODE not in obj.dims:
        raise ValueError("from_node_format: input has no 'node' dimension")

    if spatial_dims is None:
        meta_raw = obj.attrs.get(NODE_META_ATTR)
        if meta_raw is None:
            raise ValueError(
                "from_node_format: no stacking metadata in attrs; "
                "pass spatial_dims explicitly"
            )
        spatial_dims = json.loads(meta_raw)["dims"]
    spatial_dims = list(spatial_dims)

    unstacked = obj.set_index({NODE: spatial_dims}).unstack(NODE)
    unstacked.attrs.pop(NODE_META_ATTR, None)
    return unstacked


"""
Pair dimensions: node_i / node_j and sided coordinates
"""


def as_pair(obj, side):
    """Rename ``node`` (and everything riding on it) to one side of a pair.

    Renames the ``node`` dimension to ``node_i``/``node_j`` and applies the same
    ``_i``/``_j`` suffix to every coordinate on that dimension (``lat`` ->
    ``lat_i``, ...) as well as to a scalar ``layer`` coordinate if present.

    Parameters
    ----------
    obj : xr.DataArray or xr.Dataset
        Object in node format.
    side : {"i", "j"}
        Which side of the pair this object becomes.
    """
    suffix = _side_suffix(side)
    if NODE not in obj.dims:
        raise ValueError("as_pair: input has no 'node' dimension")

    rename = {NODE: f"{NODE}{suffix}"}
    for name in obj.coords:
        if name == NODE:
            continue
        if NODE in obj[name].dims or name == LAYER:
            rename[name] = f"{name}{suffix}"
    return obj.rename(rename)


def from_pair(obj, side):
    """Inverse of :func:`as_pair`: strip the ``_i``/``_j`` suffix of one side."""
    suffix = _side_suffix(side)
    node_side = f"{NODE}{suffix}"
    if node_side not in obj.dims:
        raise ValueError(f"from_pair: input has no '{node_side}' dimension")

    rename = {node_side: NODE}
    for name in obj.coords:
        if name == node_side or not name.endswith(suffix):
            continue
        if node_side in obj[name].dims or name == f"{LAYER}{suffix}":
            rename[name] = name[: -len(suffix)]
    return obj.rename(rename)


def transpose_network(da):
    """Swap the i and j sides of a network, including all sided coordinates.

    ``transpose_network(net).sel(node_i=a, node_j=b) == net.sel(node_i=b, node_j=a)``
    """
    rename = {}
    for name in set(da.dims) | set(da.coords):
        if name.endswith("_i"):
            rename[name] = f"{name[:-2]}_j"
        elif name.endswith("_j"):
            rename[name] = f"{name[:-2]}_i"
    return da.rename(rename)


def is_intralayer(da):
    """Whether a network connects a layer to itself (``layer_i == layer_j``).

    Networks without any layer coordinates are treated as single-layer and
    return True. Non-scalar layer coordinates (supra-adjacency form) are not
    supported here.
    """
    has_i = LAYER_I in da.coords
    has_j = LAYER_J in da.coords
    if not has_i and not has_j:
        return True
    if has_i != has_j:
        raise ValueError(
            "is_intralayer: only one of layer_i/layer_j present; "
            "the network's layer coordinates are inconsistent"
        )
    layer_i, layer_j = da[LAYER_I], da[LAYER_J]
    if layer_i.ndim != 0 or layer_j.ndim != 0:
        raise ValueError(
            "is_intralayer: layer_i/layer_j are not scalar; for supra-adjacency "
            "arrays select a single layer pair first"
        )
    return bool((layer_i == layer_j).item())


"""
Pipeline-stage contracts
"""


def require_node_format(obj, name="input", core_dims=()):
    """Validate that ``obj`` is in node format with the expected core dims.

    Raises a ValueError early instead of letting un-normalized input (e.g. raw
    ``(lat, lon, time)`` arrays) silently reach the pairwise engine.
    """
    if NODE not in obj.dims:
        raise ValueError(
            f"{name}: expected node format with a '{NODE}' dimension, got dims "
            f"{tuple(obj.dims)}. Flatten spatial dimensions first with "
            "dominosee.to_node_format()."
        )
    expected = {NODE, *core_dims}
    extra = set(obj.dims) - expected
    if extra:
        raise ValueError(
            f"{name}: unexpected dimensions {sorted(extra)}; expected exactly "
            f"{sorted(expected)}"
        )


def require_event_series(obj, name="input"):
    """Validate a boolean event series with dims (node, time)."""
    require_node_format(obj, name=name, core_dims=(TIME,))
    if obj.dtype != bool:
        raise ValueError(f"{name}: event series must be boolean, got {obj.dtype}")


def require_pairwise(da, name="input"):
    """Validate a pairwise (node_i, node_j) array."""
    missing = [d for d in (NODE_I, NODE_J) if d not in da.dims]
    if missing:
        raise ValueError(
            f"{name}: expected pairwise format with dims ('{NODE_I}', '{NODE_J}'), "
            f"missing {missing} (got dims {tuple(da.dims)})"
        )


"""
Provenance
"""


def record_provenance(obj, func_name, params):
    """Append a CF-style ``history`` entry recording a pipeline step."""
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    entry = f"{timestamp}: dominosee.{func_name}({json.dumps(params, default=str, sort_keys=True)})"
    history = obj.attrs.get("history")
    obj.attrs["history"] = f"{history}\n{entry}" if history else entry
    return obj
