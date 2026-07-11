"""Pairwise computation engine.

This module is the single place where "for every pair of nodes, run a kernel"
is implemented: node-format validation, ``node_i``/``node_j`` pairing (with
layer propagation), ``apply_ufunc`` assembly, and output dtype inference all
live here. Coupling modules (:mod:`dominosee.es`, :mod:`dominosee.eca`) are
thin wrappers around :func:`pairwise_apply`; new coupling measures only need a
new kernel in :mod:`dominosee._kernels`.
"""
import numpy as np
import xarray as xr

from . import _kernels
from .conventions import NODE_I, NODE_J, as_pair, require_node_format

__all__ = ["infer_count_dtype", "pairwise_apply"]


def infer_count_dtype(max_value):
    """Smallest unsigned integer dtype that can hold ``max_value``."""
    for dtype in (np.uint8, np.uint16, np.uint32):
        if max_value <= np.iinfo(dtype).max:
            return dtype
    return np.uint64


def _as_list(obj):
    return [obj] if isinstance(obj, (xr.DataArray, xr.Dataset)) else list(obj)


def pairwise_apply(
    kernel,
    left,
    right,
    left_core_dims,
    right_core_dims,
    output_dtype,
    kernel_kwargs=None,
    parallel=True,
    dask="parallelized",
):
    """Apply a pairwise kernel over every (node_i, node_j) combination.

    Parameters
    ----------
    kernel : callable
        Pure kernel from :mod:`dominosee._kernels`. Its positional arguments
        must be the left-side arrays followed by the right-side arrays, each
        shaped ``(n_nodes, *core)``; it must return an ``(n_i, n_j)`` matrix.
    left, right : xr.DataArray or sequence of xr.DataArray
        Node-format inputs for each side of the pair.
    left_core_dims, right_core_dims : sequence of sequences of str
        Extra core dimensions (besides ``node``) of each input, e.g.
        ``[("event",), ("event",), ()]``.
    output_dtype : numpy dtype
        Dtype of the resulting matrix (see :func:`infer_count_dtype`).
    kernel_kwargs : dict, optional
        Keyword arguments forwarded to the kernel.
    parallel : bool, optional
        Whether the kernel is numba-compiled with ``parallel=True``.
    dask : str, optional
        Passed to ``xr.apply_ufunc``.

    Returns
    -------
    xr.DataArray
        ``(node_i, node_j)`` matrix carrying the sided coordinates of both
        inputs (``lat_i``, ``lon_i``, ``layer_i``, ... and the ``_j``
        counterparts).

    Notes
    -----
    TODO(Phase 3): exploit symmetry (upper triangle) for intra-layer undirected
    measures, and compute chunk pairs written into a zarr store via
    ``to_zarr(region=...)`` instead of materializing the full matrix.
    """
    left = _as_list(left)
    right = _as_list(right)
    left_core_dims = [tuple(dims) for dims in left_core_dims]
    right_core_dims = [tuple(dims) for dims in right_core_dims]
    if len(left) != len(left_core_dims) or len(right) != len(right_core_dims):
        raise ValueError("core dim specs must match the number of input arrays per side")

    for n, (arr, extras) in enumerate(zip(left, left_core_dims)):
        require_node_format(arr, name=f"left input {n}", core_dims=extras)
    for n, (arr, extras) in enumerate(zip(right, right_core_dims)):
        require_node_format(arr, name=f"right input {n}", core_dims=extras)

    left_p = [as_pair(arr, "i") for arr in left]
    right_p = [as_pair(arr, "j") for arr in right]

    input_core_dims = [[NODE_I, *extras] for extras in left_core_dims]
    input_core_dims += [[NODE_J, *extras] for extras in right_core_dims]

    func = _kernels.compiled(kernel, parallel=parallel)

    result = xr.apply_ufunc(
        func,
        *left_p,
        *right_p,
        input_core_dims=input_core_dims,
        output_core_dims=[[NODE_I, NODE_J]],
        dask=dask,
        dask_gufunc_kwargs={"allow_rechunk": True},
        output_dtypes=[output_dtype],
        kwargs=kernel_kwargs or {},
    )

    # apply_ufunc drops coordinates that live on core dimensions; restore the
    # sided identity coordinates (node/lat/lon/... and scalar layer) per side
    for src, side_dim in ((left_p[0], NODE_I), (right_p[0], NODE_J)):
        coords = {
            name: coord
            for name, coord in src.coords.items()
            if set(coord.dims) <= {side_dim}
        }
        result = result.assign_coords(coords)

    return result
