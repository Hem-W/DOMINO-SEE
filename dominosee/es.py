"""Event Synchronization (ES) networks.

Thin orchestration layer: the numerical work lives in
:mod:`dominosee._kernels` and the pairwise machinery in
:mod:`dominosee.engine`.
"""
from typing import Tuple, Union

import cftime
import numpy as np
import pandas as pd
import xarray as xr

from . import _kernels
from .conventions import (
    EVENT,
    NODE,
    TIME,
    as_pair,
    record_provenance,
    require_event_series,
    require_node_format,
)
from .engine import infer_count_dtype, pairwise_apply

__all__ = [
    "get_event_positions",
    "get_event_time_differences",
    "get_event_sync_from_positions",
    "create_null_model_from_indices",
    "convert_null_model_for_locations",
]


def _extract_event_positions(binary_series, time_indices, max_count):
    """Extract event positions (as time indices) with a fixed output size."""
    # TODO: use VLType for event timing extraction if xarray supports VLType
    positions = np.full(max_count, -1, dtype=np.int32)

    event_pos = np.flatnonzero(binary_series)
    time_pos = time_indices[event_pos[:max_count]]

    positions[: len(time_pos)] = time_pos
    return positions


def _DataArrayTime_to_timeindex(
    dt_index: xr.DataArray, reference_date: Union[pd.Timestamp, cftime.datetime], freq: str
):
    if freq == "D":
        time_indices = (dt_index - reference_date).dt.days.values
    elif freq == "W":
        time_indices = ((dt_index - reference_date).dt.days // 7).values
    elif freq == "M":
        # Convert timestamps to periods and calculate month difference
        time_indices = np.array(
            [
                (pd.Period(dt.values, freq="M").ordinal - pd.Period(reference_date.values, freq="M").ordinal)
                for dt in dt_index
            ]
        )
    else:
        # Default to days if frequency is not recognized
        time_indices = (dt_index - reference_date).dt.days.values
    return time_indices


def get_event_positions(da: xr.DataArray, reference_date=None, freq: str = None) -> xr.Dataset:
    """
    Extract event positions from a binary event series as time indices.

    Parameters
    ----------
    da : xr.DataArray
        Boolean event series in node format with dims ``(node, time)``.
    reference_date : pd.Timestamp, optional
        Reference date for time indexing, by default None (uses first time value)
    freq : str, optional
        Frequency for time indexing, by default None (inferred from da.time)

    Returns
    -------
    xr.Dataset
        Dataset with ``event_positions`` (node, event) and ``event_count`` (node).
    """
    require_event_series(da, name="get_event_positions input")

    # Get max possible events across all nodes
    event_counts = da.sum(dim=TIME)
    max_events = int(event_counts.max().values)

    # Infer frequency from time dimension
    if freq is None:
        freq = xr.infer_freq(da[TIME])
        if freq in ["MS", "ME"]:
            freq = "M"

    dt_index = da[TIME]
    if reference_date is None:
        reference_date = da[TIME][0]

    time_indices = _DataArrayTime_to_timeindex(dt_index, reference_date, freq)

    result = xr.apply_ufunc(
        _extract_event_positions,
        da,
        input_core_dims=[[TIME]],
        output_core_dims=[[EVENT]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.int32],
        dask_gufunc_kwargs={"output_sizes": {EVENT: max_events}},
        kwargs={"max_count": max_events, "time_indices": time_indices},
    )
    result = result.assign_coords({EVENT: np.arange(max_events)})

    ds = xr.Dataset({"event_positions": result, "event_count": event_counts})
    record_provenance(ds, "get_event_positions", {"freq": freq})
    return ds


def get_event_time_differences(da_positions: xr.DataArray, event_counts: xr.DataArray = None) -> xr.DataArray:
    """
    Calculate time differences between consecutive events for each node.

    Parameters
    ----------
    da_positions : xr.DataArray
        Event positions (time indices), as returned by :func:`get_event_positions`.
    event_counts : xr.DataArray, optional
        Number of events per node, by default None (calculated from da_positions)

    Returns
    -------
    xr.DataArray
        Time differences between consecutive events; positions without a
        following event carry NaN.
    """
    if event_counts is None:
        valid_events = da_positions >= 0
        event_counts = valid_events.sum(dim=EVENT)

    next_positions = da_positions.shift({EVENT: -1})
    time_diffs = next_positions - da_positions

    event_indices = xr.DataArray(np.arange(da_positions.sizes[EVENT]), dims=[EVENT])
    valid_diffs = (da_positions >= 0) & (next_positions >= 0) & (event_indices < (event_counts - 1))
    time_diffs = time_diffs.where(valid_diffs)

    time_diffs.attrs = {
        "long_name": "Event Time Differences",
        "units": "time steps",
        "description": "Time differences between consecutive events for each node (latter - previous)",
    }
    return time_diffs


def get_event_sync_from_positions(
    positionsA: xr.DataArray,
    positionsB: xr.DataArray,
    tm: int,
    diffsA: xr.DataArray = None,
    diffsB: xr.DataArray = None,
    event_countsA: xr.DataArray = None,
    event_countsB: xr.DataArray = None,
    parallel: bool = True,
) -> xr.DataArray:
    """
    Calculate Event Synchronization between two sets of event positions.

    Parameters
    ----------
    positionsA : xr.DataArray
        Event positions of side A (``node_i`` in the result), in node format.
    positionsB : xr.DataArray
        Event positions of side B (``node_j`` in the result), in node format.
    tm : int
        Maximum time interval for synchronization.
    diffsA, diffsB : xr.DataArray, optional
        Event time differences, by default computed from the positions.
    event_countsA, event_countsB : xr.DataArray, optional
        Number of events per node, by default computed from the positions.
    parallel : bool, optional
        Whether to use parallel processing, by default True

    Returns
    -------
    xr.DataArray
        Event synchronization counts with dims ``(node_i, node_j)``.
    """
    if diffsA is None:
        diffsA = get_event_time_differences(positionsA)
    if diffsB is None:
        diffsB = get_event_time_differences(positionsB)

    if event_countsA is None:
        event_countsA = (positionsA >= 0).sum(dim=EVENT)
    if event_countsB is None:
        event_countsB = (positionsB >= 0).sum(dim=EVENT)

    max_count = max(int(event_countsA.max().values), int(event_countsB.max().values))
    output_dtype = infer_count_dtype(max_count)

    es = pairwise_apply(
        _kernels.event_sync,
        left=[positionsA, diffsA, event_countsA],
        right=[positionsB, diffsB, event_countsB],
        left_core_dims=[(EVENT,), (EVENT,), ()],
        right_core_dims=[(EVENT,), (EVENT,), ()],
        output_dtype=output_dtype,
        kernel_kwargs={"tm": tm, "output_dtype": output_dtype},
        parallel=parallel,
    )

    es.attrs.update(
        {
            "long_name": "Event Synchronization",
            "units": "count",
            "description": "Number of synchronized events between node_i (side A) and node_j (side B)",
            "tau_max": tm,
            "output_dtype": str(output_dtype),
        }
    )
    record_provenance(es, "get_event_sync_from_positions", {"tm": tm})
    return es


"""
Null model for Event Synchronization
"""


def _diagonal_mirror(arr):
    farr = arr.copy()
    pos = np.where(arr)  # only applicable for int > 0
    farr[(pos[1], pos[0])] = arr[pos]
    return farr


def create_null_model_from_indices(
    da_timeIndex: xr.DataArray,
    tm: int,
    max_events: Union[int, Tuple[int, int], np.ndarray],
    significances: Union[float, list] = [0.05],
    samples: int = 2000,
    min_es: int = None,
    parallel: bool = True,
) -> xr.DataArray:
    """
    Create a null model for event synchronization from permutations of the
    actual time indices.

    Parameters
    ----------
    da_timeIndex : xr.DataArray
        Time coordinate of the event datasets.
    tm : int
        Maximum time interval for synchronization.
    max_events : int, tuple of two ints, or numpy.ndarray
        Maximum number of events per side to tabulate.
    significances : float or list, optional
        Significance levels to calculate critical values for.
    samples : int, optional
        Number of permutation samples, by default 2000.
    min_es : int, optional
        Minimum critical value, by default None.
    parallel : bool, optional
        Whether to use parallelization, by default True.

    Returns
    -------
    xr.DataArray
        Critical values with dims ``(noeA, noeB, significance)``.
    """
    import scipy.stats as st

    sigs = np.atleast_1d(significances)

    if isinstance(max_events, (int, np.integer)):
        max_events_A = max_events_B = int(max_events)
    elif isinstance(max_events, np.ndarray):
        if max_events.size == 1:
            max_events_A = max_events_B = int(max_events.item())
        elif max_events.size == 2:
            max_events_A, max_events_B = (int(v) for v in max_events)
        else:
            raise ValueError("If max_events is a numpy array, it must have one or two elements")
    elif isinstance(max_events, tuple) and len(max_events) == 2:
        max_events_A, max_events_B = max_events
    else:
        raise ValueError("max_events must be either an int or a tuple of two ints")

    critical_values = np.zeros((max_events_A + 1, max_events_B + 1, len(sigs)), dtype="int")
    freq = xr.infer_freq(da_timeIndex)
    if freq in ["MS", "ME"]:
        freq = "M"
    time_indice = _DataArrayTime_to_timeindex(da_timeIndex, da_timeIndex[TIME].values[0], freq)

    event_sync_null = _kernels.compiled(_kernels.event_sync_null, parallel=parallel)

    for i in range(3, max_events_A + 1):
        for j in range(3, min(i + 1, max_events_B + 1)):
            cor = event_sync_null(time_indice, i, j, tm, samples)
            critical_values[i, j, :] = st.scoreatpercentile(cor, 100 - sigs * 100)

    # Mirror the matrix for symmetry
    for nsig in range(len(sigs)):
        critical_values[:, :, nsig] = _diagonal_mirror(critical_values[:, :, nsig])

    if min_es is not None:
        critical_values[critical_values < min_es] = min_es

    da_critical_values = xr.DataArray(
        critical_values,
        dims=["noeA", "noeB", "significance"],
        coords={
            "noeA": np.arange(max_events_A + 1),
            "noeB": np.arange(max_events_B + 1),
            "significance": sigs,
        },
    )
    da_critical_values = da_critical_values.assign_attrs(
        {
            "description": "Event synchronization null model for pairs of number of events",
            "tau_max": tm,
            "max_events": max_events,
            "min_es": min_es,
        }
    )
    record_provenance(
        da_critical_values,
        "create_null_model_from_indices",
        {"tm": tm, "samples": samples, "significances": list(np.atleast_1d(significances))},
    )
    return da_critical_values


def convert_null_model_for_locations(
    da_critical_values: xr.DataArray,
    da_evN_locA: xr.DataArray,
    da_evN_locB: xr.DataArray,
    sig: float = None,
) -> xr.DataArray:
    """
    Expand the null model to a ``(node_i, node_j)`` matrix of critical values.

    Parameters
    ----------
    da_critical_values : xr.DataArray
        Critical values from :func:`create_null_model_from_indices`.
    da_evN_locA, da_evN_locB : xr.DataArray
        Event counts per node (node format) for each side.
    sig : float, optional
        Significance level to select; may be omitted when the null model holds
        a single level.
    """
    require_node_format(da_evN_locA, name="da_evN_locA")
    require_node_format(da_evN_locB, name="da_evN_locB")
    da_evN_locA = as_pair(da_evN_locA, "i")
    da_evN_locB = as_pair(da_evN_locB, "j")

    if sig is None:
        if "significance" in da_critical_values.dims and da_critical_values.sizes.get("significance", 0) == 1:
            sig = float(da_critical_values.coords["significance"].item())
        else:
            raise ValueError("sig must be specified")

    da_null = da_critical_values.sel(noeA=da_evN_locA, noeB=da_evN_locB, significance=sig)
    da_null = da_null.assign_attrs(
        {
            "description": "Event synchronization null model for pairs of nodes",
            "tau_max": da_critical_values.attrs["tau_max"],
            "max_events": da_critical_values.attrs["max_events"],
            "min_es": da_critical_values.attrs["min_es"],
        }
    )
    return da_null
