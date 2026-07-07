"""Event Coincidence Analysis (ECA) networks.

Thin orchestration layer over :mod:`dominosee._kernels` and
:mod:`dominosee.engine`. ECA networks are directed (``node_i -> node_j``):

- ``precursor(i -> j)`` counts side-B events at node j inside the forward
  coincidence window of side-A node i (counts B events; binomial confidence
  with ``n = N_B``).
- ``trigger(i -> j)`` counts side-A events at node i inside the backward
  coincidence window of side-B node j (counts A events; binomial confidence
  with ``n = N_A``).
"""
import json

import numpy as np
import xarray as xr
from scipy.stats import binom

from . import _kernels
from .conventions import (
    TIME,
    as_pair,
    record_provenance,
    require_event_series,
    require_pairwise,
)
from .engine import infer_count_dtype, pairwise_apply

__all__ = [
    "get_eca_precursor_window",
    "get_eca_trigger_window",
    "get_eca_precursor",
    "get_eca_trigger",
    "get_eca_precursor_from_events",
    "get_eca_trigger_from_events",
    "get_eca_precursor_confidence",
    "get_eca_trigger_confidence",
]


"""
Coincidence windows
"""


def _window_attrs(kind: str, delt: int, sym: bool, tau: int) -> dict:
    return {
        "long_name": f"{kind} Window",
        "units": "1",
        "description": f"Window for {kind.lower()} event identification",
        "eca_params": json.dumps({"delt": delt, "sym": sym, "tau": tau}),
    }


def get_eca_precursor_window(da: xr.DataArray, delt: int = 2, sym: bool = True, tau: int = 0) -> xr.DataArray:
    """Smear an event series over the forward (precursor) coincidence window."""
    require_event_series(da, name="get_eca_precursor_window input")
    da_window = xr.apply_ufunc(
        _kernels.forward_window,
        da,
        input_core_dims=[[TIME]],
        output_core_dims=[[TIME]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[bool],
        kwargs={"delt": delt, "sym": sym, "tau": tau},
    )
    da_window.attrs = _window_attrs("Precursor", delt, sym, tau)
    return da_window


def get_eca_trigger_window(da: xr.DataArray, delt: int = 2, sym: bool = True, tau: int = 0) -> xr.DataArray:
    """Smear an event series over the backward (trigger) coincidence window.

    Note that the trigger window belongs to the *target* side (side B) of the
    directed pair: trigger counts side-A events inside side-B windows.
    """
    require_event_series(da, name="get_eca_trigger_window input")
    da_window = xr.apply_ufunc(
        _kernels.backward_window,
        da,
        input_core_dims=[[TIME]],
        output_core_dims=[[TIME]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[bool],
        kwargs={"delt": delt, "sym": sym, "tau": tau},
    )
    da_window.attrs = _window_attrs("Trigger", delt, sym, tau)
    return da_window


def _resolve_eca_params(window: xr.DataArray, delt, sym, tau) -> dict:
    """ECA parameters from explicit arguments, falling back to window attrs."""
    given = {"delt": delt, "sym": sym, "tau": tau}
    if all(v is not None for v in given.values()):
        return given
    if any(v is not None for v in given.values()):
        raise ValueError("pass all of delt/sym/tau, or none to read them from the window attrs")
    params_raw = window.attrs.get("eca_params")
    if params_raw is None:
        raise ValueError(
            "ECA parameters unavailable: pass delt/sym/tau explicitly, or use a "
            "window produced by get_eca_precursor_window/get_eca_trigger_window"
        )
    return json.loads(params_raw)


"""
Pairwise coincidence counts
"""


def get_eca_precursor(
    eventA_precursor_window: xr.DataArray,
    eventB: xr.DataArray,
    delt: int = None,
    sym: bool = None,
    tau: int = None,
    parallel: bool = True,
) -> xr.DataArray:
    """
    Count precursor coincidences for every directed pair ``node_i -> node_j``.

    Parameters
    ----------
    eventA_precursor_window : xr.DataArray
        Precursor window of side A (see :func:`get_eca_precursor_window`).
    eventB : xr.DataArray
        Binary event series of side B, node format.
    delt, sym, tau : optional
        ECA parameters; by default read from the window's ``eca_params`` attrs.
    parallel : bool, optional
        Whether to use the parallel numba kernel, by default True.
    """
    eca_params = _resolve_eca_params(eventA_precursor_window, delt, sym, tau)
    output_dtype = infer_count_dtype(eventA_precursor_window.sizes[TIME])

    da_precursor = pairwise_apply(
        _kernels.eca_precursor_counts,
        left=[eventA_precursor_window],
        right=[eventB],
        left_core_dims=[(TIME,)],
        right_core_dims=[(TIME,)],
        output_dtype=output_dtype,
        kernel_kwargs={"output_dtype": output_dtype},
        parallel=parallel,
    )
    da_precursor.attrs = {
        "long_name": "Precursor Events",
        "units": "count",
        "description": "Number of side-B events (node_j) inside the precursor window of side A (node_i)",
        "eca_params": json.dumps(eca_params),
    }
    record_provenance(da_precursor, "get_eca_precursor", eca_params)
    return da_precursor


def get_eca_trigger(
    eventA: xr.DataArray,
    eventB_trigger_window: xr.DataArray,
    delt: int = None,
    sym: bool = None,
    tau: int = None,
    parallel: bool = True,
) -> xr.DataArray:
    """
    Count trigger coincidences for every directed pair ``node_i -> node_j``.

    Parameters
    ----------
    eventA : xr.DataArray
        Binary event series of side A, node format.
    eventB_trigger_window : xr.DataArray
        Trigger window of side B (see :func:`get_eca_trigger_window`).
    delt, sym, tau : optional
        ECA parameters; by default read from the window's ``eca_params`` attrs.
    parallel : bool, optional
        Whether to use the parallel numba kernel, by default True.
    """
    eca_params = _resolve_eca_params(eventB_trigger_window, delt, sym, tau)
    output_dtype = infer_count_dtype(eventB_trigger_window.sizes[TIME])

    da_trigger = pairwise_apply(
        _kernels.eca_trigger_counts,
        left=[eventA],
        right=[eventB_trigger_window],
        left_core_dims=[(TIME,)],
        right_core_dims=[(TIME,)],
        output_dtype=output_dtype,
        kernel_kwargs={"output_dtype": output_dtype},
        parallel=parallel,
    )
    da_trigger.attrs = {
        "long_name": "Trigger Events",
        "units": "count",
        "description": "Number of side-A events (node_i) inside the trigger window of side B (node_j)",
        "eca_params": json.dumps(eca_params),
    }
    record_provenance(da_trigger, "get_eca_trigger", eca_params)
    return da_trigger


def get_eca_precursor_from_events(
    eventA: xr.DataArray,
    eventB: xr.DataArray,
    delt: int = 2,
    sym: bool = True,
    tau: int = 0,
    parallel: bool = True,
) -> xr.DataArray:
    """
    Calculate precursor coincidences from two event series (side A -> side B).

    Parameters
    ----------
    eventA : xr.DataArray
        Binary event series of side A (``node_i``), node format.
    eventB : xr.DataArray
        Binary event series of side B (``node_j``), node format.
    delt : int, optional
        Length of the coincidence window, by default 2
    sym : bool, optional
        If True, use symmetric window, by default True
    tau : int, optional
        Time lag from eventA to eventB, by default 0
    parallel : bool, optional
        Whether to use the parallel numba kernel, by default True.
    """
    window = get_eca_precursor_window(eventA, delt, sym, tau)
    return get_eca_precursor(window, eventB, parallel=parallel)


def get_eca_trigger_from_events(
    eventA: xr.DataArray,
    eventB: xr.DataArray,
    delt: int = 2,
    sym: bool = True,
    tau: int = 0,
    parallel: bool = True,
) -> xr.DataArray:
    """
    Calculate trigger coincidences from two event series (side A -> side B).

    The trigger window is built from ``eventB`` (the target side): trigger
    counts side-A events followed by a side-B event within the window.

    Parameters
    ----------
    eventA : xr.DataArray
        Binary event series of side A (``node_i``), node format.
    eventB : xr.DataArray
        Binary event series of side B (``node_j``), node format.
    delt : int, optional
        Length of the coincidence window, by default 2
    sym : bool, optional
        If True, use symmetric window, by default True
    tau : int, optional
        Time lag from eventA to eventB, by default 0
    parallel : bool, optional
        Whether to use the parallel numba kernel, by default True.
    """
    window = get_eca_trigger_window(eventB, delt, sym, tau)
    return get_eca_trigger(eventA, window, parallel=parallel)


"""
Confidence calculation
"""


def prec_confidence(kp, na, nb, TOL, T, tau):
    return binom.cdf(kp, n=nb, p=1 - (1 - TOL / (T - tau)) ** na).astype(np.float32)


def trig_confidence(kt, na, nb, TOL, T, tau):
    return binom.cdf(kt, n=na, p=1 - (1 - TOL / (T - tau)) ** nb).astype(np.float32)


def _confidence_params(counts: xr.DataArray, delt, sym, tau) -> dict:
    given = {"delt": delt, "sym": sym, "tau": tau}
    if all(v is not None for v in given.values()):
        return given
    if any(v is not None for v in given.values()):
        raise ValueError("pass all of delt/sym/tau, or none to read them from the counts attrs")
    params_raw = counts.attrs.get("eca_params")
    if params_raw is None:
        raise ValueError(
            "ECA parameters unavailable: pass delt/sym/tau explicitly, or use "
            "counts produced by get_eca_precursor/get_eca_trigger"
        )
    return json.loads(params_raw)


def get_eca_precursor_confidence(
    precursor: xr.DataArray,
    eventA: xr.DataArray,
    eventB: xr.DataArray,
    min_eventnum: int = 2,
    delt: int = None,
    sym: bool = None,
    tau: int = None,
) -> xr.DataArray:
    """
    Confidence of precursor counts under the binomial null hypothesis.

    Parameters
    ----------
    precursor : xr.DataArray
        Pairwise precursor counts (``node_i``, ``node_j``).
    eventA, eventB : xr.DataArray
        The binary event series the counts were computed from, node format.
    min_eventnum : int, optional
        Pairs where either side has fewer events get confidence 0, by default 2.
    delt, sym, tau : optional
        ECA parameters; by default read from the counts' ``eca_params`` attrs.
    """
    require_pairwise(precursor, name="precursor")
    require_event_series(eventA, name="eventA")
    require_event_series(eventB, name="eventB")

    eca_params = _confidence_params(precursor, delt, sym, tau)
    TOL = eca_params["delt"] * eca_params["sym"] + 1
    tau_val = eca_params["tau"]
    T = eventA.sizes[TIME]

    NA = as_pair(eventA.sum(dim=TIME), "i")
    NB = as_pair(eventB.sum(dim=TIME), "j")

    prec_conf = xr.apply_ufunc(
        prec_confidence,
        precursor,
        NA,
        NB,
        dask="parallelized",
        kwargs={"TOL": TOL, "T": T, "tau": tau_val},
    ).rename("prec_conf")

    if min_eventnum > 0:
        prec_conf = prec_conf.where(NA >= min_eventnum, 0.0)
        prec_conf = prec_conf.where(NB >= min_eventnum, 0.0)

    prec_conf.attrs = {
        "long_name": "Precursor confidence",
        "units": "",
        "description": "Confidence of precursor counts for directed pairs node_i -> node_j",
        "eca_params": json.dumps(eca_params),
    }
    record_provenance(prec_conf, "get_eca_precursor_confidence", {**eca_params, "min_eventnum": min_eventnum})
    return prec_conf


def get_eca_trigger_confidence(
    trigger: xr.DataArray,
    eventA: xr.DataArray,
    eventB: xr.DataArray,
    min_eventnum: int = 2,
    delt: int = None,
    sym: bool = None,
    tau: int = None,
) -> xr.DataArray:
    """
    Confidence of trigger counts under the binomial null hypothesis.

    Parameters
    ----------
    trigger : xr.DataArray
        Pairwise trigger counts (``node_i``, ``node_j``).
    eventA, eventB : xr.DataArray
        The binary event series the counts were computed from, node format.
    min_eventnum : int, optional
        Pairs where either side has fewer events get confidence 0, by default 2.
    delt, sym, tau : optional
        ECA parameters; by default read from the counts' ``eca_params`` attrs.
    """
    require_pairwise(trigger, name="trigger")
    require_event_series(eventA, name="eventA")
    require_event_series(eventB, name="eventB")

    eca_params = _confidence_params(trigger, delt, sym, tau)
    TOL = eca_params["delt"] * eca_params["sym"] + 1
    tau_val = eca_params["tau"]
    T = eventA.sizes[TIME]

    NA = as_pair(eventA.sum(dim=TIME), "i")
    NB = as_pair(eventB.sum(dim=TIME), "j")

    trig_conf = xr.apply_ufunc(
        trig_confidence,
        trigger,
        NA,
        NB,
        dask="parallelized",
        kwargs={"TOL": TOL, "T": T, "tau": tau_val},
    ).rename("trig_conf")

    if min_eventnum > 0:
        trig_conf = trig_conf.where(NA >= min_eventnum, 0.0)
        trig_conf = trig_conf.where(NB >= min_eventnum, 0.0)

    trig_conf.attrs = {
        "long_name": "Trigger confidence",
        "units": "",
        "description": "Confidence of trigger counts for directed pairs node_i -> node_j",
        "eca_params": json.dumps(eca_params),
    }
    record_provenance(trig_conf, "get_eca_trigger_confidence", {**eca_params, "min_eventnum": min_eventnum})
    return trig_conf
