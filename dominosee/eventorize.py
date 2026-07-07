# encoding: utf-8
from itertools import groupby
import numpy as np
import xarray as xr

from .conventions import LAYER, record_provenance

__all__ = [
    'get_event',
    'get_event_percentile',
]

"""
Event selection
# TODO: much of the event selection below is already implemented in xclim
"""
def _select_burst(te):
    tb = te.copy()  # time of bursts
    tb0 = np.roll(tb, 1)
    tb0[0] = False
    tb[tb & tb0] = False
    return tb


def _select_wane(te):
    tw = te.copy()  # time of wanes
    tw0 = np.roll(tw, -1)
    tw0[-1] = False
    tw[te & tw0] = False
    return tw

def _start_consecutive(event_bool, period=1):
    du_num = np.concatenate([[len(list(j)) for i, j in groupby(event_bool) if i]]) # get event durations
    due = np.zeros_like(event_bool)
    if event_bool.any():
        due[event_bool] = np.concatenate([np.ones(du)*du if du <= period 
                                                     else np.concatenate((np.ones(period)*du, np.zeros(du-period))) 
                                                     for du in du_num]) # select first period
    return due

def _end_consecutive(event_bool, period=1):
    du_num = np.concatenate([[len(list(j)) for i, j in groupby(event_bool) if i]]) # get event durations
    due = np.zeros_like(event_bool)
    if event_bool.any():
        due[event_bool] = np.concatenate([np.ones(du)*du if du <= period 
                                                     else np.concatenate((np.zeros(du-period), np.ones(period)*du)) 
                                                     for du in du_num]) # select first period
    return due


def select_start_consecutive(event_bool: xr.DataArray, period: int=1) -> xr.DataArray:
    """
    Select the first period of each consecutive event along the time dimension.
    
    Parameters
    ----------
    event_bool : xarray.DataArray
        DataArray containing boolean event values
    period : int, optional
        Number of time steps to select from the beginning of each event, by default 1
        
    Returns
    -------
    xarray.DataArray
        DataArray with the selected first period values
    """
    assert event_bool.dtype == bool, "Event time series (event_bool) should be boolean"
    
    # TODO: 处理输入数据不连续的情况，先补充成连续的再计算再还原
    return xr.apply_ufunc(
        _start_consecutive,
        event_bool,
        input_core_dims=[["time"]],
        output_core_dims=[["time"]],
        vectorize=True,
        dask='parallelized',
        kwargs={"period": period}
    )

def select_end_consecutive(event_bool: xr.DataArray, period: int=1) -> xr.DataArray:
    """
    Select the last period of each consecutive event along the time dimension.
    
    Parameters
    ----------
    event_bool : xarray.DataArray
        DataArray containing boolean event values
    period : int, optional
        Number of time steps to select from the end of each event, by default 1
        
    Returns
    -------
    xarray.DataArray
        DataArray with the selected end period values
    """
    assert event_bool.dtype == bool, "Event time series (event_bool) should be boolean"
    
    return xr.apply_ufunc(
        _end_consecutive,
        event_bool,
        input_core_dims=[['time']],
        output_core_dims=[['time']],
        vectorize=True,
        dask='parallelized',
        kwargs={"period": period}
    )


"""
Extreme definitions
"""
def cut_single_threshold(ts, th, extreme="above", select=None, select_period=None):
    """
    Apply cut_single_threshold to an xarray DataArray.
    
    Parameters
    ----------
    ts : xarray.DataArray
        DataArray containing data with dimension 'time'
    th : float
        Threshold value
    extreme : {'above', 'below'}, optional
        Whether to select values above or below the threshold, by default "above"
    select : {'burst', 'wane', 'start', 'end'}, optional
        Type of event selection, by default None
    select_period : int, optional
        Number of time steps to select from the beginning or end of each event, by default None
    
    Returns
    -------
    xarray.DataArray
        DataArray with the events in datatype bool
    """
    assert extreme in ["above", "below"], "extreme should be 'above' or 'below'"

    if extreme == "above":
        te = ts >= th
    elif extreme == "below":
        te = ts <= th
    
    select_period = 1 if select_period is None and select is not None else select_period
    
    if select == "burst":
        te = _select_burst(te)
    elif select == "wane":
        te = _select_wane(te)
    elif select == "start":
        te = select_start_consecutive(te, select_period)
    elif select == "end":
        te = select_end_consecutive(te, select_period)
    return te


def cut_single_percentile(ts, q, extreme="above", select=None, select_period=None):
    """
    Apply percentile-based threshold to a time series.
    
    Parameters
    ----------
    ts : numpy.ndarray or xarray.DataArray
        Time series data
    q : float
        Percentile value (0-1)
    extreme : {'above', 'below'}, optional
        Whether to select values above or below the percentile threshold, by default "above"
    select : {'burst', 'wane', 'start', 'end'}, optional
        Type of event selection, by default None
    select_period : int, optional
        Number of time steps to select from the beginning or end of each event, by default None
    
    Returns
    -------
    numpy.ndarray or xarray.DataArray
        Boolean array indicating events
    """
    assert extreme in ["above", "below"], "extreme should be 'above' or 'below'"
    
    # Calculate threshold as percentile for this specific time series
    threshold = np.nanquantile(ts, q)
    
    if extreme == "above":
        te = ts >= threshold
    elif extreme == "below":
        te = ts <= threshold
    
    select_period = 1 if select_period is None and select is not None else select_period
    
    if select == "burst":
        te = _select_burst(te)
    elif select == "wane":
        te = _select_wane(te)
    elif select == "start":
        te = select_start_consecutive(te, select_period)
    elif select == "end":
        te = select_end_consecutive(te, select_period)
    return te


"""
Exposed API to DataArray
"""
def get_event(da: xr.DataArray, threshold: float, extreme: str, 
              event_name: str=None, select: str=None, select_period: int=None) -> xr.DataArray:
    """
    Apply get_event to an xarray DataArray.
    
    Parameters
    ----------
    da : xarray.DataArray
        DataArray containing data with dimension 'time'
    threshold : float
        Threshold value
    extreme : {'above', 'below'}, optional
        Whether to select values above or below the threshold, by default "above"
    event_name : str, optional
        Name of the event, by default "event"
    select : {'burst', 'wane', 'start', 'end'}, optional
        Type of event selection, by default None
    select_period : int, optional
        Number of time steps to select from the beginning or end of each event, by default None
    
    Returns
    -------
    xarray.DataArray
        DataArray with the events in datatype bool
    """
    event_name = "event" if event_name is None else event_name
    da = xr.apply_ufunc(
        cut_single_threshold,
        da,
        input_core_dims=[['time']],
        output_core_dims=[['time']],
        vectorize=True,
        dask='parallelized',
        kwargs={"th": threshold, "extreme": extreme, "select": select, "select_period": select_period}
    ).rename(event_name)

    da = da.assign_coords({LAYER: event_name})
    da.attrs = {
        "threshold": threshold,
        "extreme": extreme,
        "long_name": f"{event_name} events",
        "description": f"Events with {threshold} {extreme} threshold",
        "event_name": event_name,
        "select": select,
        "select_period": select_period
    }
    record_provenance(da, "get_event", {"threshold": threshold, "extreme": extreme,
                                        "event_name": event_name, "select": select,
                                        "select_period": select_period})
    return da



def get_event_percentile(da: xr.DataArray, percentile: float, extreme: str="above",
                        event_name: str=None, select: str=None, select_period: int=None) -> xr.DataArray:
    """
    Apply percentile-based event detection to an xarray DataArray. The percentile threshold
    is calculated separately for each spatial point.
    
    Parameters
    ----------
    da : xarray.DataArray
        DataArray containing data with dimension 'time'
    percentile : float
        Percentile value between 0 and 1 to use as threshold
    extreme : {'above', 'below'}, optional
        Whether to select values above or below the percentile threshold, by default "above"
    event_name : str, optional
        Name of the event, by default "event"
    select : {'burst', 'wane', 'start', 'end'}, optional
        Type of event selection, by default None
    select_period : int, optional
        Number of time steps to select from the beginning or end of each event, by default None
    
    Returns
    -------
    xarray.DataArray
        DataArray with the events in datatype bool
    """
    assert 0 <= percentile <= 1, "percentile must be between 0 and 1"
    event_name = "event" if event_name is None else event_name
    
    da = xr.apply_ufunc(
        cut_single_percentile,
        da,
        input_core_dims=[['time']],
        output_core_dims=[['time']],
        vectorize=True,
        dask='parallelized',
        kwargs={"q": percentile, "extreme": extreme, "select": select, "select_period": select_period}
    ).rename(event_name)

    da = da.assign_coords({LAYER: event_name})
    da.attrs = {
        "percentile": percentile,
        "extreme": extreme,
        "long_name": f"{event_name} events",
        "description": f"Events with {percentile} percentile {extreme} threshold",
        "event_name": event_name,
        "select": select,
        "select_period": select_period
    }
    record_provenance(da, "get_event_percentile", {"percentile": percentile, "extreme": extreme,
                                                   "event_name": event_name, "select": select,
                                                   "select_period": select_period})
    return da