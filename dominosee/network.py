"""Thresholding pairwise statistics into link (adjacency) matrices.

Comparison operators drop ``attrs`` in xarray, so every ``get_link_from_*``
function restores the input metadata on the boolean result and records the
thresholding parameters. Pass ``directed`` to record edge directedness
(ECA networks are directed ``node_i -> node_j``, ES networks undirected).
"""
import xarray as xr

from .conventions import record_provenance

__all__ = [
    "get_link_from_threshold",
    "get_link_from_significance",
    "get_link_from_confidence",
    "get_link_from_quantile",
    "get_link_from_critical_values",
]


def _finalize_link(da_link: xr.DataArray, source: xr.DataArray, func_name: str, params: dict,
                   directed: bool = None) -> xr.DataArray:
    da_link.attrs = dict(source.attrs)
    da_link.attrs.pop("units", None)
    da_link.attrs["long_name"] = "Network links"
    if directed is not None:
        da_link.attrs["directed"] = directed
    record_provenance(da_link, func_name, params)
    return da_link


def get_link_from_threshold(da_sig: xr.DataArray, threshold: float, directed: bool = None) -> xr.DataArray:
    """
    Get the link from threshold DataArray

    Parameters
    ----------
    da_sig : xr.DataArray
        DataArray containing the values to compare
    threshold : float
        Threshold value
    directed : bool, optional
        Whether the resulting links are directed; recorded in attrs.

    Returns
    -------
    xr.DataArray
        Boolean DataArray where True indicates a link
    """
    da_link = da_sig >= threshold
    return _finalize_link(da_link, da_sig, "get_link_from_threshold",
                          {"threshold": threshold}, directed)


def get_link_from_significance(da_sig: xr.DataArray, p_threshold: float, directed: bool = None) -> xr.DataArray:
    """
    Get the link from significance DataArray

    Parameters
    ----------
    da_sig : xr.DataArray
        DataArray containing the values to compare
    p_threshold : float
        Significance level
    directed : bool, optional
        Whether the resulting links are directed; recorded in attrs.

    Returns
    -------
    xr.DataArray
        Boolean DataArray where True indicates a link
    """
    da_link = da_sig <= p_threshold
    return _finalize_link(da_link, da_sig, "get_link_from_significance",
                          {"p_threshold": p_threshold}, directed)


def get_link_from_confidence(da_conf: xr.DataArray, confidence_level: float, directed: bool = None) -> xr.DataArray:
    """
    Get the link from confidence DataArray

    Parameters
    ----------
    da_conf : xr.DataArray
        DataArray containing the values to compare
    confidence_level : float
        Confidence level
    directed : bool, optional
        Whether the resulting links are directed; recorded in attrs.

    Returns
    -------
    xr.DataArray
        Boolean DataArray where True indicates a link
    """
    da_link = da_conf >= confidence_level
    return _finalize_link(da_link, da_conf, "get_link_from_confidence",
                          {"confidence_level": confidence_level}, directed)


def get_link_from_quantile(da_quant: xr.DataArray, q: float, directed: bool = None) -> xr.DataArray:
    """
    Get the link from quantile DataArray

    Parameters
    ----------
    da_quant : xr.DataArray
        DataArray containing the values to compare
    q : float
        Quantile level
    directed : bool, optional
        Whether the resulting links are directed; recorded in attrs.

    Returns
    -------
    xr.DataArray
        Boolean DataArray where True indicates a link
    """
    quant = da_quant.quantile(q)
    da_link = (da_quant >= quant).drop_vars("quantile", errors="ignore")
    return _finalize_link(da_link, da_quant, "get_link_from_quantile", {"q": q}, directed)


def get_link_from_critical_values(da_valu: xr.DataArray, critical_value: xr.DataArray,
                                  rule: str = "greater", directed: bool = None) -> xr.DataArray:
    """
    Get the link from critical values DataArray

    Parameters
    ----------
    da_valu : xr.DataArray
        DataArray containing the values to compare
    critical_value : xr.DataArray
        DataArray containing the critical values
    rule : str, optional
        Comparison rule, either "greater" or "greater_equal" (default: "greater")
    directed : bool, optional
        Whether the resulting links are directed; recorded in attrs.

    Returns
    -------
    xr.DataArray
        Boolean DataArray where True indicates a link
    """
    if rule == "greater":
        da_link = da_valu > critical_value
    elif rule == "greater_equal":
        da_link = da_valu >= critical_value
    else:
        raise ValueError("rule must be either 'greater' or 'greater_equal'")
    return _finalize_link(da_link, da_valu, "get_link_from_critical_values",
                          {"rule": rule}, directed)
