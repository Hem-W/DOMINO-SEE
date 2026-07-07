Quickstart
==========

This quickstart guide will help you get up and running with `dominosee` for analyzing interconnected hydroclimatic extreme events using Event Coincidence Analysis (ECA).

Basic Setup
-----------

First, import the necessary packages:

.. code-block:: python

    import numpy as np
    import xarray as xr
    import dominosee as dsee

Creating Sample Data
--------------------

Let's create a sample climate dataset with SPI (Standardized Precipitation Index) values:

.. code-block:: python

    # Create a sample dataset
    nx, ny, nt = 20, 20, 365  # 20x20 grid, 365 days
    
    # Create coordinates
    lats = np.linspace(-90, 90, nx)
    lons = np.linspace(-180, 180, ny)
    times = xr.date_range("1950-01-01", periods=nt, freq="D")
    
    # Create standard normal data for SPI values
    spi_data = np.random.normal(0, 1, size=(nx, ny, nt))
    
    # Create xarray Dataset
    spi = xr.Dataset(
        data_vars={"SPI1": (["lat", "lon", "time"], spi_data)},
        coords={"lat": lats, "lon": lons, "time": times}
    )

Extracting Extreme Events
--------------------------

Identify extreme events using a threshold:

.. code-block:: python

    from dominosee.eventorize import get_event

    # Extract drought events (SPI < -1.0)
    da_event = get_event(
        spi.SPI1,
        threshold=-1.0,
        extreme="below",
        event_name="drought"
    )

The ``event_name`` becomes the event's **layer** — it is recorded as a scalar
``layer`` coordinate and propagates through the whole pipeline.

Flattening Space Into Nodes
---------------------------

DOMINO-SEE networks relate pairs of spatial points, called **nodes**. Flatten
the ``lat``/``lon`` grid into the canonical integer ``node`` dimension first —
``lat`` and ``lon`` become auxiliary coordinates riding on ``node``, so the
result still serializes to netCDF/zarr:

.. code-block:: python

    da_event = dsee.to_node_format(da_event)
    # dims: (node, time); coords: node, lat(node), lon(node), layer

Event Coincidence Analysis (ECA)
---------------------------------

Calculate event coincidences between location pairs:

.. code-block:: python

    from dominosee.eca import (
        get_eca_precursor_from_events,
        get_eca_trigger_from_events,
        get_eca_precursor_confidence,
        get_eca_trigger_confidence
    )
    
    # Calculate precursor and trigger events
    da_precursor = get_eca_precursor_from_events(
        eventA=da_event, 
        eventB=da_event, 
        delt=2,  # Time window
        sym=True,  # Symmetric window
        tau=0
    )
    
    da_trigger = get_eca_trigger_from_events(
        eventA=da_event, 
        eventB=da_event, 
        delt=10, 
        sym=True, 
        tau=0
    )
    
    # Calculate statistical confidence
    da_prec_conf = get_eca_precursor_confidence(
        precursor=da_precursor, 
        eventA=da_event, 
        eventB=da_event
    )
    
    da_trig_conf = get_eca_trigger_confidence(
        trigger=da_trigger,
        eventA=da_event,
        eventB=da_event
    )

All pairwise results have the fixed dimensions ``(node_i, node_j)`` with the
sided coordinates ``lat_i``/``lon_i``/``layer_i`` (and the ``_j``
counterparts). ECA networks are directed: entry ``[i, j]`` refers to the pair
``node_i -> node_j``.

Constructing Networks
---------------------

Create network adjacency matrices from significant connections:

.. code-block:: python

    from dominosee.network import get_link_from_confidence

    # Create network from ECA confidence levels
    da_link = (
        get_link_from_confidence(da_prec_conf, 0.99) &
        get_link_from_confidence(da_trig_conf, 0.99)
    )

    # Calculate network density
    density = da_link.sum().values / da_link.size * 100
    print(f"Network density: {density:.2f}%")


