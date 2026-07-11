"""
Test suite for the quickstart tutorial.

This module tests that all code examples in docs/source/quickstart.rst
are executable and produce expected results.
"""

import numpy as np
import pytest
import xarray as xr

import dominosee as dsee


class TestQuickstartTutorial:
    """Test all code examples from the quickstart tutorial."""

    @pytest.fixture
    def sample_spi_dataset(self):
        """Create sample SPI dataset as shown in quickstart."""
        # Create a sample dataset
        nx, ny, nt = 20, 20, 365  # 20x20 grid, 365 days

        # Create coordinates
        lats = np.linspace(-90, 90, nx)
        lons = np.linspace(-180, 180, ny)
        times = xr.date_range("1950-01-01", periods=nt, freq="D")

        # Create standard normal data for SPI values
        np.random.seed(42)  # For reproducibility
        spi_data = np.random.normal(0, 1, size=(nx, ny, nt))

        # Create xarray Dataset
        spi = xr.Dataset(
            data_vars={"SPI1": (["lat", "lon", "time"], spi_data)},
            coords={"lat": lats, "lon": lons, "time": times},
        )

        return spi

    @pytest.fixture
    def sample_node_events(self, sample_spi_dataset):
        """Events flattened into node format, as shown in quickstart."""
        from dominosee.eventorize import get_event

        da_event = get_event(
            sample_spi_dataset.SPI1, threshold=-1.0, extreme="below", event_name="drought"
        )
        return dsee.to_node_format(da_event)

    def test_imports(self):
        """Test that all required imports work."""
        assert np is not None
        assert xr is not None
        assert dsee is not None

    def test_create_sample_data(self, sample_spi_dataset):
        """Test creating sample SPI dataset."""
        spi = sample_spi_dataset

        # Verify dataset structure
        assert isinstance(spi, xr.Dataset)
        assert "SPI1" in spi.data_vars
        assert set(spi.dims) == {"lat", "lon", "time"}
        assert spi.sizes["lat"] == 20
        assert spi.sizes["lon"] == 20
        assert spi.sizes["time"] == 365

        # Verify coordinates
        assert "lat" in spi.coords
        assert "lon" in spi.coords
        assert "time" in spi.coords

    def test_extract_extreme_events(self, sample_spi_dataset):
        """Test extracting extreme events using get_event."""
        from dominosee.eventorize import get_event

        spi = sample_spi_dataset

        # Extract drought events (SPI < -1.0)
        da_event = get_event(
            spi.SPI1, threshold=-1.0, extreme="below", event_name="drought"
        )

        # Verify event extraction
        assert isinstance(da_event, xr.DataArray)
        assert da_event.dtype == bool
        assert da_event.shape == spi.SPI1.shape
        assert da_event.name == "drought"

        # The event name becomes the layer coordinate
        assert da_event["layer"].item() == "drought"

        # Verify attributes
        assert da_event.attrs["threshold"] == -1.0
        assert da_event.attrs["extreme"] == "below"
        assert da_event.attrs["event_name"] == "drought"

        # Verify some events were detected
        assert da_event.sum() > 0

    def test_to_node_format(self, sample_node_events):
        """Test flattening the spatial grid into nodes."""
        da_event = sample_node_events

        assert set(da_event.dims) == {"node", "time"}
        assert da_event.sizes["node"] == 400
        assert da_event["lat"].dims == ("node",)
        assert da_event["lon"].dims == ("node",)

    def test_eca_precursor_trigger(self, sample_node_events):
        """Test ECA precursor and trigger calculation."""
        from dominosee.eca import (
            get_eca_precursor_from_events,
            get_eca_trigger_from_events,
        )

        da_event = sample_node_events

        # Calculate precursor and trigger events
        da_precursor = get_eca_precursor_from_events(
            eventA=da_event, eventB=da_event, delt=2, sym=True, tau=0
        )

        da_trigger = get_eca_trigger_from_events(
            eventA=da_event, eventB=da_event, delt=10, sym=True, tau=0
        )

        # Verify precursor results
        assert isinstance(da_precursor, xr.DataArray)
        assert da_precursor.dims == ("node_i", "node_j")
        assert da_precursor.sizes["node_i"] == 400
        assert da_precursor.sizes["node_j"] == 400
        assert "lat_i" in da_precursor.coords and "lon_j" in da_precursor.coords
        assert da_precursor["layer_i"].item() == "drought"
        assert "eca_params" in da_precursor.attrs

        # Verify trigger results
        assert isinstance(da_trigger, xr.DataArray)
        assert da_trigger.dims == ("node_i", "node_j")
        assert da_trigger.sizes["node_i"] == 400
        assert da_trigger.sizes["node_j"] == 400
        assert "eca_params" in da_trigger.attrs

    def test_eca_confidence(self, sample_node_events):
        """Test ECA confidence calculation."""
        from dominosee.eca import (
            get_eca_precursor_from_events,
            get_eca_trigger_from_events,
            get_eca_precursor_confidence,
            get_eca_trigger_confidence,
        )

        da_event = sample_node_events

        da_precursor = get_eca_precursor_from_events(
            eventA=da_event, eventB=da_event, delt=2, sym=True, tau=0
        )

        da_trigger = get_eca_trigger_from_events(
            eventA=da_event, eventB=da_event, delt=10, sym=True, tau=0
        )

        # Calculate statistical confidence
        da_prec_conf = get_eca_precursor_confidence(
            precursor=da_precursor, eventA=da_event, eventB=da_event
        )

        da_trig_conf = get_eca_trigger_confidence(
            trigger=da_trigger, eventA=da_event, eventB=da_event
        )

        # Verify precursor confidence
        assert isinstance(da_prec_conf, xr.DataArray)
        assert da_prec_conf.dims == ("node_i", "node_j")
        assert da_prec_conf.min() >= 0.0
        assert da_prec_conf.max() <= 1.0

        # Verify trigger confidence
        assert isinstance(da_trig_conf, xr.DataArray)
        assert da_trig_conf.dims == ("node_i", "node_j")
        assert da_trig_conf.min() >= 0.0
        assert da_trig_conf.max() <= 1.0

    def test_complete_workflow(self, sample_node_events):
        """Test the complete quickstart workflow end-to-end."""
        from dominosee.eca import (
            get_eca_precursor_from_events,
            get_eca_trigger_from_events,
            get_eca_precursor_confidence,
            get_eca_trigger_confidence,
        )
        from dominosee.network import get_link_from_confidence

        da_event = sample_node_events

        # Calculate precursor and trigger events
        da_precursor = get_eca_precursor_from_events(
            eventA=da_event, eventB=da_event, delt=2, sym=True, tau=0
        )

        da_trigger = get_eca_trigger_from_events(
            eventA=da_event, eventB=da_event, delt=10, sym=True, tau=0
        )

        # Calculate statistical confidence
        da_prec_conf = get_eca_precursor_confidence(
            precursor=da_precursor, eventA=da_event, eventB=da_event
        )

        da_trig_conf = get_eca_trigger_confidence(
            trigger=da_trigger, eventA=da_event, eventB=da_event
        )

        # Create network from ECA confidence levels
        da_link = get_link_from_confidence(
            da_prec_conf, 0.99
        ) & get_link_from_confidence(da_trig_conf, 0.99)

        # Calculate network density
        density = da_link.sum().values / da_link.size * 100

        # Verify complete workflow succeeded
        assert isinstance(da_link, xr.DataArray)
        assert da_link.dtype == bool
        assert da_link.dims == ("node_i", "node_j")
        assert 0 <= density <= 100

        # Print network density as in the tutorial
        print(f"Network density: {density:.2f}%")
