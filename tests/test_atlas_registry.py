"""Tests for pineuro.atlas_registry — atlas source metadata registry."""

from pineuro.atlas_registry import ATLAS_SOURCES

# =============================================================================
# Registry structure
# =============================================================================


class TestAtlasSourcesRegistry:
    def test_registry_is_dict(self):
        assert isinstance(ATLAS_SOURCES, dict)

    def test_not_empty(self):
        assert len(ATLAS_SOURCES) > 0

    def test_all_entries_have_display_name(self):
        for key, entry in ATLAS_SOURCES.items():
            assert "display_name" in entry, f"Missing display_name for {key}"
            assert isinstance(entry["display_name"], str)

    def test_all_entries_have_type(self):
        for key, entry in ATLAS_SOURCES.items():
            assert "type" in entry, f"Missing type for {key}"
            assert entry["type"] in ("network", "anatomical", "tissue")


# =============================================================================
# Expected atlases
# =============================================================================


class TestExpectedAtlases:
    def test_yeo_7_present(self):
        assert "yeo_7" in ATLAS_SOURCES
        assert ATLAS_SOURCES["yeo_7"]["type"] == "network"
        assert ATLAS_SOURCES["yeo_7"]["n_networks"] == 7

    def test_yeo_17_present(self):
        assert "yeo_17" in ATLAS_SOURCES
        assert ATLAS_SOURCES["yeo_17"]["type"] == "network"
        assert ATLAS_SOURCES["yeo_17"]["n_networks"] == 17

    def test_schaefer_atlases_present(self):
        schaefer_keys = [k for k in ATLAS_SOURCES if k.startswith("schaefer")]
        assert len(schaefer_keys) >= 2

    def test_harvard_oxford_present(self):
        assert "harvard_oxford" in ATLAS_SOURCES
        assert ATLAS_SOURCES["harvard_oxford"]["type"] == "anatomical"

    def test_gray_matter_present(self):
        assert "gray_matter" in ATLAS_SOURCES
        assert ATLAS_SOURCES["gray_matter"]["type"] == "tissue"


# =============================================================================
# Network atlases have n_networks
# =============================================================================


class TestNetworkAtlasMetadata:
    def test_network_atlases_have_n_networks(self):
        for key, entry in ATLAS_SOURCES.items():
            if entry["type"] == "network":
                assert "n_networks" in entry, f"Network atlas {key} missing n_networks"
                assert entry["n_networks"] in (7, 17)

    def test_schaefer_atlases_have_n_rois(self):
        for key, entry in ATLAS_SOURCES.items():
            if key.startswith("schaefer"):
                assert "n_rois" in entry, f"{key} missing n_rois"
                assert entry["n_rois"] in (100, 400)
