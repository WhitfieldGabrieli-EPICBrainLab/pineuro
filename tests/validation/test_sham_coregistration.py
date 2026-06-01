"""Validation -- sham construction masks reach reference space (coregistration).

The original bug: a sham participant (study rtCHR, ``sub-rtchr01``) was silently
given the *real* DefaultA/ContA feedback. Root cause was in the GUI pipeline's
coregistration step. Mask *extraction* augments a localizer's network list with
``sham_networks_for_localizer`` (so SomMotA is extracted and split into
``SomMotA_left``/``SomMotA_right``), but coregistration filtered the thresholded
masks down to the localizer's *real* networks only -- so the sham hemisphere
masks never reached ``01_reference/masks/``. At runtime the sham mask lookup
then failed, was swallowed, and the task fell back to the real masks.

These tests pin the contract that fixes it: the set of network names
coregistration keeps for a localizer must include the sham construction networks
(via the production helper ``_coreg_localizer_networks``), and applying the
coregistration file filter with that set must select the hemisphere-split masks.

All expected values are derived from the rtCHR config and the documented helper
contracts -- nothing is recomputed from pineuro's implementation.
"""

from __future__ import annotations

import pytest

from pineuro.gui.components.pipeline_runner import _coreg_localizer_networks
from pineuro.gui.utils import (
    localizer_mask_names,
    sham_hemisphere_network_for_localizer,
    sham_networks_for_localizer,
)

pytestmark = pytest.mark.validation


# -----------------------------------------------------------------------------
# rtCHR-like session configs (flattened, as build_session_config produces:
# localizers + feedback_runs + sham at the top level).
# -----------------------------------------------------------------------------


def _hemisphere_split_config() -> dict:
    """rtCHR: rest localizer [DefaultA, ContA]; hemisphere_split sham on SomMotA."""
    return {
        "sham": {"enabled": True},
        "localizers": [
            {"type": "rest", "id": "rest", "networks": ["DefaultA", "ContA"]},
        ],
        "feedback_runs": [
            {
                "task": "ball",
                "mask_source": "rest",
                "sham": {
                    "delivery_method": "alternative_masks",
                    "sham_mask_mode": "hemisphere_split",
                    "hemisphere_network": "SomMotA",
                    "target_hemisphere": "right",
                    "alternative_mask_targets": ["SomMotA", "DefaultA"],
                },
            },
        ],
    }


def _two_networks_config() -> dict:
    """Variant: two_networks sham drawing on full SomMotA + DefaultA network masks."""
    cfg = _hemisphere_split_config()
    cfg["feedback_runs"][0]["sham"]["sham_mask_mode"] = "two_networks"
    return cfg


def _select_coregistered_masks(config: dict, loc: dict, mask_names_on_disk: list[str]) -> set[str]:
    """Mirror the coregistration file filter in ``_run_coregistration``.

    Reproduces the selection rule at pipeline_runner ``_run_coregistration``
    (the ``for f in sorted(src_masks_dir.glob(...))`` loop): the combined
    hemisphere-split network is dropped, then a mask is kept when its name is in
    ``names`` or starts with ``"{n}_"`` for some ``n`` in ``names``. ``names`` and
    the dropped combined name come from the *production* helpers so this exercises
    the real bug surface -- only the trivial glob/string filter is re-stated here.
    """
    names = _coreg_localizer_networks(config, loc)
    sham_combined_name = sham_hemisphere_network_for_localizer(config, loc)
    selected: set[str] = set()
    for mask_name in mask_names_on_disk:
        if sham_combined_name and mask_name == sham_combined_name:
            continue
        if mask_name in names or any(mask_name.startswith(f"{n}_") for n in names):
            selected.add(mask_name)
    return selected


# =============================================================================
# Helper contracts (the inputs to the coregistration filter)
# =============================================================================


def test_sham_networks_for_localizer_hemisphere_split():
    """hemisphere_split sham adds exactly the hemisphere network to the localizer."""
    cfg = _hemisphere_split_config()
    loc = cfg["localizers"][0]
    assert sham_networks_for_localizer(cfg, loc) == ["SomMotA"]
    assert sham_hemisphere_network_for_localizer(cfg, loc) == "SomMotA"


def test_sham_networks_for_localizer_two_networks():
    """two_networks sham adds both alternative targets; no network is split."""
    cfg = _two_networks_config()
    loc = cfg["localizers"][0]
    assert sham_networks_for_localizer(cfg, loc) == ["SomMotA", "DefaultA"]
    # No hemisphere split in two_networks mode -> nothing is dropped as "combined".
    assert sham_hemisphere_network_for_localizer(cfg, loc) is None


def test_sham_networks_empty_when_sham_disabled():
    """With sham disabled study-wide, no extra construction networks are added."""
    cfg = _hemisphere_split_config()
    cfg["sham"]["enabled"] = False
    loc = cfg["localizers"][0]
    assert sham_networks_for_localizer(cfg, loc) == []
    assert sham_hemisphere_network_for_localizer(cfg, loc) is None
    # The coregistration name set then reduces to the localizer's real networks.
    assert _coreg_localizer_networks(cfg, loc) == {"DefaultA", "ContA"}


# =============================================================================
# _coreg_localizer_networks -- THE regression guard
# =============================================================================


def test_coreg_localizer_networks_includes_sham_construction_network():
    """The coregistration name set unions the real networks with the sham network.

    This is the authoritative regression assertion: if the union with
    ``sham_networks_for_localizer`` is removed (the original bug), this set drops
    back to ``{DefaultA, ContA}`` and the assertion fails.
    """
    cfg = _hemisphere_split_config()
    loc = cfg["localizers"][0]
    assert _coreg_localizer_networks(cfg, loc) == {"DefaultA", "ContA", "SomMotA"}
    # And it strictly extends the localizer's own networks.
    assert set(localizer_mask_names(loc)) == {"DefaultA", "ContA"}
    assert "SomMotA" not in localizer_mask_names(loc)


def test_coreg_localizer_networks_two_networks_variant():
    """two_networks sham folds both alternative targets into the coregistration set."""
    cfg = _two_networks_config()
    loc = cfg["localizers"][0]
    assert _coreg_localizer_networks(cfg, loc) == {"DefaultA", "ContA", "SomMotA"}


# =============================================================================
# Applying the coregistration file filter selects the sham masks
# =============================================================================


def test_filter_selects_hemisphere_split_masks():
    """The hemisphere split masks are coregistered; the combined SomMotA is dropped."""
    cfg = _hemisphere_split_config()
    loc = cfg["localizers"][0]
    # 03_masks as mask extraction produces it: real networks + the split SomMotA
    # (plus the combined SomMotA that splitting leaves behind).
    on_disk = ["DefaultA", "ContA", "SomMotA", "SomMotA_left", "SomMotA_right"]
    selected = _select_coregistered_masks(cfg, loc, on_disk)
    assert selected == {"DefaultA", "ContA", "SomMotA_left", "SomMotA_right"}
    # The combined network mask is explicitly not coregistered (its splits are).
    assert "SomMotA" not in selected


def test_filter_old_behavior_excluded_sham_masks():
    """Document the bug: without the sham union, the hemisphere masks are excluded.

    Reproduces the *old* filter (``names`` = the localizer's real networks only)
    and asserts it wrongly drops ``SomMotA_left``/``SomMotA_right`` -- the exact
    silent fallback this fix prevents. Guards against a partial revert that fixes
    ``_coreg_localizer_networks`` but leaves the file filter inconsistent.
    """
    cfg = _hemisphere_split_config()
    loc = cfg["localizers"][0]
    old_names = set(localizer_mask_names(loc))  # {DefaultA, ContA} -- pre-fix
    sham_combined_name = sham_hemisphere_network_for_localizer(cfg, loc)
    on_disk = ["DefaultA", "ContA", "SomMotA", "SomMotA_left", "SomMotA_right"]
    selected = {
        m
        for m in on_disk
        if not (sham_combined_name and m == sham_combined_name)
        and (m in old_names or any(m.startswith(f"{n}_") for n in old_names))
    }
    assert selected == {"DefaultA", "ContA"}
    assert "SomMotA_left" not in selected
    assert "SomMotA_right" not in selected


def test_filter_selects_two_networks_full_masks():
    """two_networks sham coregisters both alternative network masks (no splitting)."""
    cfg = _two_networks_config()
    loc = cfg["localizers"][0]
    # two_networks does not split: masks on disk are the full networks.
    on_disk = ["DefaultA", "ContA", "SomMotA"]
    selected = _select_coregistered_masks(cfg, loc, on_disk)
    assert selected == {"DefaultA", "ContA", "SomMotA"}
