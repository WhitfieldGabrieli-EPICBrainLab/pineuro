"""Shared visual style tokens and layout helpers for the pineuro GUI.

Single source of truth for the GUI's look: the type scale, spacing rhythm,
semantic color tokens (tied to the Material color-scheme seed), and the
responsive form-layout helpers. Import these instead of hardcoding ``size=`` /
``width=`` / ``color=`` at each call site so every screen stays visually
consistent and the forms adapt to the window size.

Usage
-----
Text::

    from pineuro.gui import styles as st
    st.title("Study Manager")          # screen title
    st.section("Scan Parameters")      # section header
    st.caption("Optional notes ...")   # muted helper text

Responsive form rows (fields fill the column width and align across rows;
paired fields drop to one-per-line on narrow windows)::

    st.field_row(self.tr_field)                       # one field, full width
    st.field_pair(self.start_field, self.step_field)  # two fields side by side
    st.browse_row(self.base_dir_field, browse_button)  # field + trailing icon
"""

from __future__ import annotations

import contextlib

import flet as ft

# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------

#: Material color-scheme seed. All themed controls (buttons, inputs, panels,
#: dividers) and the semantic color tokens below derive from this one color,
#: so the whole palette is changed by editing a single value.
SEED_COLOR = ft.Colors.BLUE


def theme() -> ft.Theme:
    """The application :class:`ft.Theme`: seeded M3 palette + compact density."""
    return ft.Theme(color_scheme_seed=SEED_COLOR, visual_density=ft.VisualDensity.COMPACT)


# ---------------------------------------------------------------------------
# Type scale (point sizes for ``ft.Text(size=...)``)
# ---------------------------------------------------------------------------

TITLE = 22  # screen / header-bar titles
SECTION = 16  # section headers within a screen or tab
SUBSECTION = 14  # bold-ish sub-group labels (paired with FontWeight.W_500)
BODY = 14  # default body text
CAPTION = 12  # hints, helper text, metadata, timestamps

# ---------------------------------------------------------------------------
# Icon sizes (point sizes for ``ft.Icon(size=...)``)
#
# Distinct from the text type scale above: icons read larger than text at the
# same nominal size, so they get their own rungs. Most status / action icons
# are ``ICON_MD``; the XL/XXL/HERO rungs are for empty-state placeholders.
# ---------------------------------------------------------------------------

ICON_SM = 16  # inline info / status icons sitting in a text row
ICON_MD = 20  # standard status / action icons (the common case)
ICON_LG = 24  # dialog-title / prominent status icons
ICON_XL = 48  # empty-state placeholder icons (preview pending)
ICON_XXL = 64  # large empty-state placeholders (idle collection)
ICON_HERO = 100  # app landing / splash icon

# ---------------------------------------------------------------------------
# Spacing rhythm
# ---------------------------------------------------------------------------

GAP_XS = 4
GAP_SM = 8
GAP_MD = 12
GAP = 16  # between GAP_MD and GAP_LG (4pt grid; absorbs ad-hoc 15s)
GAP_LG = 20
PAD = 20  # standard screen / section / header padding

# ---------------------------------------------------------------------------
# Semantic colors
#
# These resolve against the seeded ColorScheme (see ``theme()``), so headers,
# accents, and surfaces stay coherent if ``SEED_COLOR`` changes. Prefer these
# over raw ``ft.Colors.BLUE_50`` / ``GREY_600`` literals.
# ---------------------------------------------------------------------------

HEADER_BG = ft.Colors.PRIMARY_CONTAINER  # header bars (was BLUE_50)
ON_HEADER = ft.Colors.ON_PRIMARY_CONTAINER  # text/icons on header bars
ACCENT = ft.Colors.PRIMARY  # accent borders / highlight icons (was BLUE_700)
MUTED = ft.Colors.ON_SURFACE_VARIANT  # caption / helper text (was GREY_600)
SIDEBAR_BG = ft.Colors.SURFACE_CONTAINER_HIGH  # sidebars / rails (was GREY_50)
SELECTED_BG = ft.Colors.SECONDARY_CONTAINER  # selected list row (was BLUE_100)
SUBTLE_BG = ft.Colors.SURFACE_CONTAINER_HIGHEST  # subtle panels (was GREY_100)
PANEL_BORDER = ft.Colors.OUTLINE_VARIANT  # 1px borders (was GREY_300)

# ---------------------------------------------------------------------------
# Status colors (foreground / background / border roles)
#
# Run/step status and quality readouts. ERROR uses the M3 semantic role so it
# tracks the scheme; SUCCESS / WARNING / INFO have no M3 equivalent, so they are
# fixed palette colors chosen to read on SURFACE / SUBTLE_BG. Prefer these over
# raw ft.Colors.GREEN / ORANGE / RED literals.
# ---------------------------------------------------------------------------

SUCCESS = ft.Colors.GREEN_700  # success text / icon (was GREEN / GREEN_500)
SUCCESS_FG = ft.Colors.GREEN_800  # text on SUCCESS_BG
SUCCESS_BG = ft.Colors.GREEN_50  # success panel / banner background
SUCCESS_BORDER = ft.Colors.GREEN_400

WARNING = ft.Colors.ORANGE_700  # warning text / icon (was ORANGE / ORANGE_500 / AMBER_700)
WARNING_FG = ft.Colors.ORANGE_800
WARNING_BG = ft.Colors.ORANGE_50
WARNING_BORDER = ft.Colors.ORANGE_200

ERROR = ft.Colors.ERROR  # M3 semantic error (was RED / RED_500 / RED_700)
ERROR_FG = ft.Colors.ON_ERROR_CONTAINER
ERROR_BG = ft.Colors.ERROR_CONTAINER
ERROR_BORDER = ft.Colors.ERROR

INFO = ft.Colors.BLUE_700  # informational (non-accent) text / icon
INFO_FG = ft.Colors.BLUE_800
INFO_BG = ft.Colors.BLUE_50
INFO_BORDER = ft.Colors.BLUE_200

NEUTRAL = ft.Colors.GREY_500  # idle / no-data / disabled (was GREY_400 / GREY_500)
NEUTRAL_BG = ft.Colors.GREY_200  # placeholder image background
ON_NEUTRAL = ft.Colors.WHITE  # text on a saturated status chip

#: Pipeline step-status palette, keyed by ``StepStatus`` *value* strings (not the
#: enum) to keep this module flet-only and free of a screens->styles import
#: cycle. Call sites: ``st.STEP_STATUS_COLORS[status.value]``.
STEP_STATUS_COLORS: dict[str, str] = {
    "pending": NEUTRAL,
    "running": INFO,
    "complete": SUCCESS,
    "interrupted": WARNING,
    "failed": ERROR,
    "skipped": ft.Colors.BLUE_GREY_300,  # bespoke; no semantic role
}

# ---------------------------------------------------------------------------
# Mask provenance colors (tissue / MNI-prior vs manual / alternative-atlas)
#
# Domain category colors that mark where a mask came from, distinct from run
# status. Tissue / MNI-prior masks read PURPLE; manual / alternative-atlas masks
# read TEAL. (Sham-hemisphere panels use INFO; target / non-target chips use
# SUCCESS / WARNING -- there is no third provenance family.)
# ---------------------------------------------------------------------------

MASK_TISSUE = ft.Colors.PURPLE  # tissue / MNI-prior icon + accent
MASK_TISSUE_FG = ft.Colors.PURPLE_700  # tissue header text
MASK_TISSUE_LABEL = ft.Colors.PURPLE_400  # "MNI template prior" sublabel
MASK_TISSUE_BG = ft.Colors.PURPLE_50
MASK_TISSUE_BORDER = ft.Colors.PURPLE_200

MASK_MANUAL = ft.Colors.TEAL  # manual / alternative-atlas icon + accent
MASK_MANUAL_BG = ft.Colors.TEAL_50
MASK_MANUAL_BORDER = ft.Colors.TEAL_200

# ---------------------------------------------------------------------------
# Responsive form layout
#
# Form fields are laid out in 12-column ``ResponsiveRow``s instead of carrying a
# fixed pixel ``width``. A field with no width fills its column, so every row
# shares the same edges (alignment) and the form reflows with the window
# (good on small + large screens). Breakpoints are the Flet defaults
# (sm = 576px): paired fields sit side by side on wider windows and stack on
# narrow ones, so nothing is clipped.
# ---------------------------------------------------------------------------

#: Soft cap for form readability on very wide monitors (see ``cap_form_width``).
MAX_FORM_WIDTH = 760

# Standard modal-dialog content widths (replaces the ad-hoc 300..700 spread).
DIALOG_SM = 400  # short prompts (single field, confirmation)
DIALOG_MD = 520  # typical multi-field dialogs
DIALOG_LG = 700  # wide content (tables, multi-column, setup guides)
DIALOG_XL = 800  # very wide content (feedback summary, multi-plot, QC viewer)

# Recurring component dimensions (previews, plots, progress, task mirror).
PROGRESS_BAR_W = 300  # collection / feedback progress bars
PREVIEW_H_XS = 120  # info-only overlay thumbnails (hemisphere / tissue cards)
PREVIEW_H_SM = 150  # small inline overlay / preview thumbnails
PREVIEW_H_MD = 200  # medium previews
PREVIEW_H_LG = 300  # large previews / plot panels
OVERLAY_H = 180  # mask-overlay images in the mask dialogs
DICOM_PREVIEW = 256  # DICOM middle-slice square preview (collection_progress)
TASK_MIRROR_W = 480  # task-mirror stimulus width (16:9)
TASK_MIRROR_H = 270  # task-mirror stimulus height
PLOT_IMAGE_W = 750  # rendered feedback / connectivity plot image width
MASK_OVERLAY_W = 500  # mask-overlay card / image column width

COL_FULL: dict[str, int] = {"xs": 12}  # one field per row, fills the width
COL_HALF: dict[str, int] = {"xs": 12, "sm": 6}  # two per row >=576px, else stacked

#: Fixed width for small auxiliary fields (short counts) that sit beside a
#: filling control and shouldn't stretch (e.g. a "Volumes: 2" box next to a
#: checkbox). Use sparingly — most fields should be responsive (``field_row``).
COMPACT_W = 120


def _fill_if_dropdown(control: ft.Control) -> None:
    """Make a Dropdown fill the width its parent gives it.

    A ``ft.TextField`` (and ``ft.Container``) stretches to fill its parent column
    on its own, but a ``ft.Dropdown`` (Flutter ``DropdownMenu``) sizes to its
    content unless told to expand. ``expand=True`` makes it fill — inside a
    ``ResponsiveRow`` column (full or half width) and inside a plain ``Row`` alike
    — so it lines up edge-to-edge with the text fields around it. No-op otherwise.
    """
    if isinstance(control, ft.Dropdown):
        control.expand = True


def fill_dropdown(control: ft.Control) -> ft.Control:
    """Tag *control* to fill the flex slot it sits in (e.g. inside a plain ``Row``).

    Clears any fixed width and sets ``expand`` so the control shares the row, plus
    the dropdown fill-fix. Use for dropdowns laid out by hand in an ``ft.Row``
    (the ``field_row``/``field_pair`` helpers handle their own children).
    """
    with contextlib.suppress(AttributeError):
        control.width = None  # type: ignore[attr-defined]
    control.expand = True
    _fill_if_dropdown(control)
    return control


def colspan(control: ft.Control, col: dict[str, int] | None = None) -> ft.Control:
    """Tag *control* with a responsive ``col`` span and drop any fixed width."""
    control.col = col or COL_FULL  # type: ignore[assignment]
    with contextlib.suppress(AttributeError):
        control.width = None  # type: ignore[attr-defined]  # let the column decide
    _fill_if_dropdown(control)
    return control


def field_row(*controls: ft.Control, col: dict[str, int] | None = None) -> ft.ResponsiveRow:
    """One (or more) full-width field(s) as a ``ResponsiveRow``.

    Each control fills the row (``COL_FULL``) unless *col* overrides it. Use for
    the single-field-per-line case so the field fills the form width and lines
    up with the rows above and below it.
    """
    return ft.ResponsiveRow(
        [colspan(c, col) for c in controls],
        run_spacing=GAP_SM,
        spacing=GAP_SM,
    )


def field_pair(left: ft.Control, right: ft.Control) -> ft.ResponsiveRow:
    """Two fields sharing a row (``COL_HALF`` each); stacks on narrow windows."""
    return ft.ResponsiveRow(
        [colspan(left, COL_HALF), colspan(right, COL_HALF)],
        run_spacing=GAP_SM,
        spacing=GAP_SM,
    )


def browse_row(field_control: ft.Control, *trailing: ft.Control) -> ft.Row:
    """A text field (or dropdown) that fills the row, followed by trailing button(s)."""
    fill_dropdown(field_control)
    return ft.Row(
        [field_control, *trailing],
        vertical_alignment=ft.CrossAxisAlignment.CENTER,
        spacing=GAP_SM,
    )


def wrap(control: ft.Control) -> ft.Row:
    """Wrap a control in a ``Row`` so a parent ``Column`` won't stretch it.

    Replaces the per-file ``_f`` helpers. Prefer :func:`field_row` for form
    fields; this remains for non-field controls that should keep intrinsic size.
    """
    return ft.Row([control])


def cap_form_width(page: ft.Page, container: ft.Container, *, reserved: float = 0.0):
    """Return an ``on_resize`` handler that caps *container* at ``MAX_FORM_WIDTH``.

    Keeps form fields readable on very wide monitors while still shrinking on
    small windows: the container width tracks ``min(available, MAX_FORM_WIDTH)``.
    *reserved* is the horizontal chrome (sidebar, dividers, padding) to subtract
    from the page width. Call the returned handler once after building to apply
    the initial width.
    """

    def _apply(_event: object = None) -> None:
        avail = (page.width or 1400) - reserved
        container.width = max(320.0, min(float(avail), float(MAX_FORM_WIDTH)))
        with contextlib.suppress(Exception):
            container.update()

    return _apply


# ---------------------------------------------------------------------------
# Text factories
# ---------------------------------------------------------------------------


def title(text: str, **kwargs: object) -> ft.Text:
    """Screen / header-bar title (size ``TITLE``, bold)."""
    return ft.Text(text, size=TITLE, weight=ft.FontWeight.BOLD, **kwargs)  # type: ignore[arg-type]


def section(text: str, **kwargs: object) -> ft.Text:
    """Section header (size ``SECTION``, bold)."""
    return ft.Text(text, size=SECTION, weight=ft.FontWeight.BOLD, **kwargs)  # type: ignore[arg-type]


def subsection(text: str, **kwargs: object) -> ft.Text:
    """Sub-group label (size ``SUBSECTION``, medium weight)."""
    return ft.Text(text, size=SUBSECTION, weight=ft.FontWeight.W_500, **kwargs)  # type: ignore[arg-type]


def caption(text: str, **kwargs: object) -> ft.Text:
    """Muted helper / hint text (size ``CAPTION``, italic, ``MUTED`` color)."""
    kwargs.setdefault("color", MUTED)
    kwargs.setdefault("italic", True)
    return ft.Text(text, size=CAPTION, **kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Summary-card factories
# ---------------------------------------------------------------------------


def stat_row(
    label: str,
    value: str,
    *,
    value_color: str | None = None,
    value_weight: ft.FontWeight = ft.FontWeight.W_500,
    label_weight: ft.FontWeight | None = None,
    label_color: str | None = MUTED,
) -> ft.Row:
    """A label/value stat line: muted label left, weighted value right, space-between.

    The default (muted label, ``W_500`` value, no value color) covers the common
    ``Mean: 0.12 mm`` stat rows. Override per row:

    - ``value_color`` — colored values (the ">threshold" count, quality verdict).
    - ``value_weight=ft.FontWeight.BOLD`` — emphasized values (tSNR "Final", the
      quality verdict, the metric "Total").
    - ``label_weight=ft.FontWeight.W_500`` + ``label_color=None`` — the
      "Quality:" / "Total:" rows whose label is weighted, not muted.

    Indentation (e.g. the ``"  Hits:"`` per-item rows) lives in *label*: pass the
    leading spaces in the string, because it is content, not style.
    """
    label_kwargs: dict[str, object] = {"size": BODY, "color": label_color}
    if label_weight is not None:
        label_kwargs["weight"] = label_weight
    value_kwargs: dict[str, object] = {"size": BODY, "weight": value_weight}
    if value_color is not None:
        value_kwargs["color"] = value_color
    return ft.Row(
        [
            ft.Text(label, **label_kwargs),  # type: ignore[arg-type]
            ft.Text(value, **value_kwargs),  # type: ignore[arg-type]
        ],
        alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
    )


def card_shell(
    icon: ft.IconData,
    title: str,
    *body: ft.Control,
    icon_color: str = ACCENT,
    width: float = 170,
) -> ft.Container:
    """Standard summary-card chrome: a bordered ``SUBTLE_BG`` panel with an
    icon+title header, a small spacer, then *body* controls in a tight column.

    ``width`` defaults to the QC-card width (170); pass ``width=200`` for the
    wider task-metrics card. ``icon_color`` defaults to ``ACCENT`` (the metric
    card); the QC cards pass their quality color so the icon tints with the
    verdict.
    """
    return ft.Container(
        content=ft.Column(
            [
                ft.Row(
                    [
                        ft.Icon(icon, color=icon_color, size=ICON_MD),
                        ft.Text(title, size=SUBSECTION, weight=ft.FontWeight.BOLD),
                    ],
                    spacing=GAP_SM,
                ),
                ft.Container(height=GAP_SM),
                *body,
            ],
            spacing=GAP_XS,
        ),
        padding=GAP,
        bgcolor=SUBTLE_BG,
        border_radius=8,
        border=ft.Border.all(1, PANEL_BORDER),
        width=width,
    )
