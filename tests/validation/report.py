"""Render the validation suite's recorded metrics into a human-readable report.

Produces a markdown report, a standalone HTML report, and a per-stage
pass/fail summary figure. Pure stdlib + matplotlib (Agg backend) — no jinja2.
The metrics JSON written alongside is the durable artifact; this report can be
regenerated from it via :func:`generate_from_json`.

The report is written for a neuroimaging researcher who has never seen pineuro:
an intro and glossary up front, stages in pipeline order each with a plain
description, and per-test plain-language explanations. Internal code names and
linear-algebra jargon are kept out.
"""

from __future__ import annotations

import html
import json
from datetime import datetime
from pathlib import Path

# Operator symbol -> plain-English phrase for the threshold check.
_OP_PHRASE = {
    ">": "must exceed",
    ">=": "must be at least",
    "<": "must stay below",
    "<=": "must be at most",
    "==": "must equal exactly",
}

# Acronyms given their canonical casing when a metric slug becomes a display label.
_ACRONYMS = {
    "glm": "GLM",
    "fd": "FD",
    "dvars": "DVARS",
    "tsnr": "tSNR",
    "hrf": "HRF",
    "ica": "ICA",
    "roi": "ROI",
    "qc": "QC",
    "sd": "SD",
    "svd": "SVD",
}

# Opening paragraph — what the suite is and how to read it.
_INTRO = (
    "This report checks that pineuro analyses fMRI data correctly, not just that "
    "the code runs. Each test builds synthetic brain images with a known, built-in "
    "answer, runs them through the real pineuro code, and measures how far the "
    "result strays from that answer. A check passes when its measurement meets the "
    "target; for most checks the target is an error of zero, so a measured 0 is the "
    "expected, healthy result. Values far below a millionth are floating-point "
    "rounding, not real disagreement."
)

# Stage display metadata: each stage key -> heading, pipeline-order index, and a
# one-line plain description. Stages render in `order`, not alphabetically.
_STAGE_INFO: dict[str, dict] = {
    "phantom": {
        "order": 0,
        "name": "Synthetic test data",
        "blurb": "The synthetic brain images the other tests run on. These checks confirm the "
        "images carry the signal, noise, and HRF shape those tests rely on.",
    },
    "preprocessing": {
        "order": 1,
        "name": "Image preprocessing",
        "blurb": "Motion correction, brain extraction, averaging over time, and applying spatial "
        "transforms to the fMRI images.",
    },
    "mask_extraction": {
        "order": 2,
        "name": "Region-of-interest extraction",
        "blurb": "Locating the brain regions used for feedback, either from a task GLM or from "
        "independent component analysis.",
    },
    "hrf": {
        "order": 3,
        "name": "Hemodynamic response function",
        "blurb": "The model of how the BOLD signal rises and falls after neural activity. It "
        "underlies the task designs and the feedback model.",
    },
    "design": {
        "order": 4,
        "name": "Task design regressors",
        "blurb": "The expected-activation timecourses for block and event-related task designs.",
    },
    "feedback_glm": {
        "order": 5,
        "name": "Real-time feedback: the GLM solver",
        "blurb": "The general linear model at the core of the real-time estimator, and how it "
        "combines the voxels within a region.",
    },
    "feedback": {
        "order": 6,
        "name": "Real-time feedback: signal recovery",
        "blurb": "The real-time estimator recovering each region's activation from incoming "
        "volumes, with its head-motion and image-quality metrics.",
    },
    "smoothing": {
        "order": 7,
        "name": "Activation smoothing and scaling",
        "blurb": "Smoothing the activation signal over time and rescaling it before it reaches the participant.",
    },
    "connectivity": {
        "order": 8,
        "name": "Functional connectivity",
        "blurb": "The sliding-window correlation between two regions, used for connectivity-based feedback.",
    },
    "collection_qc": {
        "order": 9,
        "name": "Scan quality monitoring",
        "blurb": "The live quality-control monitor: head-motion and image-stability metrics computed as the scan runs.",
    },
    "qc_helpers": {
        "order": 10,
        "name": "Quality-control report helpers",
        "blurb": "Supporting calculations for the QC report: mask volumes, centroids, and slice selection for figures.",
    },
    "sham": {
        "order": 11,
        "name": "Sham (placebo) feedback",
        "blurb": "Sham mode, which replays a previous participant's recorded feedback instead of computing it live.",
    },
    "pipeline": {
        "order": 12,
        "name": "End-to-end pipeline",
        "blurb": "The whole chain run together, from preprocessing through real-time feedback, "
        "confirming the stages hand off correctly.",
    },
}


def _stage_meta(stage: str) -> dict:
    """Display metadata for a stage key, with a fallback for an unknown key."""
    return _STAGE_INFO.get(
        stage,
        {"order": 99, "name": stage.replace("_", " ").capitalize(), "blurb": ""},
    )


def _test_func_name(nodeid: str) -> str:
    """The bare test function name — ``file::func[param]`` -> ``func``.

    Strips both the file prefix and any ``[param]`` suffix so the parametrized
    cases of one test merge under a single block in the report.
    """
    tail = nodeid.rsplit("::", 1)[-1]
    return tail.split("[", 1)[0]


def _test_title(test_records: list[dict]) -> str:
    """Heading for a test block — the first sentence of its docstring.

    Falls back to a humanized function name when a test has no docstring.
    """
    doc = str(test_records[0].get("test_doc", "")).strip()
    if doc:
        first = doc.split(". ")[0].rstrip(".")
        if first:
            return first[0].upper() + first[1:]
    name = _test_func_name(test_records[0]["test"])
    name = name[5:] if name.startswith("test_") else name
    return name.replace("_", " ").capitalize()


def _humanize_metric(slug: str) -> str:
    """Turn a metric slug into a readable label, casing known acronyms."""
    return " ".join(_ACRONYMS.get(w, w) for w in slug.split("_"))


def _stage_anchor(heading: str) -> str:
    """GitHub-style heading slug, for a markdown table-of-contents link."""
    slug = "".join(c for c in heading.lower() if c.isalnum() or c in " -")
    return slug.replace(" ", "-")


def _escape_leading_underscore(text: str) -> str:
    """Stop a leading underscore from being read as markdown emphasis."""
    return "\\" + text if text.startswith("_") else text


def _result_text(passed: bool | None) -> str:
    if passed is None:
        return "INFO"
    return "PASS" if passed else "FAIL"


def _fmt_number(x: float) -> str:
    """Format a metric value readably: integers plain, tiny/huge in sci notation."""
    if x != x:  # NaN
        return "NaN"
    if x in (float("inf"), float("-inf")):
        return "inf" if x > 0 else "-inf"
    ax = abs(x)
    if ax == 0:
        return "0"
    if ax == int(ax) and ax < 1e6:  # exact integer-valued metric
        return str(int(x))
    if ax < 1e-4 or ax >= 1e6:  # machine-precision bounds, large counts
        return f"{x:.2e}"
    return f"{x:.4g}"


def _value_phrase(record: dict) -> str:
    """``<value> <units>`` trimmed."""
    return f"{_fmt_number(record['value'])} {record.get('units', '')}".strip()


def _check_phrase(record: dict) -> str:
    """Plain-English restatement of the threshold check, or '' if info-only."""
    if record["threshold"] is None:
        return ""
    op = _OP_PHRASE.get(record["op"], record["op"])
    thr = f"{_fmt_number(record['threshold'])} {record.get('units', '')}".strip()
    return f"{op} {thr}"


def _group_by_stage(records: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for record in records:
        grouped.setdefault(record["stage"], []).append(record)
    return grouped


def _group_by_test(records: list[dict]) -> dict[str, list[dict]]:
    """Group a stage's records by test function name (parametrized cases merge)."""
    grouped: dict[str, list[dict]] = {}
    for record in records:
        grouped.setdefault(_test_func_name(record["test"]), []).append(record)
    return grouped


def _counts(records: list[dict]) -> tuple[int, int, int]:
    total = sum(1 for r in records if r["passed"] is not None)
    passed = sum(1 for r in records if r["passed"])
    n_tests = len({r["test"] for r in records})
    return passed, total, n_tests


def _collapsible(records: list[dict]) -> bool:
    """True when the records are parametrized cases of one test.

    Parametrized cases share one description and one metric meaning, and each
    comes from a distinct pytest node-id. A test with several *different*
    metrics keeps one node-id across all of them, so it does not collapse.
    """
    if len(records) < 2:
        return False
    one_description = len({r.get("description", "") for r in records}) == 1
    distinct_nodes = len({r["test"] for r in records}) == len(records)
    return one_description and distinct_nodes


def _collapse_summary(records: list[dict]) -> tuple[str, str]:
    """``(verdict, summary_phrase)`` for a collapsed group of parametrized cases."""
    n = len(records)
    n_thr = sum(1 for r in records if r["passed"] is not None)
    passed = sum(1 for r in records if r["passed"])
    worst = _fmt_number(max(r["value"] for r in records))
    if passed == n_thr:
        return "PASS", f"all {n} pass (largest measured value {worst})"
    return "FAIL", f"{passed} of {n} pass (largest measured value {worst})"


# =============================================================================
# Markdown
# =============================================================================


def _metric_lines_md(record: dict) -> list[str]:
    """A metric as a bullet: the plain description, then the measurement beneath."""
    verdict = _result_text(record["passed"])
    check = _check_phrase(record)
    value = _value_phrase(record)
    if check:
        measure = f"Measured {value}; {check}. **{verdict}**"
    else:
        measure = f"Measured {value} (informational)."
    lead = record.get("description", "").strip() or _humanize_metric(record["metric"])
    return [f"- {_escape_leading_underscore(lead)}", f"  {measure}"]


def _collapsed_lines_md(records: list[dict]) -> list[str]:
    """One bullet standing in for a whole parametrized test's repeated cases."""
    verdict, summary = _collapse_summary(records)
    measure = f"Checked across {len(records)} input cases; {summary}. **{verdict}**"
    lead = records[0].get("description", "").strip() or _humanize_metric(records[0]["metric"])
    return [f"- {_escape_leading_underscore(lead)}", f"  {measure}"]


def _test_block_md(test_records: list[dict]) -> list[str]:
    lines = [f"### {_test_title(test_records)}", ""]
    if _collapsible(test_records):
        lines += _collapsed_lines_md(test_records)
    else:
        for record in test_records:
            lines += _metric_lines_md(record)
    lines.append("")
    return lines


def _markdown(records: list[dict], figure: Path | None) -> str:
    passed, total, n_tests = _counts(records)
    by_stage = _group_by_stage(records)
    ordered = sorted(by_stage, key=lambda s: (_stage_meta(s)["order"], s))

    lines = [
        "# pineuro Validation Report",
        "",
        f"_Generated {datetime.now().isoformat(timespec='seconds')}_",
        "",
        f"**{passed} of {total} checks passed** across {n_tests} tests.",
        "",
        _INTRO,
        "",
    ]
    if figure is not None:
        lines += [f"![per-stage pass/fail summary]({figure.parent.name}/{figure.name})", ""]

    lines += ["## Contents", "", "| Section | Checks passed |", "| --- | --- |"]
    for i, stage in enumerate(ordered, 1):
        heading = f"{i}. {_stage_meta(stage)['name']}"
        s_passed, s_total, _ = _counts(by_stage[stage])
        lines.append(f"| [{heading}](#{_stage_anchor(heading)}) | {s_passed} / {s_total} |")
    lines.append("")

    for i, stage in enumerate(ordered, 1):
        meta = _stage_meta(stage)
        lines += [f"## {i}. {meta['name']}", ""]
        if meta["blurb"]:
            lines += [meta["blurb"], ""]
        for test_records in _group_by_test(by_stage[stage]).values():
            lines += _test_block_md(test_records)
    return "\n".join(lines)


# =============================================================================
# HTML
# =============================================================================

_HTML_STYLE = """
body { font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 2rem auto; max-width: 880px;
       color: #1a1a1a; line-height: 1.5; }
h1 { margin-bottom: 0.2rem; }
h2 { margin-top: 2.2rem; border-bottom: 1px solid #ddd; padding-bottom: 0.2rem; }
h3 { margin-top: 1.3rem; margin-bottom: 0.2rem; font-size: 16px; }
p.intro, p.blurb { color: #333; }
p.blurb { margin: 0.2rem 0 0.6rem; }
table { border-collapse: collapse; margin: 0.5rem 0; }
th, td { border: 1px solid #ddd; padding: 5px 10px; text-align: left; font-size: 14px; }
th { background: #f4f4f4; }
ul.metrics { margin: 0.4rem 0 1rem; padding-left: 1.2rem; }
ul.metrics li { margin-bottom: 0.6rem; }
.measure { color: #555; font-size: 13px; margin-top: 0.15rem; }
.PASS { color: #2e9e4f; font-weight: 600; } .FAIL { color: #d64545; font-weight: 600; }
.INFO { color: #888; font-weight: 600; }
img { max-width: 720px; margin: 1rem 0; }
""".strip()


def _metric_li_html(record: dict) -> str:
    value = html.escape(_value_phrase(record))
    verdict = _result_text(record["passed"])
    check = _check_phrase(record)
    if check:
        measure = f"Measured {value}; {html.escape(check)}. "
    else:
        measure = f"Measured {value} (informational). "
    measure += f'<span class="{verdict}">{verdict}</span>'
    lead = record.get("description", "").strip() or _humanize_metric(record["metric"])
    return f'<li>{html.escape(lead)}<div class="measure">{measure}</div></li>'


def _collapsed_li_html(records: list[dict]) -> str:
    verdict, summary = _collapse_summary(records)
    measure = (
        f'Checked across {len(records)} input cases; {html.escape(summary)}. <span class="{verdict}">{verdict}</span>'
    )
    lead = records[0].get("description", "").strip() or _humanize_metric(records[0]["metric"])
    return f'<li>{html.escape(lead)}<div class="measure">{measure}</div></li>'


def _test_block_html(test_records: list[dict]) -> list[str]:
    parts = [f"<h3>{html.escape(_test_title(test_records))}</h3>", '<ul class="metrics">']
    if _collapsible(test_records):
        parts.append(_collapsed_li_html(test_records))
    else:
        parts += [_metric_li_html(record) for record in test_records]
    parts.append("</ul>")
    return parts


def _html(records: list[dict], figure: Path | None) -> str:
    passed, total, n_tests = _counts(records)
    by_stage = _group_by_stage(records)
    ordered = sorted(by_stage, key=lambda s: (_stage_meta(s)["order"], s))

    parts = [
        "<!DOCTYPE html>",
        '<html lang="en"><head><meta charset="utf-8">',
        "<title>pineuro Validation Report</title>",
        f"<style>{_HTML_STYLE}</style></head><body>",
        "<h1>pineuro Validation Report</h1>",
        f"<p><em>Generated {html.escape(datetime.now().isoformat(timespec='seconds'))}</em></p>",
        f"<p><strong>{passed} of {total} checks passed</strong> across {n_tests} tests.</p>",
        f'<p class="intro">{html.escape(_INTRO)}</p>',
    ]
    if figure is not None:
        parts.append(f'<img src="{figure.parent.name}/{figure.name}" alt="per-stage pass/fail summary">')

    parts.append("<h2>Contents</h2><table><tr><th>Section</th><th>Checks passed</th></tr>")
    for i, stage in enumerate(ordered, 1):
        s_passed, s_total, _ = _counts(by_stage[stage])
        name = html.escape(f"{i}. {_stage_meta(stage)['name']}")
        parts.append(f'<tr><td><a href="#stage-{i}">{name}</a></td><td>{s_passed} / {s_total}</td></tr>')
    parts.append("</table>")

    for i, stage in enumerate(ordered, 1):
        meta = _stage_meta(stage)
        parts.append(f'<h2 id="stage-{i}">{i}. {html.escape(meta["name"])}</h2>')
        if meta["blurb"]:
            parts.append(f'<p class="blurb">{html.escape(meta["blurb"])}</p>')
        for test_records in _group_by_test(by_stage[stage]).values():
            parts += _test_block_html(test_records)
    parts.append("</body></html>")
    return "\n".join(parts)


# =============================================================================
# Summary figure
# =============================================================================


def _summary_figure(records: list[dict], figures_dir: Path) -> Path | None:
    """Per-stage stacked pass/fail bar chart; returns None if nothing to plot."""
    thresholded = [r for r in records if r["passed"] is not None]
    if not thresholded:
        return None

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stages: dict[str, dict[str, int]] = {}
    for record in thresholded:
        counts = stages.setdefault(record["stage"], {"pass": 0, "fail": 0})
        counts["pass" if record["passed"] else "fail"] += 1

    keys = sorted(stages, key=lambda s: (_stage_meta(s)["order"], s))
    labels = [_stage_meta(s)["name"] for s in keys]
    passes = [stages[s]["pass"] for s in keys]
    fails = [stages[s]["fail"] for s in keys]
    positions = list(range(len(keys)))

    fig, ax = plt.subplots(figsize=(8.0, 0.5 * len(keys) + 1.6))
    ax.barh(positions, passes, color="#2e9e4f", label="pass")
    ax.barh(positions, fails, left=passes, color="#d64545", label="fail")
    ax.set_yticks(positions)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("checks")
    ax.set_title("pineuro validation — per-stage pass / fail")
    ax.legend(loc="lower right")
    fig.tight_layout()

    path = figures_dir / "summary.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def generate(records: list[dict], out_dir: Path | str) -> dict[str, Path | None]:
    """Write ``validation_report.md``, ``validation_report.html`` and the figure."""
    out_dir = Path(out_dir)
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    figure = _summary_figure(records, figures_dir)
    md_path = out_dir / "validation_report.md"
    md_path.write_text(_markdown(records, figure), encoding="utf-8")
    html_path = out_dir / "validation_report.html"
    html_path.write_text(_html(records, figure), encoding="utf-8")
    return {"markdown": md_path, "html": html_path, "figure": figure}


def generate_from_json(json_path: Path | str, out_dir: Path | str) -> dict[str, Path | None]:
    """Regenerate the report from a previously written ``validation_metrics.json``."""
    records = json.loads(Path(json_path).read_text(encoding="utf-8"))
    return generate(records, out_dir)
