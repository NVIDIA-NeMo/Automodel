#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Aggregate Titans evaluation artifacts and render parity figures and a report."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import logging
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)

CAVEATS = (
    "No compatible parameter-matched floor or full-attention control is available; "
    "the figures compare Titans architectures and the within-checkpoint TTT on/off intervention only.",
    "S-NIAH-N / essay at context length 512 is unavailable and is not interpolated or otherwise imputed.",
    "The co-author protocol established n=300 cells as three 100-sample seeds. "
    "These AutoModel runs contain 300 deterministic samples per cell, but their manifests do not establish "
    "three-seed provenance; treat them as protocol-aligned rather than bit-exact replications.",
)
RULER_FIELDS = (
    "architecture",
    "ttt_mode",
    "task_or_benchmark",
    "context_length",
    "metric",
    "value",
    "unit",
    "sample_count",
    "token_count",
    "checkpoint",
    "code_revision",
    "valid",
    "complete",
    "status",
    "source",
    "artifact",
)
COLORS = ("#0b7285", "#e8590c", "#5c940d", "#9c36b5", "#2a78d6", "#c92a2a")


@dataclass(frozen=True)
class ReportRow:
    """One tidy metric observation with its evaluation provenance."""

    architecture: str
    ttt_mode: str
    task_or_benchmark: str
    context_length: int | None
    metric: str
    value: float | int | None
    unit: str
    sample_count: int | None
    token_count: int | None
    checkpoint: str | None
    code_revision: str | None
    valid: bool
    complete: bool
    status: str
    source: str
    artifact: str


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ttt_mode(label: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
    if normalized in {"on", "enabled", "enable", "ttt_on", "writes_on"}:
        return "on"
    if normalized in {"off", "disabled", "disable", "ttt_off", "writes_off", "no_ttt"}:
        return "off"
    return label


def _jsonl_count(path: Path) -> tuple[int, list[int] | None]:
    count = 0
    ordinals: list[int] = []
    with path.open() as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            count += 1
            ordinal = row.get("_sample_ordinal")
            if isinstance(ordinal, int):
                ordinals.append(ordinal)
            else:
                ordinals = []
    return count, ordinals or None


def _ruler_location(root: Path, summary_path: Path) -> tuple[str, str, int]:
    relative = summary_path.relative_to(root)
    parts = relative.parts
    if len(parts) < 5 or parts[-2:] != ("pred", "summary.csv"):
        raise ValueError(f"{summary_path}: expected <ttt_mode>/<architecture>/<context>/pred/summary.csv below {root}")
    try:
        context_length = int(parts[-3])
    except ValueError as error:
        raise ValueError(f"{summary_path}: context directory must be an integer") from error
    return _ttt_mode(parts[-5]), parts[-4], context_length


def _ruler_task_row(
    *,
    summary_path: Path,
    architecture: str,
    ttt_mode: str,
    context_length: int,
    summary_row: dict[str, str],
) -> ReportRow:
    task = summary_row.get("task", "")
    issues: list[str] = []
    value: float | None = None
    samples: int | None = None
    try:
        value = float(summary_row["score"])
        if not math.isfinite(value) or not 0 <= value <= 100:
            raise ValueError
    except (KeyError, ValueError):
        issues.append("invalid score")
    try:
        samples = int(summary_row["num_samples"])
        if samples < 0:
            raise ValueError
    except (KeyError, ValueError):
        issues.append("invalid num_samples")

    merged_path = summary_path.parent / f"{task}.jsonl"
    manifest_path = Path(f"{merged_path}.manifest.json")
    manifest: dict[str, Any] = {}
    if not task:
        issues.append("missing task")
    if not merged_path.is_file():
        issues.append("missing merged predictions")
    if not manifest_path.is_file():
        issues.append("missing merged manifest")
    if manifest_path.is_file():
        try:
            manifest = _read_json(manifest_path)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            issues.append(f"invalid manifest: {error}")

    if manifest:
        expected_mode = ttt_mode == "on" if ttt_mode in {"on", "off"} else None
        settings = manifest.get("generation_settings", {})
        checks = {
            "task": task,
            "context_length": context_length,
        }
        for key, expected in checks.items():
            if manifest.get(key) != expected:
                issues.append(f"manifest {key} mismatch")
        if expected_mode is not None and settings.get("enable_ttt_updates") is not expected_mode:
            issues.append("manifest TTT mode mismatch")

    merged_count: int | None = None
    ordinals: list[int] | None = None
    if merged_path.is_file():
        try:
            merged_count, ordinals = _jsonl_count(merged_path)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            issues.append(f"invalid merged predictions: {error}")
    requested = manifest.get("requested_sample_count") if manifest else None
    complete = (
        not issues
        and samples is not None
        and merged_count == samples
        and isinstance(requested, int)
        and samples == requested
        and ordinals == list(range(requested))
    )
    valid = not issues
    if valid and not complete:
        status = "valid; sample count or ordinal coverage is incomplete"
    elif valid:
        status = "valid and complete"
    else:
        status = "; ".join(issues)
    return ReportRow(
        architecture=architecture,
        ttt_mode=ttt_mode,
        task_or_benchmark=task,
        context_length=context_length,
        metric="accuracy",
        value=value,
        unit="percent",
        sample_count=samples,
        token_count=None,
        checkpoint=manifest.get("checkpoint"),
        code_revision=manifest.get("code_git_revision"),
        valid=valid,
        complete=complete,
        status=status,
        source="ruler",
        artifact=str(summary_path),
    )


def collect_ruler(roots: list[Path]) -> list[ReportRow]:
    """Collect RULER summary rows and validate their merged manifests."""
    rows: list[ReportRow] = []
    for root in roots:
        for summary_path in sorted(root.rglob("summary.csv")):
            try:
                mode, architecture, context_length = _ruler_location(root, summary_path)
            except ValueError as error:
                LOGGER.warning("%s", error)
                continue
            with summary_path.open(newline="") as stream:
                for summary_row in csv.DictReader(stream):
                    rows.append(
                        _ruler_task_row(
                            summary_path=summary_path,
                            architecture=architecture,
                            ttt_mode=mode,
                            context_length=context_length,
                            summary_row=summary_row,
                        )
                    )
    return rows


def _lm_location(root: Path, summary_path: Path) -> tuple[str, str, str]:
    parts = summary_path.relative_to(root).parts
    if len(parts) < 4 or parts[-3] != "lm" or not parts[-1].endswith(".summary.json"):
        raise ValueError(f"{summary_path}: expected <ttt_mode>/lm/<architecture>/<benchmark>.summary.json below {root}")
    benchmark = parts[-1][: -len(".summary.json")]
    return _ttt_mode(parts[-4]), parts[-2], benchmark


def _lm_integrity(
    summary_path: Path, summary: dict[str, Any], manifest: dict[str, Any], mode: str, benchmark: str
) -> tuple[bool, bool, str]:
    issues: list[str] = []
    if summary.get("benchmark") != benchmark or manifest.get("benchmark") != benchmark:
        issues.append("benchmark mismatch")
    metadata = summary.get("metadata")
    if not isinstance(metadata, dict):
        issues.append("missing summary metadata")
        metadata = {}
    expected_mode = mode == "on" if mode in {"on", "off"} else None
    if expected_mode is not None and metadata.get("enable_ttt_updates") is not expected_mode:
        issues.append("metadata TTT mode mismatch")

    summary_entry = manifest.get("summary")
    if not isinstance(summary_entry, dict):
        issues.append("missing summary manifest entry")
    else:
        if summary_entry.get("path") != summary_path.name:
            issues.append("summary path mismatch")
        if summary_entry.get("sha256") != _sha256(summary_path):
            issues.append("summary hash mismatch")

    source_indices = summary.get("source_indices")
    metrics = summary.get("metrics")
    if not isinstance(source_indices, list) or not all(isinstance(index, int) for index in source_indices):
        issues.append("invalid source indices")
        source_indices = []
    if not isinstance(metrics, dict):
        issues.append("missing metrics")
        metrics = {}
    sample_count = metrics.get("samples")
    partials = manifest.get("partials")
    partial_rows = None
    if isinstance(partials, list) and all(isinstance(partial, dict) for partial in partials):
        try:
            partial_rows = sum(int(partial["rows"]) for partial in partials)
        except (KeyError, TypeError, ValueError):
            issues.append("invalid partial row counts")
        for partial in partials:
            relative_path = partial.get("path")
            if not isinstance(relative_path, str):
                issues.append("invalid partial path")
                continue
            partial_path = summary_path.parent / relative_path
            if not partial_path.is_file():
                issues.append(f"missing partial {relative_path}")
                continue
            if partial.get("sha256") != _sha256(partial_path):
                issues.append(f"partial hash mismatch: {relative_path}")
            actual_rows = sum(bool(line.strip()) for line in partial_path.read_text().splitlines())
            if partial.get("rows") != actual_rows:
                issues.append(f"partial row count mismatch: {relative_path}")
    else:
        issues.append("invalid partial manifest")
    complete = (
        not issues
        and isinstance(sample_count, int)
        and sample_count == len(source_indices)
        and sample_count == partial_rows
        and source_indices == list(range(sample_count))
    )
    valid = not issues
    if valid and not complete:
        status = "valid; merged source coverage is incomplete"
    elif valid:
        status = "valid and complete"
    else:
        status = "; ".join(issues)
    return valid, complete, status


def collect_lm(roots: list[Path]) -> list[ReportRow]:
    """Collect standard-LM metrics and verify summary-to-manifest integrity."""
    rows: list[ReportRow] = []
    for root in roots:
        for summary_path in sorted(root.rglob("*.summary.json")):
            try:
                mode, architecture, benchmark = _lm_location(root, summary_path)
            except ValueError as error:
                LOGGER.warning("%s", error)
                continue
            manifest_path = summary_path.with_name(f"{benchmark}.manifest.json")
            try:
                summary = _read_json(summary_path)
            except (OSError, ValueError, json.JSONDecodeError) as error:
                LOGGER.warning("%s: %s", summary_path, error)
                continue
            manifest: dict[str, Any] = {}
            if manifest_path.is_file():
                try:
                    manifest = _read_json(manifest_path)
                except (OSError, ValueError, json.JSONDecodeError):
                    pass
            valid, complete, status = _lm_integrity(summary_path, summary, manifest, mode, benchmark)
            metrics = summary.get("metrics", {})
            metadata = summary.get("metadata", {})
            sample_count = metrics.get("samples") if isinstance(metrics.get("samples"), int) else None
            token_count = metrics.get("token_count") if isinstance(metrics.get("token_count"), int) else None
            metric_units = {
                "perplexity": "perplexity",
                "bits_per_byte": "bits/byte",
                "last_word_exact_accuracy": "fraction",
            }
            for metric, unit in metric_units.items():
                value = metrics.get(metric)
                if value is None:
                    continue
                if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                    value = None
                    valid = False
                    status = f"{status}; invalid {metric}".strip("; ")
                elif metric in {"perplexity", "bits_per_byte"} and value < 0:
                    value = None
                    valid = False
                    status = f"{status}; invalid {metric}".strip("; ")
                elif metric == "last_word_exact_accuracy" and not 0 <= value <= 1:
                    value = None
                    valid = False
                    status = f"{status}; invalid {metric}".strip("; ")
                rows.append(
                    ReportRow(
                        architecture=architecture,
                        ttt_mode=mode,
                        task_or_benchmark=benchmark,
                        context_length=None,
                        metric=metric,
                        value=value,
                        unit=unit,
                        sample_count=sample_count,
                        token_count=token_count,
                        checkpoint=metadata.get("checkpoint"),
                        code_revision=metadata.get("code_git_revision"),
                        valid=valid,
                        complete=complete,
                        status=status,
                        source="lm",
                        artifact=str(summary_path),
                    )
                )
    return rows


def evaluation_caveats(rows: list[ReportRow]) -> list[str]:
    """Return static caveats plus data-dependent scientific warnings."""
    caveats = list(CAVEATS)
    paired: dict[tuple[Any, ...], dict[str, float | int]] = {}
    for row in rows:
        if row.ttt_mode not in {"on", "off"} or not row.valid or not row.complete or row.value is None:
            continue
        key = (row.source, row.architecture, row.task_or_benchmark, row.context_length, row.metric)
        paired.setdefault(key, {})[row.ttt_mode] = row.value
    matched = [modes for modes in paired.values() if modes.keys() == {"on", "off"}]
    if matched and all(modes["on"] == modes["off"] for modes in matched):
        caveats.append(
            "Every matched aggregate metric is exactly identical with TTT updates on and off. "
            "Treat the dashed ablation curves as coincident, not as independent evidence of a TTT benefit."
        )
    return caveats


def write_tidy_outputs(rows: list[ReportRow], output_dir: Path, roots: dict[str, list[Path]]) -> tuple[Path, Path]:
    """Write deterministic tidy CSV and JSON aggregate files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    ordered = sorted(
        rows,
        key=lambda row: (
            row.source,
            row.task_or_benchmark,
            row.architecture,
            row.ttt_mode,
            row.context_length if row.context_length is not None else -1,
            row.metric,
        ),
    )
    csv_path = output_dir / "titans_eval_tidy.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=RULER_FIELDS)
        writer.writeheader()
        writer.writerows(asdict(row) for row in ordered)
    json_path = output_dir / "titans_eval_tidy.json"
    payload = {
        "schema_version": 1,
        "roots": {kind: [str(path) for path in paths] for kind, paths in roots.items()},
        "caveats": evaluation_caveats(rows),
        "records": [asdict(row) for row in ordered],
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return csv_path, json_path


def _task_group(task: str, overrides: dict[str, str]) -> str | None:
    if task in overrides:
        return overrides[task]
    normalized = re.sub(r"[^a-z0-9]+", "_", task.lower()).strip("_")
    if "pk" in normalized or "password" in normalized:
        return "pk"
    if "essay" in normalized or normalized == "s_niah_n":
        return "n"
    return None


def _matplotlib() -> Any | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except ImportError:
        LOGGER.warning("matplotlib is unavailable; tidy outputs and HTML tables will still be generated")
        return None


def _save_figure(figure: Any, output_dir: Path, stem: str) -> list[Path]:
    paths = [output_dir / f"{stem}.png", output_dir / f"{stem}.svg"]
    figure.savefig(paths[0], dpi=220, bbox_inches="tight")
    figure.savefig(paths[1], bbox_inches="tight")
    return paths


def _style_axes(axis: Any) -> None:
    axis.grid(axis="y", color="#e1e0d9", linewidth=0.8)
    axis.spines[["top", "right"]].set_visible(False)


def plot_ruler(rows: list[ReportRow], output_dir: Path, task_overrides: dict[str, str], plt: Any) -> list[Path]:
    """Render S-NIAH accuracy-versus-context figures without imputing absent cells."""
    outputs: list[Path] = []
    labels = {
        "pk": ("S-NIAH-PK / noise", "titans_s_niah_pk_noise"),
        "n": ("S-NIAH-N / essay", "titans_s_niah_n_essay"),
    }
    for group, (title, stem) in labels.items():
        figure, axis = plt.subplots(figsize=(7.2, 4.5))
        grouped: dict[tuple[str, str], list[ReportRow]] = {}
        for row in rows:
            if (
                row.source == "ruler"
                and row.metric == "accuracy"
                and row.value is not None
                and row.valid
                and row.complete
                and _task_group(row.task_or_benchmark, task_overrides) == group
            ):
                grouped.setdefault((row.architecture, row.ttt_mode), []).append(row)
        architectures = sorted({architecture for architecture, _ in grouped})
        color_map = {architecture: COLORS[index % len(COLORS)] for index, architecture in enumerate(architectures)}
        for (architecture, mode), series in sorted(grouped.items()):
            series = sorted(series, key=lambda row: row.context_length or -1)
            axis.plot(
                [row.context_length for row in series],
                [row.value for row in series],
                color=color_map[architecture],
                linestyle="-" if mode == "on" else "--" if mode == "off" else ":",
                marker="o",
                linewidth=2,
                label=f"{architecture} · TTT {mode}",
            )
        if not grouped:
            axis.text(
                0.5, 0.5, "No matching measurements available", ha="center", va="center", transform=axis.transAxes
            )
        axis.set_title(title, loc="left", fontweight="bold")
        axis.set_xlabel("Context length (tokens)")
        axis.set_ylabel("Accuracy (%)")
        axis.set_ylim(0, 100)
        if group == "n":
            axis.text(
                0.01,
                0.01,
                "essay@512 unavailable; no value imputed",
                transform=axis.transAxes,
                fontsize=9,
                color="#9b5a00",
            )
        if grouped:
            axis.legend(frameon=False, fontsize=8, ncol=2)
        _style_axes(axis)
        figure.tight_layout()
        outputs.extend(_save_figure(figure, output_dir, stem))
        plt.close(figure)
    return outputs


def plot_lm(rows: list[ReportRow], output_dir: Path, plt: Any) -> list[Path]:
    """Render standard-LM comparison panels for requested metrics."""
    outputs: list[Path] = []
    configurations = (
        ("wikitext103", ("perplexity", "bits_per_byte"), "WikiText-103", "titans_lm_wikitext103"),
        ("fineweb_edu", ("perplexity", "bits_per_byte"), "FineWeb-Edu", "titans_lm_fineweb_edu"),
        (
            "lambada",
            ("perplexity", "last_word_exact_accuracy"),
            "LAMBADA",
            "titans_lm_lambada",
        ),
    )
    ylabels = {
        "perplexity": "Perplexity (↓)",
        "bits_per_byte": "Bits per byte (↓)",
        "last_word_exact_accuracy": "Last-word accuracy",
    }
    for benchmark, metrics, title, stem in configurations:
        figure, axes = plt.subplots(1, 2, figsize=(9.6, 4.2))
        benchmark_rows = [
            row
            for row in rows
            if row.source == "lm"
            and row.task_or_benchmark == benchmark
            and row.value is not None
            and row.valid
            and row.complete
        ]
        architectures = sorted({row.architecture for row in benchmark_rows})
        x_positions = list(range(len(architectures)))
        for axis, metric in zip(axes, metrics):
            plotted = False
            for mode in sorted({row.ttt_mode for row in benchmark_rows}):
                values = {
                    row.architecture: float(row.value)
                    for row in benchmark_rows
                    if row.metric == metric and row.ttt_mode == mode and row.value is not None
                }
                if not values:
                    continue
                axis.plot(
                    [
                        x_positions[architectures.index(architecture)]
                        for architecture in architectures
                        if architecture in values
                    ],
                    [values[architecture] for architecture in architectures if architecture in values],
                    color="#0b7285" if mode == "on" else "#e8590c",
                    linestyle="-" if mode == "on" else "--" if mode == "off" else ":",
                    marker="o",
                    linewidth=2,
                    label=f"TTT {mode}",
                )
                plotted = True
            axis.set_title(ylabels[metric], fontsize=10)
            axis.set_xticks(x_positions, architectures, rotation=25, ha="right")
            if metric == "last_word_exact_accuracy":
                axis.set_ylim(0, 1)
            if plotted:
                axis.legend(frameon=False, fontsize=8)
            else:
                axis.text(0.5, 0.5, "Unavailable", ha="center", va="center", transform=axis.transAxes)
            _style_axes(axis)
        figure.suptitle(title, x=0.05, ha="left", fontweight="bold")
        figure.tight_layout()
        outputs.extend(_save_figure(figure, output_dir, stem))
        plt.close(figure)
    return outputs


def _embedded_svg(path: Path) -> str:
    text = path.read_text()
    return re.sub(r"<\?xml[^>]*>\s*|<!DOCTYPE[^>]*>\s*", "", text, count=2)


def write_html(rows: list[ReportRow], output_dir: Path, figure_paths: list[Path]) -> Path:
    """Write a self-contained option_b-style HTML report with inline SVG figures."""
    valid_count = sum(row.valid for row in rows)
    complete_count = sum(row.complete for row in rows)
    figure_cards = []
    for path in figure_paths:
        if path.suffix == ".svg":
            figure_cards.append(f'<figure class="card">{_embedded_svg(path)}</figure>')
    table_rows = []
    for row in rows:
        value = "unavailable" if row.value is None else f"{row.value:.6g}"
        context = "—" if row.context_length is None else f"{row.context_length:,}"
        state = "good" if row.valid and row.complete else "warn"
        table_rows.append(
            "<tr>"
            f"<td>{html.escape(row.source)}</td><td>{html.escape(row.architecture)}</td>"
            f"<td>{html.escape(row.ttt_mode)}</td><td>{html.escape(row.task_or_benchmark)}</td>"
            f"<td>{context}</td><td>{html.escape(row.metric)}</td><td>{value}</td>"
            f'<td><span class="pill {state}">{html.escape(row.status)}</span></td>'
            "</tr>"
        )
    generated_figures = (
        "\n".join(figure_cards)
        if figure_cards
        else '<div class="card"><p>No figures were embedded because matplotlib was unavailable.</p></div>'
    )
    caveats = "".join(f"<li>{html.escape(caveat)}</li>" for caveat in evaluation_caveats(rows))
    document = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Titans evaluation parity report</title>
<style>
:root{{--page:#f9f9f7;--card:#fff;--ink:#0b0b0b;--mut:#686762;--line:#e1e0d9;
--good:#236b3b;--goodbg:#e9f5ed;--warn:#9b5a00;--warnbg:#fff4df}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--page);color:var(--ink);
font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}}
section{{min-height:100vh;padding:42px 52px;border-bottom:1px solid var(--line)}} h1,h2{{margin:0 0 8px}}
.kicker{{color:var(--mut);font-size:12px;text-transform:uppercase;letter-spacing:.09em}}
.lede,.note{{color:var(--mut);max-width:900px}} .tiles{{display:grid;
grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;margin:18px 0}}
.tile,.card{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px}}
.tile .v{{font-size:27px;font-weight:700}} .tile .k{{font-size:11px;color:var(--mut);text-transform:uppercase}}
.figures{{display:grid;grid-template-columns:repeat(auto-fit,minmax(460px,1fr));gap:16px}}
figure{{margin:0}} figure svg{{width:100%;height:auto}} table{{border-collapse:collapse;width:100%;font-size:12px}}
th,td{{padding:6px 8px;border-bottom:1px solid var(--line);text-align:left}} th{{color:var(--mut)}}
.pill{{display:inline-block;border-radius:999px;padding:2px 8px}} .good{{color:var(--good);background:var(--goodbg)}}
.warn{{color:var(--warn);background:var(--warnbg)}} code{{font-family:ui-monospace,monospace}}
@media(max-width:700px){{section{{padding:28px 18px}}.figures{{grid-template-columns:1fr}}}}
</style></head><body>
<section><p class="kicker">Titans evaluation parity · aggregate report</p>
<h1>TTT intervention and evaluation status</h1>
<p class="lede">Measurements are read directly from merged RULER and language-model artifacts.
Missing cells remain unavailable; this report performs no interpolation or imputation.</p>
<div class="tiles"><div class="tile"><div class="v">{len(rows)}</div><div class="k">metric records</div></div>
<div class="tile"><div class="v">{valid_count}/{len(rows)}</div><div class="k">valid</div></div>
<div class="tile"><div class="v">{complete_count}/{len(rows)}</div><div class="k">complete</div></div></div>
<div class="card"><h2>Comparison caveats</h2><ul>{caveats}</ul></div></section>
<section><p class="kicker">Figures · solid TTT on · dashed TTT off</p><h2>Evaluation comparisons</h2>
<div class="figures">{generated_figures}</div></section>
<section><p class="kicker">Provenance and integrity</p><h2>All aggregated observations</h2>
<div class="card"><table><thead><tr><th>source</th><th>architecture</th><th>TTT</th><th>task / benchmark</th>
<th>context</th><th>metric</th><th>value</th><th>validity / completeness</th></tr></thead>
<tbody>{"".join(table_rows)}</tbody></table></div>
<p class="note">Checkpoint and code-revision provenance is preserved in the accompanying tidy JSON and CSV.
Null provenance fields mean the source evaluator did not record that value. “Complete” means all samples declared
by a RULER merged manifest, or all source indices and partials in an LM merge, are present; it does not claim that
an LM subset is the entire upstream benchmark.</p></section>
</body></html>
"""
    path = output_dir / "titans_eval_report.html"
    path.write_text(document)
    return path


def parse_args() -> argparse.Namespace:
    """Parse aggregation and rendering arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ruler-root", type=Path, action="append", default=[])
    parser.add_argument("--lm-root", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--task-group",
        action="append",
        default=[],
        metavar="TASK=GROUP",
        help="Map a RULER task to pk or n when its name is not recognized.",
    )
    return parser.parse_args()


def _task_overrides(values: list[str]) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for value in values:
        task, separator, group = value.partition("=")
        if not separator or not task or group not in {"pk", "n"}:
            raise ValueError(f"invalid --task-group {value!r}; expected TASK=pk or TASK=n")
        overrides[task] = group
    return overrides


def main() -> None:
    """Aggregate artifacts, render figures, and write the self-contained report."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    if not args.ruler_root and not args.lm_root:
        raise ValueError("Provide at least one --ruler-root or --lm-root")
    roots = {
        "ruler": [path.resolve() for path in args.ruler_root],
        "lm": [path.resolve() for path in args.lm_root],
    }
    rows = collect_ruler(roots["ruler"]) + collect_lm(roots["lm"])
    csv_path, json_path = write_tidy_outputs(rows, args.output_dir, roots)
    figure_paths: list[Path] = []
    plt = _matplotlib()
    if plt is not None:
        figure_paths.extend(plot_ruler(rows, args.output_dir, _task_overrides(args.task_group), plt))
        figure_paths.extend(plot_lm(rows, args.output_dir, plt))
    html_path = write_html(rows, args.output_dir, figure_paths)
    LOGGER.info("Wrote %s, %s, and %s (%d records)", csv_path, json_path, html_path, len(rows))


if __name__ == "__main__":
    main()
