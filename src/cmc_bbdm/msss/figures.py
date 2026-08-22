"""Publication figures rendered only from S1 artifact tables."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


class MSSSFigureError(ValueError):
    """Raised when a required S1 figure source is invalid."""


FULL = "#222222"
MSSS = "#0072B2"
OVER = "#D55E00"
SENSITIVITY = "#A6A6A6"
SKY = "#56B4E9"


def _rows(path: Path) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = [dict(row) for row in csv.DictReader(handle, strict=True)]
    except (OSError, UnicodeError, csv.Error) as error:
        raise MSSSFigureError(f"figure source is unreadable: {path.name}") from error
    if not rows:
        raise MSSSFigureError(f"figure source is empty: {path.name}")
    return rows


def _save(fig: plt.Figure, root: Path, stem: str) -> tuple[Path, Path]:
    pdf = root / f"{stem}.pdf"
    png = root / f"{stem}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return pdf, png


def _curve_figure(
    rows: list[dict[str, str]],
    *,
    xlabel: str,
    stem: str,
    root: Path,
    reverse_x: bool = False,
) -> tuple[Path, Path]:
    primary = [row for row in rows if row["primary_eligible"] == "true"]
    x = np.asarray([float(row["value"]) for row in primary])
    y = np.asarray([float(row["equal_domain_mae"]) for row in primary])
    low = np.asarray([float(row["ci_low"]) for row in primary])
    high = np.asarray([float(row["ci_high"]) for row in primary])
    full_mae = float(primary[0]["full_equal_domain_mae"])
    fig, ax = plt.subplots(figsize=(5.4, 3.5))
    ax.fill_between(x, low, high, color=SKY, alpha=0.18, linewidth=0)
    ax.plot(x, y, color=MSSS, marker="s", linewidth=1.8, markersize=4.5)
    ax.scatter([x[0]], [y[0]], color=FULL, marker="o", s=35, zorder=4, label="FULL")
    ax.axhspan(full_mae, full_mae * 1.05, color="#D9D9D9", alpha=0.55, label="5% non-inferiority")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Equal-domain CAI MAE")
    if reverse_x:
        ax.invert_xaxis()
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="best")
    fig.tight_layout()
    return _save(fig, root, stem)


def _wavelet_figure(rows: list[dict[str, str]], root: Path) -> tuple[Path, Path]:
    fig, ax = plt.subplots(figsize=(5.8, 3.7))
    series: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        key = (row["wavelet"] or "db2", row["mode"] or "low_only")
        series.setdefault(key, []).append(row)
    for (family, mode), values in series.items():
        ordered = sorted(values, key=lambda row: int(float(row["level"] or 0)))
        x = np.asarray([int(float(row["level"] or 0)) for row in ordered])
        y = np.asarray([float(row["equal_domain_mae"]) for row in ordered])
        primary = family == "db2" and mode == "low_only"
        label = f"{family} {mode.replace('_', ' ')}"
        ax.plot(
            x,
            y,
            color=MSSS if primary else (SKY if mode == "low_only" else SENSITIVITY),
            marker="s" if primary else "o",
            linestyle="-" if mode == "low_only" else "--",
            linewidth=1.9 if primary else 1.0,
            markersize=4 if primary else 3,
            alpha=1.0 if primary else 0.75,
            label=label,
        )
    full_mae = float(rows[0]["full_equal_domain_mae"])
    ax.axhspan(full_mae, full_mae * 1.05, color="#D9D9D9", alpha=0.5)
    ax.set_xlabel("DWT decomposition level")
    ax.set_ylabel("Equal-domain CAI MAE")
    ax.set_xticks((0, 1, 2, 3))
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=6.5, ncol=2, loc="best")
    fig.tight_layout()
    return _save(fig, root, "figure_c_wavelet_scale")


def _combined_figure(root: Path) -> tuple[Path, Path]:
    curve_rows = {
        axis: _rows(root / f"{axis}_curve.csv")
        for axis in ("sampling", "gaussian", "wavelet")
    }
    selections = [
        row
        for row in _rows(root / "msss_selection.csv")
        if row["scope"] == "global_descriptive"
    ]
    specificity = {
        row["axis"]: row
        for row in _rows(root / "spatial_specificity.csv")
        if row["dataset_id"] == "EQUAL_DOMAIN"
    }
    fig, (left, right) = plt.subplots(1, 2, figsize=(9.2, 3.6))
    offsets = (-0.22, 0.0, 0.22)
    labels = ("FULL", "MSSS", "OVER-COARSE")
    colors = (FULL, MSSS, OVER)
    markers = ("o", "s", "^")
    for axis_index, axis in enumerate(("sampling", "gaussian", "wavelet")):
        rows_by_id = {row["condition_id"]: row for row in curve_rows[axis]}
        selection = next(row for row in selections if row["axis"] == axis)
        condition_ids = (
            selection["full_condition_id"],
            selection["selected_condition_id"],
            selection["over_coarse_condition_id"],
        )
        for offset, label, color, marker, condition_id in zip(
            offsets, labels, colors, markers, condition_ids, strict=True
        ):
            if not condition_id:
                continue
            row = rows_by_id[condition_id]
            left.scatter(
                axis_index + offset,
                float(row["relative_gap"]) * 100.0,
                color=color,
                marker=marker,
                s=42,
                label=label if axis_index == 0 else None,
                zorder=3,
            )
    left.axhspan(0.0, 5.0, color="#D9D9D9", alpha=0.55)
    left.axhline(5.0, color="#666666", linestyle="--", linewidth=0.9)
    left.set_xticks(range(3), ("Sampling", "Gaussian", "Wavelet"))
    left.set_ylabel("MAE gap from FULL (%)")
    left.spines[["top", "right"]].set_visible(False)
    left.grid(axis="y", color="#D9D9D9", linewidth=0.6)
    left.legend(frameon=False, fontsize=8, loc="best")

    axes = ("sampling", "gaussian", "wavelet")
    effects = [float(specificity[axis]["ssg"]) for axis in axes]
    lows = [float(specificity[axis]["ci_low"]) for axis in axes]
    highs = [float(specificity[axis]["ci_high"]) for axis in axes]
    lower_errors = np.maximum(0.0, np.asarray(effects) - np.asarray(lows))
    upper_errors = np.maximum(0.0, np.asarray(highs) - np.asarray(effects))
    right.bar(range(3), effects, color=MSSS, width=0.58)
    right.errorbar(
        range(3),
        effects,
        yerr=[lower_errors, upper_errors],
        fmt="none",
        ecolor=FULL,
        capsize=3,
        linewidth=1,
    )
    right.axhline(0.0, color=FULL, linewidth=0.8)
    right.set_xticks(range(3), ("Sampling", "Gaussian", "Wavelet"))
    right.set_ylabel("SSG: shuffled MAE - scale MAE")
    right.spines[["top", "right"]].set_visible(False)
    right.grid(axis="y", color="#D9D9D9", linewidth=0.6)
    fig.tight_layout()
    return _save(fig, root, "figure_d_sufficiency_specificity")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render_s1_figures(package_dir: str | Path) -> dict[str, object]:
    """Render all mandatory figures from already written CSV sources."""

    root = Path(package_dir)
    if not root.is_dir():
        raise MSSSFigureError("figure package directory is unavailable")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    outputs: list[Path] = []
    outputs.extend(
        _curve_figure(
            _rows(root / "sampling_curve.csv"),
            xlabel="Requested measurement density",
            stem="figure_a_sampling_scale",
            root=root,
            reverse_x=True,
        )
    )
    outputs.extend(
        _curve_figure(
            _rows(root / "gaussian_curve.csv"),
            xlabel="Gaussian sigma (pixels)",
            stem="figure_b_gaussian_scale",
            root=root,
        )
    )
    outputs.extend(_wavelet_figure(_rows(root / "wavelet_curve.csv"), root))
    outputs.extend(_combined_figure(root))
    manifest = {
        "schema_version": 1,
        "sources": {
            name: _sha256(root / name)
            for name in (
                "sampling_curve.csv",
                "gaussian_curve.csv",
                "wavelet_curve.csv",
                "spatial_specificity.csv",
                "msss_selection.csv",
            )
        },
        "outputs": {path.name: _sha256(path) for path in outputs},
    }
    (root / "figure_manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return manifest


__all__ = ["MSSSFigureError", "render_s1_figures"]
