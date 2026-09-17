"""Generate the figures used by docs/RETARGETING_TECHNICAL_REPORT.md.

The script reads only the checked experiment JSON files.  It does not load a
checkpoint or rerun training, so a regenerated figure is tied to the same
metrics that support the report text.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "assets"


def read(relative: str) -> dict:
    with (ROOT / relative).open(encoding="utf-8") as stream:
        return json.load(stream)


def operational(result: dict) -> dict:
    return result["wuji_comparison"]["candidate_operational"]


def add_values(axis, bars, fmt: str = ".1f") -> None:
    for bar in bars:
        value = bar.get_height()
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            format(value, fmt),
            ha="center",
            va="bottom",
            fontsize=8,
        )


def plot_tradeoffs() -> None:
    t1 = read("reports/stage2/T1_validation.json")
    t2 = read("reports/stage2/T2_validation.json")
    h0 = read("reports/stage3/H0_validation.json")
    h1 = read("reports/stage3/H1_validation.json")

    original = t2["wuji_comparison"]["original_geort_operational"]
    wuji = t2["wuji_comparison"]["filtered"]["operational"]
    rows = [
        ("Original\nGeoRT", 99.7126, original, 0.0),
        ("T1", 100 * t1["severe_fraction"]["little"], operational(t1), max(t1["tip_shift_p95_mm_per_finger"])),
        ("T2", 100 * t2["severe_fraction"]["little"], operational(t2), max(t2["tip_shift_p95_mm_per_finger"])),
        ("H0\ntip-only", 100 * h0["severe_fraction"]["little"], operational(h0), max(h0["tip_shift_p95_mm_per_finger"])),
        ("H1\nskeleton", 100 * h1["severe_fraction"]["little"], operational(h1), max(h1["tip_shift_p95_mm_per_finger"])),
        ("Wuji\nfiltered", 0.0, wuji, np.nan),
    ]
    labels = [row[0] for row in rows]
    colors = ["#586174", "#3b82f6", "#6d5bd0", "#d97706", "#059669", "#be123c"]
    x = np.arange(len(rows))
    metrics = [
        ("Little-finger severe backbend (%)", [row[1] for row in rows], None),
        ("Distal-axis error (deg)", [row[2]["distal_axis_error_deg"]["mean"] for row in rows], None),
        ("Opening error on open poses (mm)", [row[2]["opening_error_mm_open_poses"]["mean"] for row in rows], None),
        ("Observed response-direction error (deg)", [row[2]["local_response"]["response_direction_error_deg"]["mean"] for row in rows], None),
        ("Max per-finger tip shift P95 (mm)", [row[3] for row in rows], 5.0),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.2))
    for axis, (title, values, threshold) in zip(axes.flat, metrics):
        bars = axis.bar(x, values, color=colors, width=0.72)
        add_values(axis, bars)
        axis.set_title(title, fontsize=11)
        axis.set_xticks(x, labels, fontsize=8)
        axis.grid(axis="y", alpha=0.22)
        axis.spines[["top", "right"]].set_visible(False)
        if threshold is not None:
            axis.axhline(threshold, color="#991b1b", linestyle="--", linewidth=1.2, label="stage gate")
            axis.legend(frameon=False, fontsize=8)
        if np.isnan(values).any():
            axis.text(x[-1], 0.3, "N/A*", ha="center", va="bottom", fontsize=8, color=colors[-1])

    axes.flat[-1].axis("off")
    axes.flat[-1].text(
        0.02,
        0.94,
        "Reading guide",
        fontsize=13,
        fontweight="bold",
        va="top",
        transform=axes.flat[-1].transAxes,
    )
    axes.flat[-1].text(
        0.02,
        0.80,
        "Lower is better for every panel.\n\n"
        "* Wuji tip shift from Original GeoRT is not a\n"
        "candidate-preservation metric: it uses a different\n"
        "native model and operating point.\n\n"
        "No single panel is a system-quality score.",
        fontsize=10,
        linespacing=1.35,
        va="top",
        transform=axes.flat[-1].transAxes,
    )
    fig.suptitle("Validation trade-offs on the same 348-frame development segment", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "retargeting_tradeoffs.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_information_ablation() -> None:
    audit = read("reports/stage3/head_audit_verified.json")
    models = audit["models"]
    names = ["Frozen T2", "H0 tip-only", "H1 skeleton"]
    teacher_mse = [
        audit["frozen_base_train_teacher_normalized_mse"],
        models["H0"]["warmup_train_teacher_normalized_mse"],
        models["H1"]["warmup_train_teacher_normalized_mse"],
    ]
    head_change = [0.0, models["H0"]["validation"]["head_joint_change_deg_mean"], models["H1"]["validation"]["head_joint_change_deg_mean"]]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    x = np.arange(3)
    width = 0.36
    mse_bars = axes[0].bar(x - width / 2, teacher_mse, width, label="teacher normalized MSE", color="#4f46e5")
    other = axes[0].twinx()
    change_bars = other.bar(x + width / 2, head_change, width, label="mean head correction", color="#059669")
    axes[0].set_xticks(x, names)
    axes[0].set_ylabel("Normalized MSE")
    other.set_ylabel("Joint correction (deg)")
    axes[0].set_title("The new heads learn and change the output")
    axes[0].grid(axis="y", alpha=0.22)
    axes[0].spines["top"].set_visible(False)
    other.spines["top"].set_visible(False)
    add_values(axes[0], mse_bars, ".4f")
    add_values(other, change_bars, ".3f")
    axes[0].legend(
        [mse_bars, change_bars],
        ["teacher normalized MSE", "mean head correction"],
        frameon=False,
        fontsize=8,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=2,
    )

    fingers = ["thumb", "index", "middle", "ring", "little"]
    near_tip_max = [audit["information_pairs"]["train"][finger]["near_tip_max_axis_difference_deg"] for finger in fingers]
    bars = axes[1].bar(np.arange(5), near_tip_max, color="#d97706")
    add_values(axes[1], bars)
    axes[1].axhline(20.0, color="#991b1b", linestyle="--", linewidth=1.3, label="information test threshold")
    axes[1].set_xticks(np.arange(5), fingers)
    axes[1].set_ylabel("Maximum distal-axis difference (deg)")
    axes[1].set_title("Near-tip pairs do not contain the target distinction")
    axes[1].grid(axis="y", alpha=0.22)
    axes[1].spines[["top", "right"]].set_visible(False)
    axes[1].set_ylim(0, 22)
    axes[1].legend(frameon=False, fontsize=8, loc="lower right")
    axes[1].text(
        0.02,
        0.98,
        "tip distance <= 2 mm; frames >= 30 rows apart",
        transform=axes[1].transAxes,
        va="top",
        fontsize=8,
        color="#4b5563",
    )

    fig.suptitle("Stage 3 separates learning capacity from information coverage", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "skeleton_information_ablation.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    plot_tradeoffs()
    plot_information_ablation()
