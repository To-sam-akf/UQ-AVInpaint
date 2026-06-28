#!/usr/bin/env python3
"""Build semantic perturbation ablation tables and figures from test JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


EXPERIMENTS = [
    ("O", "Original\nVIAI-AV", "../original_viai_av_baseline_eval"),
    ("A", "EC-VIAI-AV\nCandidate-Scorer", "A_stage8_90000_fused_readonly"),
    ("B", "Semantic Evidence\nFine-Tuning", "B_stage10c_95000_no_perturb"),
    ("C", "Semantic Perturbation\n5k, w=0.35", "C_stage10d_95000_perturb"),
    ("D", "Semantic Perturbation\nFinal, w=0.35", "D_stage10d_100000_perturb"),
    ("E", "No Wrong-Video\nAugmentation", "E_no_wrong_aug_95000"),
    ("F", "Heuristic-Only\nPerturbation", "F_heuristic_perturb_95000"),
    ("G0.2", "Semantic Perturbation\n5k, w=0.2", "G_sem_w0.2_95000"),
    ("G0.5", "Semantic Perturbation\n5k, w=0.5", "G_sem_w0.5_95000"),
]

MODES = ["none", "flow_zero", "no_video", "wrong_video_cross_instrument"]

FIELDS = [
    "top1_missing_l1",
    "best_of_k_missing_l1",
    "mel_l1_missing",
    "probe_l1_missing",
    "psnr_missing",
    "ssim",
    "evidence_mean",
    "heuristic_evidence_mean",
    "semantic_evidence_mean",
    "gate_mean",
    "gate_target_mean",
    "gate_target_gap",
    "uncertainty_mean",
    "uncertainty_error_spearman",
]


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_results(root: Path) -> pd.DataFrame:
    rows = []
    for exp_id, method, dirname in EXPERIMENTS:
        exp_root = (root / dirname).resolve()
        for mode in MODES:
            files = sorted((exp_root / mode).glob("*_test.json"))
            if not files:
                raise FileNotFoundError(f"missing test json: {exp_root / mode}")
            data = _load_json(files[0])
            row = {
                "id": exp_id,
                "method": method.replace("\n", " "),
                "method_plot": method,
                "dirname": dirname,
                "mode": mode,
                "json_path": str(files[0]),
            }
            for field in FIELDS:
                row[field] = data.get(field)
            rows.append(row)
    return pd.DataFrame(rows)


def build_compact_table(df: pd.DataFrame) -> pd.DataFrame:
    def val(exp_id: str, mode: str, field: str) -> float:
        rows = df[(df["id"] == exp_id) & (df["mode"] == mode)]
        if rows.empty:
            raise KeyError((exp_id, mode, field))
        return float(rows.iloc[0][field])

    compact = []
    for exp_id, method, _dirname in EXPERIMENTS:
        compact.append(
            {
                "id": exp_id,
                "method": method.replace("\n", " "),
                "none_top1_l1": val(exp_id, "none", "top1_missing_l1"),
                "flow_top1_l1": val(exp_id, "flow_zero", "top1_missing_l1"),
                "no_video_top1_l1": val(exp_id, "no_video", "top1_missing_l1"),
                "wrong_top1_l1": val(
                    exp_id, "wrong_video_cross_instrument", "top1_missing_l1"
                ),
                "none_best_k_l1": val(exp_id, "none", "best_of_k_missing_l1"),
                "wrong_best_k_l1": val(
                    exp_id, "wrong_video_cross_instrument", "best_of_k_missing_l1"
                ),
                "none_psnr": val(exp_id, "none", "psnr_missing"),
                "no_video_psnr": val(exp_id, "no_video", "psnr_missing"),
                "wrong_psnr": val(
                    exp_id, "wrong_video_cross_instrument", "psnr_missing"
                ),
                "none_ssim": val(exp_id, "none", "ssim"),
                "wrong_ssim": val(exp_id, "wrong_video_cross_instrument", "ssim"),
                "wrong_gate": val(
                    exp_id, "wrong_video_cross_instrument", "gate_mean"
                ),
                "no_video_gate": val(exp_id, "no_video", "gate_mean"),
                "wrong_unc_spearman": val(
                    exp_id,
                    "wrong_video_cross_instrument",
                    "uncertainty_error_spearman",
                ),
                "none_unc_spearman": val(
                    exp_id, "none", "uncertainty_error_spearman"
                ),
            }
        )
    return pd.DataFrame(compact)


def build_quality_table(compact: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "id",
        "method",
        "none_top1_l1",
        "flow_top1_l1",
        "no_video_top1_l1",
        "wrong_top1_l1",
        "none_best_k_l1",
        "wrong_best_k_l1",
        "none_psnr",
        "no_video_psnr",
        "wrong_psnr",
        "none_ssim",
        "wrong_ssim",
    ]
    return compact[cols].copy()


def build_robustness_table(compact: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "id",
        "method",
        "wrong_gate",
        "no_video_gate",
        "wrong_unc_spearman",
        "none_unc_spearman",
    ]
    return compact[cols].copy()


def _set_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 220,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _bar(ax, labels, values, title, ylabel, color="#4C78A8"):
    ax.bar(labels, values, color=color)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=35)
    for tick in ax.get_xticklabels():
        tick.set_ha("right")
    for i, value in enumerate(values):
        ax.text(i, value, f"{value:.3f}", ha="center", va="bottom", fontsize=7)


def plot_figures(df: pd.DataFrame, compact: pd.DataFrame, out_dir: Path) -> None:
    _set_style()
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = compact["id"].tolist()

    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    _bar(
        ax,
        labels,
        compact["wrong_gate"].tolist(),
        "Wrong-video gate mean",
        "gate_mean",
        "#E45756",
    )
    ax.axhline(0.108308, color="black", linestyle="--", linewidth=1, label="target")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "wrong_video_gate_mean.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    _bar(
        ax,
        labels,
        compact["no_video_gate"].tolist(),
        "No-video gate mean",
        "gate_mean",
        "#72B7B2",
    )
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1, label="target")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "no_video_gate_mean.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    _bar(
        ax,
        labels,
        compact["wrong_unc_spearman"].tolist(),
        "Wrong-video uncertainty-error Spearman",
        "Spearman",
        "#54A24B",
    )
    fig.tight_layout()
    fig.savefig(out_dir / "wrong_video_uncertainty_spearman.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    x = range(len(labels))
    width = 0.38
    ax.bar(
        [i - width / 2 for i in x],
        compact["none_top1_l1"],
        width=width,
        label="none",
        color="#4C78A8",
    )
    ax.bar(
        [i + width / 2 for i in x],
        compact["wrong_top1_l1"],
        width=width,
        label="wrong video",
        color="#F58518",
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_title("Missing-region top1 L1")
    ax.set_ylabel("top1_missing_l1")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "top1_l1_none_vs_wrong.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.0, 4.4))
    x = range(len(labels))
    width = 0.25
    ax.bar(
        [i - width for i in x],
        compact["none_top1_l1"],
        width=width,
        label="none",
        color="#4C78A8",
    )
    ax.bar(
        list(x),
        compact["no_video_top1_l1"],
        width=width,
        label="no video",
        color="#72B7B2",
    )
    ax.bar(
        [i + width for i in x],
        compact["wrong_top1_l1"],
        width=width,
        label="wrong video",
        color="#F58518",
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_title("Missing-region top1 L1 by visual condition")
    ax.set_ylabel("top1_missing_l1 (lower is better)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "quality_top1_l1_by_condition.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.0, 4.4))
    ax.bar(
        [i - width for i in x],
        compact["none_psnr"],
        width=width,
        label="none",
        color="#4C78A8",
    )
    ax.bar(
        list(x),
        compact["no_video_psnr"],
        width=width,
        label="no video",
        color="#72B7B2",
    )
    ax.bar(
        [i + width for i in x],
        compact["wrong_psnr"],
        width=width,
        label="wrong video",
        color="#F58518",
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_title("Missing-region PSNR by visual condition")
    ax.set_ylabel("psnr_missing (higher is better)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "quality_psnr_by_condition.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    scatter = ax.scatter(
        compact["wrong_gate"],
        compact["wrong_unc_spearman"],
        s=85,
        c=compact["none_top1_l1"],
        cmap="viridis_r",
        edgecolor="black",
        linewidth=0.6,
    )
    for _, row in compact.iterrows():
        ax.text(row["wrong_gate"] + 0.008, row["wrong_unc_spearman"], row["id"])
    ax.set_title("Wrong-video robustness trade-off")
    ax.set_xlabel("wrong_video gate_mean (lower is better)")
    ax.set_ylabel("wrong_video uncertainty Spearman (higher is better)")
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("none top1 L1 (lower is better)")
    fig.tight_layout()
    fig.savefig(out_dir / "wrong_video_tradeoff_scatter.png")
    plt.close(fig)

    pivot = df.pivot(index="id", columns="mode", values="gate_mean").loc[labels]
    fig, ax = plt.subplots(figsize=(9.0, 4.7))
    pivot.plot(kind="bar", ax=ax, width=0.78)
    ax.set_title("Gate mean by perturbation mode")
    ax.set_ylabel("gate_mean")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(title="mode", ncols=2)
    fig.tight_layout()
    fig.savefig(out_dir / "gate_mean_by_mode.png")
    plt.close(fig)


def _markdown_table(table: pd.DataFrame) -> str:
    table = table.copy()
    numeric_cols = [c for c in table.columns if c not in {"id", "method"}]
    for col in numeric_cols:
        table[col] = table[col].map(lambda x: f"{x:.6f}")

    headers = list(table.columns)
    md_rows = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in table.iterrows():
        md_rows.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    return "\n".join(md_rows)


def write_markdown(
    compact: pd.DataFrame,
    quality: pd.DataFrame,
    robustness: pd.DataFrame,
    out_path: Path,
    figures_dir: Path,
) -> None:
    rel_fig = figures_dir.as_posix()
    lines = [
        "# Semantic Perturbation Ablation Tables and Figures",
        "",
        "## Main Quality Table",
        "",
        _markdown_table(quality),
        "",
        "## Main Robustness Table",
        "",
        _markdown_table(robustness),
        "",
        "## Full Compact Table",
        "",
        _markdown_table(compact),
        "",
        "## Figure Files",
        "",
        f"- `{rel_fig}/wrong_video_gate_mean.png`",
        f"- `{rel_fig}/no_video_gate_mean.png`",
        f"- `{rel_fig}/wrong_video_uncertainty_spearman.png`",
        f"- `{rel_fig}/top1_l1_none_vs_wrong.png`",
        f"- `{rel_fig}/quality_top1_l1_by_condition.png`",
        f"- `{rel_fig}/quality_psnr_by_condition.png`",
        f"- `{rel_fig}/wrong_video_tradeoff_scatter.png`",
        f"- `{rel_fig}/gate_mean_by_mode.png`",
        "",
        "## Recommended Main Result",
        "",
        "Use `D Semantic Perturbation Final, w=0.35` as the robustness-oriented final model.",
        "It has the lowest wrong-video gate, strong no-video suppression, and the best",
        "wrong-video uncertainty-error Spearman among the tested methods.",
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("checkpoints/ablation_stage10d"),
        help="Semantic perturbation ablation result root.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("docs/figures/stage10d_ablation"),
        help="Output directory for tables and figures.",
    )
    parser.add_argument(
        "--tables-md",
        type=Path,
        default=Path("docs/stage10d_final_tables.md"),
        help="Markdown summary path.",
    )
    args = parser.parse_args()

    df = load_results(args.root)
    compact = build_compact_table(df)
    quality = build_quality_table(compact)
    robustness = build_robustness_table(compact)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_dir / "stage10d_ablation_detailed_metrics.csv", index=False)
    compact.to_csv(args.out_dir / "stage10d_ablation_compact_table.csv", index=False)
    quality.to_csv(args.out_dir / "stage10d_ablation_quality_table.csv", index=False)
    robustness.to_csv(
        args.out_dir / "stage10d_ablation_robustness_table.csv", index=False
    )
    plot_figures(df, compact, args.out_dir)
    write_markdown(compact, quality, robustness, args.tables_md, args.out_dir)

    print(f"wrote detailed csv: {args.out_dir / 'stage10d_ablation_detailed_metrics.csv'}")
    print(f"wrote compact csv: {args.out_dir / 'stage10d_ablation_compact_table.csv'}")
    print(f"wrote quality csv: {args.out_dir / 'stage10d_ablation_quality_table.csv'}")
    print(f"wrote robustness csv: {args.out_dir / 'stage10d_ablation_robustness_table.csv'}")
    print(f"wrote figures: {args.out_dir}")
    print(f"wrote markdown: {args.tables_md}")


if __name__ == "__main__":
    main()
