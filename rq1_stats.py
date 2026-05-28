import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import friedmanchisquare
import scikit_posthocs as sp


CSV_PATH = "analysis_tables/topk_coverage_table.csv"
METRIC = "top200_coverage"
BLOCK_COL = "domain"
GROUP_COL = "algorithm"
OUTPUT_DIR = "analysis_tables/stats_outputs"
HIGHER_IS_BETTER = True


def compute_critical_difference(k: int, n: int, alpha: float = 0.05) -> float:
    """
    Compute the Nemenyi critical difference for average ranks.

    CD = q_alpha * sqrt(k(k+1)/(6n))

    For alpha=0.05 and large samples, a commonly used q_alpha is:
    - 2.728 for k=5
    Here we provide a small lookup table for common k.
    """
    q_alpha_table_05 = {
        2: 1.960,
        3: 2.344,
        4: 2.569,
        5: 2.728,
        6: 2.850,
        7: 2.949,
        8: 3.031,
        9: 3.102,
        10: 3.164,
    }

    if alpha != 0.05:
        raise ValueError("This script currently supports alpha=0.05 only.")

    if k not in q_alpha_table_05:
        raise ValueError(
            f"No built-in q_alpha value for k={k}. "
            "Extend q_alpha_table_05 if you need more algorithms."
        )

    q_alpha = q_alpha_table_05[k]
    cd = q_alpha * math.sqrt(k * (k + 1) / (6 * n))
    return cd


def plot_cd_diagram(avg_ranks: pd.Series, cd: float, output_path: str) -> None:
    """
    Draw a simple Critical Difference (CD) diagram.

    Lower rank is better.
    """
    sorted_ranks = avg_ranks.sort_values()
    labels = sorted_ranks.index.tolist()
    ranks = sorted_ranks.values.tolist()

    k = len(ranks)
    min_rank = 1
    max_rank = k

    fig_width = max(10, 1.6 * k)
    fig, ax = plt.subplots(figsize=(fig_width, 3.8))

    ax.set_xlim(min_rank - 0.5, max_rank + 0.5)
    ax.set_ylim(-1.6, 1.6)
    ax.axis("off")

    # Main axis
    ax.hlines(0.8, min_rank, max_rank, linewidth=1.5)

    for r in range(min_rank, max_rank + 1):
        ax.vlines(r, 0.72, 0.88, linewidth=1.2)
        ax.text(r, 1.0, str(r), ha="center", va="bottom", fontsize=10)

    # CD bar
    cd_start = min_rank
    cd_end = min_rank + cd
    ax.hlines(1.35, cd_start, cd_end, linewidth=2)
    ax.vlines([cd_start, cd_end], 1.28, 1.42, linewidth=2)
    ax.text((cd_start + cd_end) / 2, 1.46, f"CD = {cd:.3f}", ha="center", va="bottom", fontsize=10)

    # Algorithms
    left_y = 0.45
    right_y = 0.45
    left_step = 0.22
    right_step = 0.22

    midpoint = (min_rank + max_rank) / 2

    left_items = []
    right_items = []

    for label, rank in zip(labels, ranks):
        if rank <= midpoint:
            left_items.append((label, rank))
        else:
            right_items.append((label, rank))

    # Left side
    for i, (label, rank) in enumerate(left_items):
        y = left_y - i * left_step
        ax.vlines(rank, 0.8, y, linewidth=1)
        ax.hlines(y, min_rank - 0.15, rank, linewidth=1)
        ax.text(min_rank - 0.2, y, label, ha="right", va="center", fontsize=10)

    # Right side
    for i, (label, rank) in enumerate(right_items):
        y = right_y - i * right_step
        ax.vlines(rank, 0.8, y, linewidth=1)
        ax.hlines(y, rank, max_rank + 0.15, linewidth=1)
        ax.text(max_rank + 0.2, y, label, ha="left", va="center", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(CSV_PATH)

    required_cols = {BLOCK_COL, GROUP_COL, METRIC}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    pivot = df.pivot(index=BLOCK_COL, columns=GROUP_COL, values=METRIC)

    # Drop incomplete rows to keep repeated-measures design valid
    pivot = pivot.dropna(axis=0, how="any")

    if pivot.shape[0] < 2:
        raise ValueError("Need at least 2 blocks/domains after dropping missing values.")

    if pivot.shape[1] < 3:
        raise ValueError("Nemenyi/Friedman comparison is intended for at least 3 algorithms.")

    print(f"Metric: {METRIC}")
    print(f"Blocks ({BLOCK_COL}): {pivot.shape[0]}")
    print(f"Algorithms ({GROUP_COL}): {pivot.shape[1]}")
    print()

    # Friedman test
    stat, p_value = friedmanchisquare(*[pivot[col] for col in pivot.columns])
    print("Friedman statistic:", stat)
    print("p-value:", p_value)
    print()

    # Rank each block: rank 1 = best
    rank_ascending = not HIGHER_IS_BETTER
    ranks = pivot.rank(axis=1, method="average", ascending=rank_ascending)
    avg_ranks = ranks.mean(axis=0).sort_values()

    print("Average ranks (lower is better):")
    print(avg_ranks)
    print()

    # Save average ranks
    avg_ranks_df = avg_ranks.reset_index()
    avg_ranks_df.columns = [GROUP_COL, "average_rank"]
    avg_ranks_path = os.path.join(OUTPUT_DIR, f"{METRIC}_average_ranks.csv")
    avg_ranks_df.to_csv(avg_ranks_path, index=False)

    # Nemenyi post-hoc
    nemenyi = sp.posthoc_nemenyi_friedman(pivot)
    nemenyi_path = os.path.join(OUTPUT_DIR, f"{METRIC}_nemenyi_pvalues.csv")
    nemenyi.to_csv(nemenyi_path)

    print("Nemenyi post-hoc p-value matrix:")
    print(nemenyi)
    print()

    # Critical Difference
    k = pivot.shape[1]
    n = pivot.shape[0]
    cd = compute_critical_difference(k=k, n=n, alpha=0.05)

    print(f"Critical Difference (alpha=0.05): {cd:.4f}")
    print()

    # CD diagram
    cd_plot_path = os.path.join(OUTPUT_DIR, f"{METRIC}_cd_diagram.png")
    plot_cd_diagram(avg_ranks=avg_ranks, cd=cd, output_path=cd_plot_path)

    print("Saved files:")
    print(f"- {avg_ranks_path}")
    print(f"- {nemenyi_path}")
    print(f"- {cd_plot_path}")


if __name__ == "__main__":
    main()