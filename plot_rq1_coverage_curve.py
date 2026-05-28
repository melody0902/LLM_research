import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def plot_average_coverage_curve(df: pd.DataFrame, save_path: str):
    coverage_cols = ["top200_coverage", "top500_coverage", "top1000_coverage"]
    ks = [200, 500, 1000]

    avg_df = df.groupby("algorithm", as_index=False)[coverage_cols].mean()

    plt.figure(figsize=(8, 5.2))

    for _, row in avg_df.sort_values("algorithm").iterrows():
        ys = [row[col] for col in coverage_cols]
        plt.plot(ks, ys, marker="o", label=row["algorithm"])

        # add value labels
        for x, y in zip(ks, ys):
            plt.text(
                x,
                y + 0.015,
                f"{y:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    plt.xlabel("k in top-k coverage")
    plt.ylabel("Coverage")
    plt.title("Average top-k coverage across domains")
    plt.xticks(ks)
    plt.ylim(0, 1.0)
    plt.legend(title="Algorithm")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    return avg_df


def plot_domainwise_coverage_curves(df: pd.DataFrame, save_dir: str):
    coverage_cols = ["top200_coverage", "top500_coverage", "top1000_coverage"]
    ks = [200, 500, 1000]

    for domain, g in df.groupby("domain"):
        plt.figure(figsize=(8, 5.2))

        for _, row in g.sort_values("algorithm").iterrows():
            ys = [row[col] for col in coverage_cols]
            plt.plot(ks, ys, marker="o", label=row["algorithm"])

            # add value labels
            for x, y in zip(ks, ys):
                plt.text(
                    x,
                    y + 0.015,
                    f"{y:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        plt.xlabel("k in top-k coverage")
        plt.ylabel("Coverage")
        plt.title(f"Top-k coverage in domain: {domain}")
        plt.xticks(ks)
        plt.ylim(0, 1.0)
        plt.legend(title="Algorithm")
        plt.tight_layout()

        out_path = os.path.join(save_dir, f"coverage_curve_{domain}.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", required=True, help="Path to topk_coverage_table.csv")
    ap.add_argument("--save_dir", default="analysis_figures", help="Directory to save plots")
    args = ap.parse_args()

    df = pd.read_csv(args.input_csv)

    os.makedirs(args.save_dir, exist_ok=True)

    avg_plot_path = os.path.join(args.save_dir, "coverage_curve_avg.png")
    avg_table_path = os.path.join(args.save_dir, "coverage_curve_avg_table.csv")

    avg_df = plot_average_coverage_curve(df, avg_plot_path)
    avg_df.to_csv(avg_table_path, index=False)

    plot_domainwise_coverage_curves(df, args.save_dir)

    print("Figures generated:")
    print(f" - {avg_plot_path}")
    print(f" - {avg_table_path}")
    for domain in sorted(df['domain'].unique()):
        print(f" - {os.path.join(args.save_dir, f'coverage_curve_{domain}.png')}")


if __name__ == "__main__":
    main()