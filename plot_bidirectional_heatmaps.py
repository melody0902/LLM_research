import json
import glob
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_bidirectional_heatmaps(compare_dir, output_dir="plots_bidir"):
    """
    支援兩種 compare JSON 檔名格式：

    (A) in-domain:      full_{domain}_{algoA}_vs_{algoB}.json
        -> 每個 Domain / Type / Metric 畫一張 algo×algo bidirectional heatmap

    (B) cross-domain:   full_{algo}_{domainA}_vs_{domainB}.json
        -> 每個 Algo / Type / Metric 畫一張 domain×domain bidirectional heatmap
    """
    os.makedirs(output_dir, exist_ok=True)

    metrics = ["avg_d_lex", "avg_d_ast", "avg_d_file", "avg_rwmd_total"]
    records = []

    known_domains = ["ai", "bio", "med", "mis", "security"]

    # 讀取所有 compare JSON
    for fpath in glob.glob(os.path.join(compare_dir, "*.json")):
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)

        fname = os.path.basename(fpath).replace(".json", "")
        parts = fname.split("_")
        if len(parts) < 5:
            continue

        t = parts[0]  # full / nostop

        # 新版格式必須有 a2b / b2a
        if "results" not in data or "a2b" not in data["results"] or "b2a" not in data["results"]:
            print(f"[WARN] Missing a2b/b2a in: {fpath}")
            continue

        # 判斷是哪種格式：
        # parts[1] 若是 domain => in-domain
        # 否則 => cross-domain
        if parts[1] in known_domains:
            mode = "indomain"
            group = parts[1]  # domain
            left = parts[2]   # algoA
            right = parts[4]  # algoB
        else:
            mode = "crossdomain"
            group = parts[1]  # algo
            left = parts[2]   # domainA
            right = parts[4]  # domainB

        for m in metrics:
            a2b_val = data["results"]["a2b"].get(m, None)
            b2a_val = data["results"]["b2a"].get(m, None)
            if a2b_val is None or b2a_val is None:
                continue

            records.append({
                "Mode": mode,
                "Type": t,
                "Group": group,   # indomain=Domain, crossdomain=Algo
                "Metric": m,
                "Left": left,     # indomain=AlgoA, crossdomain=DomainA
                "Right": right,   # indomain=AlgoB, crossdomain=DomainB
                "A2B": a2b_val,
                "B2A": b2a_val
            })

    df = pd.DataFrame(records)
    if df.empty:
        print("[ERROR] No valid data found. Check compare_dir or JSON schema.")
        return

    # 依照模式分開畫：indomain & crossdomain
    for mode in sorted(df["Mode"].unique()):
        df_mode = df[df["Mode"] == mode]

        # 每個 group/type/metric 畫一張：group=domain(模式A) 或 algo(模式B)
        for group in sorted(df_mode["Group"].unique()):
            for t in sorted(df_mode["Type"].unique()):
                for metric in metrics:
                    sub = df_mode[
                        (df_mode["Group"] == group) &
                        (df_mode["Type"] == t) &
                        (df_mode["Metric"] == metric)
                    ]
                    if sub.empty:
                        continue

                    # 軸 labels：indomain -> algos；crossdomain -> domains
                    labels = sorted(list(set(sub["Left"].unique()) | set(sub["Right"].unique())))
                    mat = pd.DataFrame(np.nan, index=labels, columns=labels)

                    for _, row in sub.iterrows():
                        a = row["Left"]
                        b = row["Right"]
                        mat.loc[a, b] = row["A2B"]  # 上三角：A→B
                        mat.loc[b, a] = row["B2A"]  # 下三角：B→A

                    plt.figure(figsize=(8, 6))
                    sns.heatmap(
                        mat,
                        annot=True,
                        fmt=".3f",
                        cmap="YlGnBu",
                        linewidths=0.5,
                        linecolor="white",
                        cbar=True
                    )

                    if mode == "indomain":
                        plt.title(f"{metric} (upper=A→B, lower=B→A)\nDomain: {group.upper()} ({t.upper()})")
                        plt.xlabel("Target (AlgoB)")
                        plt.ylabel("Source (AlgoA)")
                        out_name = f"heatmap_bidir_{metric}_domain-{group}_{t}.png"
                    else:
                        plt.title(f"{metric} (upper=A→B, lower=B→A)\nAlgo: {group} ({t.upper()})")
                        plt.xlabel("Target (DomainB)")
                        plt.ylabel("Source (DomainA)")
                        out_name = f"heatmap_bidir_{metric}_algo-{group}_{t}.png"

                    plt.savefig(os.path.join(output_dir, out_name), dpi=300, bbox_inches="tight")
                    plt.close()

    print(f"[DONE] Bidirectional heatmaps saved to: {output_dir}/")


if __name__ == "__main__":
    # 你可以各自跑兩個資料夾，也可以把兩邊 JSON 都丟進同一個 compare_dir 再一起畫
    plot_bidirectional_heatmaps("outputs/0123_200green_synset_flat/algo", output_dir="plots_bidir_indomain")
    plot_bidirectional_heatmaps("outputs/0123_200green_synset_flat/crossdomain", output_dir="plots_bidir_crossdomain")
