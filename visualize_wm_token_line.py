import os
import re
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# rewritten_{domain}_{algorithm}_wm_token_freq.json
# =========================
FILENAME_RE = re.compile(
    r"rewritten_(?P<domain>[^_]+)_(?P<algorithm>[^_]+)_wm_token_freq\.json"
)

# =========================
# Stopword lists
# =========================
EN_STOPWORDS = {
    "the","of","and","to","in","for","with","on","at","by","from",
    "is","are","was","were","be","been","being",
    "a","an","this","that","these","those",
    "it","its","as","or","but","if","then",
    "we","you","they","he","she","them","his","her",
    "not","no","yes","do","does","did",
}

ZH_STOPWORDS = {
    "的","了","在","是","有","和","不","為","對","與","及","或",
    "也","而","但","若","則","並","其","於","之","所",
}

ALL_STOPWORDS = EN_STOPWORDS | ZH_STOPWORDS

# =========================
# Token normalization
# =========================
def normalize_token(tok: str) -> str:
    if tok is None:
        return ""
    t = tok.strip().lower()
    # 把奇怪的空白、換行統一
    t = t.replace("\n", "").replace("\t", "")
    return t

def is_valid_token(tok: str) -> bool:
    if tok == "":
        return False
    if tok in ALL_STOPWORDS:
        return False
    # 可選：過濾純標點（很常見但沒資訊）
    if all(not ch.isalnum() for ch in tok):
        return False
    return True

# =========================
# Load + preprocess
# =========================
def load_all_freq(output_dir: str) -> pd.DataFrame:
    records = []
    for fname in os.listdir(output_dir):
        m = FILENAME_RE.match(fname)
        if not m:
            continue

        domain = m.group("domain")
        algorithm = m.group("algorithm")
        path = os.path.join(output_dir, fname)

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for x in data:
            tok = normalize_token(str(x["token"]))
            if not is_valid_token(tok):
                continue

            records.append({
                "domain": domain,
                "algorithm": algorithm,
                "token": tok,
                "count": int(x["count"]),
            })

    df = pd.DataFrame(records)

    # 🔥 關鍵：相同 token（小寫後）一定要合併
    df = (
        df.groupby(["domain", "algorithm", "token"], as_index=False)["count"]
        .sum()
    )

    return df

# =========================
# Enrich features
# =========================
def enrich(df: pd.DataFrame) -> pd.DataFrame:
    totals = (
        df.groupby(["domain", "algorithm"])["count"]
        .sum()
        .reset_index()
        .rename(columns={"count": "total_injected"})
    )

    df = df.merge(totals, on=["domain", "algorithm"], how="left")
    df["normalized_count"] = df["count"] / df["total_injected"].replace(0, np.nan)
    df["normalized_count"] = df["normalized_count"].fillna(0.0)
    return df

# =========================
# Plot utils
# =========================
def safe_label(tok: str, max_len=12):
    return tok if len(tok) <= max_len else tok[:max_len-1] + "…"

def choose_topk_union(df, group_col, value_col, topk):
    tokens = set()
    for _, g in df.groupby(group_col):
        g2 = g.sort_values(value_col, ascending=False).head(topk)
        tokens.update(g2["token"].tolist())
    return df[df["token"].isin(tokens)].copy()

def plot_single_line(df, ycol, title, out_path):
    d = df.sort_values(ycol, ascending=False)
    labels = [safe_label(t) for t in d["token"]]
    y = d[ycol].values
    x = np.arange(len(labels))

    plt.figure(figsize=(max(10, len(labels) * 0.35), 5))
    plt.plot(x, y, marker="o", linewidth=1)

    plt.xticks(x, labels, rotation=60, ha="right")
    plt.ylabel(ycol)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_grouped_line(df, group_col, ycol, title, out_path):
    pivot = df.pivot_table(index="token", columns=group_col, values=ycol, fill_value=0)
    pivot["__sum__"] = pivot.sum(axis=1)
    pivot = pivot.sort_values("__sum__", ascending=False).drop(columns="__sum__")

    tokens = [safe_label(t) for t in pivot.index.tolist()]
    groups = pivot.columns.tolist()
    x = np.arange(len(tokens))

    plt.figure(figsize=(max(10, len(tokens) * 0.45), 5))
    for g in groups:
        plt.plot(x, pivot[g].values, marker="o", linewidth=1, label=str(g))

    plt.xticks(x, tokens, rotation=60, ha="right")
    plt.ylabel(ycol)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
# =========================
# main
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--save_dir", default="analysis_outputs_bars")
    ap.add_argument("--topk", type=int, default=30)
    ap.add_argument("--metric", choices=["raw","norm"], default="norm")
    ap.add_argument("--mode", choices=["split","grouped","both"], default="both")
    args = ap.parse_args()

    df = load_all_freq(args.output_dir)
    if df.empty:
        raise SystemExit(" No valid tokens after preprocessing")

    df = enrich(df)
    os.makedirs(args.save_dir, exist_ok=True)
    df.to_csv(os.path.join(args.save_dir, "token_long_preprocessed.csv"), index=False)

    ycol = "count" if args.metric == "raw" else "normalized_count"

    # 固定 domain
    for domain, dfd in df.groupby("domain"):
        dfd_top = choose_topk_union(dfd, "algorithm", ycol, args.topk)

        if args.mode in ("split","both"):
            for algo, g in dfd_top.groupby("algorithm"):
                plot_single_line(
                    g, ycol,
                    f"Domain={domain} | Algo={algo} | {ycol}",
                    f"{args.save_dir}/line_domain-{domain}_algo-{algo}_{args.metric}.png"
                )

        if args.mode in ("grouped","both"):
            plot_grouped_line(  
                dfd_top, "algorithm", ycol,
                f"Domain={domain} | grouped by algorithm | {ycol}",
                f"{args.save_dir}/grouped_line_domain-{domain}_{args.metric}.png"
            )
    # 固定 algorithm
    for algo, dfa in df.groupby("algorithm"):
        dfa_top = choose_topk_union(dfa, "domain", ycol, args.topk)

        if args.mode in ("split","both"):
            for domain, g in dfa_top.groupby("domain"):
                plot_single_line(
                    g, ycol,
                    f"Algo={algo} | Domain={domain} | {ycol}",
                    f"{args.save_dir}/line_algo-{algo}_domain-{domain}_{args.metric}.png"
                )

        if args.mode in ("grouped","both"):
            plot_grouped_line(
                dfa_top, "domain", ycol,
                f"Algo={algo} | grouped by domain | {ycol}",
                f"{args.save_dir}/grouped_line_algo-{algo}_{args.metric}.png"
            )

    print(f" Done. Outputs in {args.save_dir}")

if __name__ == "__main__":
    main()
