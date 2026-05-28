# import os
# import re
# import json
# import argparse
# import pandas as pd

# # =========================
# # filename pattern
# # =========================
# FILENAME_RE = re.compile(
#     r"rewritten_(?P<domain>[^_]+)_(?P<algorithm>[^_]+)_wm_token_freq\.json"
# )

# # =========================
# # Stopwords
# # =========================
# EN_STOPWORDS = {
#     "the","of","and","to","in","for","with","on","at","by","from",
#     "is","are","was","were","be","been","being",
#     "a","an","this","that","these","those",
#     "it","its","as","or","but","if","then",
#     "we","you","they","he","she","them","his","her",
#     "not","no","yes","do","does","did",
# }

# ZH_STOPWORDS = {
#     "的","了","在","是","有","和","不","為","對","與","及","或",
#     "也","而","但","若","則","並","其","於","之","所",
# }

# STOPWORDS = EN_STOPWORDS | ZH_STOPWORDS

# # =========================
# # Token preprocessing
# # =========================
# def normalize_token(tok: str) -> str:
#     if tok is None:
#         return ""
#     return tok.strip().lower().replace("\n", "").replace("\t", "")

# def is_valid_token(tok: str) -> bool:
#     if tok == "":
#         return False
#     if tok in STOPWORDS:
#         return False
#     if all(not c.isalnum() for c in tok):
#         return False
#     return True

# # =========================
# # Load + preprocess
# # =========================
# def load_all_freq(output_dir: str) -> pd.DataFrame:
#     records = []

#     for fname in os.listdir(output_dir):
#         m = FILENAME_RE.match(fname)
#         if not m:
#             continue

#         domain = m.group("domain")
#         algorithm = m.group("algorithm")
#         path = os.path.join(output_dir, fname)

#         with open(path, "r", encoding="utf-8") as f:
#             data = json.load(f)

#         for x in data:
#             tok = normalize_token(str(x["token"]))
#             if not is_valid_token(tok):
#                 continue

#             records.append({
#                 "domain": domain,
#                 "algorithm": algorithm,
#                 "token": tok,
#                 "count": int(x["count"]),
#             })

#     df = pd.DataFrame(records)

#     df = (
#         df.groupby(["domain", "algorithm", "token"], as_index=False)["count"]
#         .sum()
#     )

#     return df

# # =========================
# # Build tables
# # =========================
# def build_count_tables(df: pd.DataFrame):
#     dist = (
#         df.groupby(["domain", "algorithm", "count"])
#         .size()
#         .reset_index(name="num_tokens")
#         .sort_values(["domain", "algorithm", "count"])
#     )

#     totals = (
#         df.groupby(["domain", "algorithm"])
#         .size()
#         .reset_index(name="num_tokens")
#     )
#     totals["count"] = "TOTAL"

#     dist_with_total = pd.concat([dist, totals], ignore_index=True)

#     summary_rows = []
#     for (domain, algorithm), g in df.groupby(["domain", "algorithm"]):
#         total = len(g)
#         c1 = (g["count"] == 1).sum()
#         c2 = (g["count"] == 2).sum()
#         c3p = (g["count"] >= 3).sum()

#         summary_rows.append({
#             "domain": domain,
#             "algorithm": algorithm,
#             "total_tokens": total,
#             "count_eq_1": c1,
#             "count_eq_2": c2,
#             "count_ge_3": c3p,
#             "ratio_eq_1": c1 / total if total > 0 else 0.0,
#         })

#     summary = pd.DataFrame(summary_rows)

#     return dist_with_total, summary

# def compute_topk_coverage(df: pd.DataFrame, ks=(20, 50, 100, 200, 500, 1000)):
#     rows = []

#     for (domain, algorithm), g in df.groupby(["domain", "algorithm"]):
#         g_sorted = g.sort_values("count", ascending=False)
#         total = g_sorted["count"].sum()

#         row = {
#             "domain": domain,
#             "algorithm": algorithm
#         }

#         for k in ks:
#             topk = g_sorted.head(k)["count"].sum()
#             row[f"top{k}_coverage"] = topk / total if total > 0 else 0.0

#         rows.append(row)

#     return pd.DataFrame(rows)

# # =========================
# # main
# # =========================
# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--output_dir", required=True, help="wm_token_freq.json 所在資料夾")
#     ap.add_argument("--save_dir", default="analysis_tables", help="輸出表格資料夾")
#     args = ap.parse_args()

#     df = load_all_freq(args.output_dir)
#     if df.empty:
#         raise SystemExit("No valid tokens after preprocessing")

#     dist_table, summary_table = build_count_tables(df)
#     topk_table = compute_topk_coverage(df, ks=(20, 50, 100, 200, 500, 1000))

#     os.makedirs(args.save_dir, exist_ok=True)

#     dist_path = os.path.join(args.save_dir, "token_count_distribution.csv")
#     summary_path = os.path.join(args.save_dir, "token_count_summary.csv")
#     topk_path = os.path.join(args.save_dir, "topk_coverage_table.csv")

#     dist_table.to_csv(dist_path, index=False)
#     summary_table.to_csv(summary_path, index=False)
#     topk_table.to_csv(topk_path, index=False)

#     print("Tables generated:")
#     print(f" - {dist_path}")
#     print(f" - {summary_path}")
#     print(f" - {topk_path}")

# if __name__ == "__main__":
#     main()


# ============================================================
# analyze_wm_token_freq.py
# ============================================================

import os
import re
import json
import argparse
import pandas as pd

# =========================
# Default paths
# =========================

DEFAULT_OUTPUT_DIR = (
    "/home/soslab/Desktop/Melody/signature/llm-watermark-research/"
    "outputs/0517_200green"
)

DEFAULT_SAVE_DIR = (
    "/home/soslab/Desktop/Melody/signature/llm-watermark-research/"
    "analysis_tables/0517_200green"
)

# =========================
# Target algorithms / domains
# =========================

ALGORITHMS = ["KGW", "SWEET", "Unigram", "EXP", "SynthID"]
DOMAINS = ["ai", "bio", "med", "mis", "security"]

# =========================
# filename pattern
# =========================
# Example:
# rewritten_ai_EXP_meta-llama__Llama-3.1-8B-Instruct_wm_token_freq.json

FILENAME_RE = re.compile(
    r"^rewritten_"
    r"(?P<domain>[^_]+)_"
    r"(?P<algorithm>[^_]+)_"
    r"(?P<model>.+)"
    r"_wm_token_freq\.json$"
)

# =========================
# Stopwords
# =========================

EN_STOPWORDS = {
    "the", "of", "and", "to", "in", "for", "with", "on", "at", "by", "from",
    "is", "are", "was", "were", "be", "been", "being",
    "a", "an", "this", "that", "these", "those",
    "it", "its", "as", "or", "but", "if", "then",
    "we", "you", "they", "he", "she", "them", "his", "her",
    "not", "no", "yes", "do", "does", "did",
}

ZH_STOPWORDS = {
    "的", "了", "在", "是", "有", "和", "不", "為", "對", "與", "及", "或",
    "也", "而", "但", "若", "則", "並", "其", "於", "之", "所",
}

STOPWORDS = EN_STOPWORDS | ZH_STOPWORDS

# =========================
# Token preprocessing
# =========================

def normalize_token(tok: str) -> str:
    if tok is None:
        return ""

    tok = str(tok)
    tok = tok.strip()
    tok = tok.lower()
    tok = tok.replace("\n", "")
    tok = tok.replace("\t", "")

    return tok


def is_valid_token(tok: str) -> bool:
    if tok == "":
        return False

    if tok in STOPWORDS:
        return False

    # Remove pure punctuation / symbols
    if all(not c.isalnum() for c in tok):
        return False

    return True


# =========================
# Load + preprocess
# =========================

def load_all_freq(output_dir: str) -> pd.DataFrame:
    records = []
    matched_files = []
    skipped_files = []

    if not os.path.isdir(output_dir):
        raise NotADirectoryError(f"Output directory not found: {output_dir}")

    for fname in sorted(os.listdir(output_dir)):
        m = FILENAME_RE.match(fname)

        if not m:
            skipped_files.append((fname, "filename pattern not matched"))
            continue

        domain = m.group("domain")
        algorithm = m.group("algorithm")
        model = m.group("model")

        if domain not in DOMAINS:
            skipped_files.append((fname, f"domain not in DOMAINS: {domain}"))
            continue

        if algorithm not in ALGORITHMS:
            skipped_files.append((fname, f"algorithm not in ALGORITHMS: {algorithm}"))
            continue

        path = os.path.join(output_dir, fname)

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        file_valid_count = 0

        for x in data:
            if "token" not in x or "count" not in x:
                continue

            tok = normalize_token(x["token"])

            if not is_valid_token(tok):
                continue

            records.append({
                "domain": domain,
                "algorithm": algorithm,
                "model": model,
                "token_id": x.get("token_id", None),
                "token": tok,
                "count": int(x["count"]),
                "source_file": fname,
            })

            file_valid_count += 1

        if file_valid_count > 0:
            matched_files.append(fname)
        else:
            skipped_files.append((fname, "no valid tokens after preprocessing"))

    print("Matched files:")
    for fname in matched_files:
        print(f" - {fname}")

    print()
    print(f"Total matched files: {len(matched_files)}")
    print(f"Total skipped files: {len(skipped_files)}")

    if skipped_files:
        print()
        print("Skipped files:")
        for fname, reason in skipped_files:
            print(f" - {fname}: {reason}")

    df = pd.DataFrame(records)

    if df.empty:
        return df

    # Same token may appear multiple times after normalization.
    # Since your file names include model, keep model in the grouping.
    df = (
        df.groupby(
            ["domain", "algorithm", "model", "token"],
            as_index=False
        )
        .agg({
            "count": "sum",
            "token_id": "first",
            "source_file": lambda x: ";".join(sorted(set(x))),
        })
    )

    df = df[
        [
            "domain",
            "algorithm",
            "model",
            "token_id",
            "token",
            "count",
            "source_file",
        ]
    ]

    return df


# =========================
# Build tables
# =========================

def build_count_tables(df: pd.DataFrame):
    dist = (
        df.groupby(["domain", "algorithm", "model", "count"])
        .size()
        .reset_index(name="num_tokens")
        .sort_values(["domain", "algorithm", "model", "count"])
    )

    totals = (
        df.groupby(["domain", "algorithm", "model"])
        .size()
        .reset_index(name="num_tokens")
    )
    totals["count"] = "TOTAL"

    dist_with_total = pd.concat([dist, totals], ignore_index=True)

    summary_rows = []

    for (domain, algorithm, model), g in df.groupby(
        ["domain", "algorithm", "model"]
    ):
        total = len(g)
        c1 = (g["count"] == 1).sum()
        c2 = (g["count"] == 2).sum()
        c3p = (g["count"] >= 3).sum()

        summary_rows.append({
            "domain": domain,
            "algorithm": algorithm,
            "model": model,
            "total_tokens": total,
            "count_eq_1": c1,
            "count_eq_2": c2,
            "count_ge_3": c3p,
            "ratio_eq_1": c1 / total if total > 0 else 0.0,
            "ratio_eq_2": c2 / total if total > 0 else 0.0,
            "ratio_ge_3": c3p / total if total > 0 else 0.0,
        })

    summary = pd.DataFrame(summary_rows)

    return dist_with_total, summary


def compute_topk_coverage(df: pd.DataFrame, ks=(200, 500, 1000)):
    rows = []

    # 如果你每個 domain + algorithm 只有一個 model，
    # 這樣輸出就會是 domain, algorithm, top200_coverage...
    for (domain, algorithm), g in df.groupby(["domain", "algorithm"]):
        g_sorted = g.sort_values("count", ascending=False)
        total = g_sorted["count"].sum()

        row = {
            "domain": domain,
            "algorithm": algorithm,
        }

        for k in ks:
            topk = g_sorted.head(k)["count"].sum()
            row[f"top{k}_coverage"] = topk / total if total > 0 else 0.0

        rows.append(row)

    return pd.DataFrame(rows)


def compute_topk_coverage_by_model(df: pd.DataFrame, ks=(200, 500, 1000)):
    rows = []

    for (domain, algorithm, model), g in df.groupby(["domain", "algorithm", "model"]):
        g_sorted = g.sort_values("count", ascending=False)
        total = g_sorted["count"].sum()

        row = {
            "domain": domain,
            "algorithm": algorithm,
            "model": model,
        }

        for k in ks:
            topk = g_sorted.head(k)["count"].sum()
            row[f"top{k}_coverage"] = topk / total if total > 0 else 0.0

        rows.append(row)

    return pd.DataFrame(rows)


def build_top_tokens_table(df: pd.DataFrame, top_n=1000):
    rows = []

    for (domain, algorithm, model), g in df.groupby(["domain", "algorithm", "model"]):
        g_sorted = g.sort_values("count", ascending=False).head(top_n)

        for rank, (_, row) in enumerate(g_sorted.iterrows(), start=1):
            rows.append({
                "domain": domain,
                "algorithm": algorithm,
                "model": model,
                "rank": rank,
                "token_id": row["token_id"],
                "token": row["token"],
                "count": row["count"],
                "source_file": row["source_file"],
            })

    return pd.DataFrame(rows)


# =========================
# main
# =========================

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--output_dir",
        default=DEFAULT_OUTPUT_DIR,
        help="wm_token_freq.json 所在資料夾",
    )

    ap.add_argument(
        "--save_dir",
        default=DEFAULT_SAVE_DIR,
        help="輸出表格資料夾",
    )

    args = ap.parse_args()

    print("Input directory:")
    print(args.output_dir)
    print()

    df = load_all_freq(args.output_dir)

    if df.empty:
        raise SystemExit("No valid tokens after preprocessing")

    dist_table, summary_table = build_count_tables(df)

    # 你要的 200, 500, 1000
    topk_table = compute_topk_coverage(df, ks=(200, 500, 1000))
    topk_by_model_table = compute_topk_coverage_by_model(df, ks=(200, 500, 1000))

    top_tokens_table = build_top_tokens_table(df, top_n=1000)

    os.makedirs(args.save_dir, exist_ok=True)

    cleaned_path = os.path.join(args.save_dir, "cleaned_token_freq.csv")
    dist_path = os.path.join(args.save_dir, "token_count_distribution.csv")
    summary_path = os.path.join(args.save_dir, "token_count_summary.csv")
    topk_path = os.path.join(args.save_dir, "topk_coverage_table.csv")
    topk_by_model_path = os.path.join(args.save_dir, "topk_coverage_table_by_model.csv")
    top_tokens_path = os.path.join(args.save_dir, "top_tokens_1000.csv")

    df.to_csv(cleaned_path, index=False)
    dist_table.to_csv(dist_path, index=False)
    summary_table.to_csv(summary_path, index=False)
    topk_table.to_csv(topk_path, index=False)
    topk_by_model_table.to_csv(topk_by_model_path, index=False)
    top_tokens_table.to_csv(top_tokens_path, index=False)

    print()
    print("Tables generated:")
    print(f" - {cleaned_path}")
    print(f" - {dist_path}")
    print(f" - {summary_path}")
    print(f" - {topk_path}")
    print(f" - {topk_by_model_path}")
    print(f" - {top_tokens_path}")

    print()
    print("Top-k coverage:")
    print(topk_table.to_string(index=False))


if __name__ == "__main__":
    main()