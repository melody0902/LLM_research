import os
import json
import glob


# ============================================================
# 1) 讀取單一 json 結果檔
# ============================================================
def load_one_result(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        d = json.load(f)

    res = d.get("results", {})
    sym = res.get("symmetric", {})
    gap = res.get("direction_gap", None)

    return {
        "json_file": os.path.basename(json_path),
        "exclude_stopwords": d.get("exclude_stopwords", False),
        "top_k": d.get("top_k", None),
        "gamma": d.get("gamma", None),
        "sym_mode": d.get("sym_mode", None),
        "avg_d_lex": sym.get("avg_d_lex", None),
        "avg_d_ast": sym.get("avg_d_ast", None),
        "avg_d_file": sym.get("avg_d_file", None),
        "avg_rwmd_total": sym.get("avg_rwmd_total", None),
        "direction_gap": gap,
    }


# ============================================================
# 2) 產生 LaTeX 表格字串
# ============================================================
def to_latex_table(rows, caption, label):
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{l c r r r r r}")
    lines.append(r"\toprule")
    lines.append(r"Pair & Stop? & $d_{lex}$ & $d_{ast}$ & $d_{file}$ & RWMD & gap \\")
    lines.append(r"\midrule")

    for r in rows:
        pair_name = r["json_file"].replace(".json", "").replace("_", r"\_")
        stop_flag = "No" if (r["exclude_stopwords"] == False) else "Yes"

        lines.append(
            f"{pair_name} & {stop_flag} & "
            f"{r['avg_d_lex']:.4f} & {r['avg_d_ast']:.4f} & {r['avg_d_file']:.4f} & "
            f"{r['avg_rwmd_total']:.4f} & {r['direction_gap']:.4f} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# ============================================================
# 3) 對單一資料夾輸出一張表
# ============================================================
def export_one_dir_to_latex(result_dir, output_tex_name, caption, label):
    json_files = glob.glob(os.path.join(result_dir, "*.json"))

    if not json_files:
        print(f"⚠️ 找不到 json 檔：{result_dir}")
        return

    rows = []
    for jp in json_files:
        try:
            rows.append(load_one_result(jp))
        except Exception as e:
            print(f"⚠️ 解析失敗：{jp} -> {e}")

    # ✅ 用 RWMD 排序（由小到大）
    rows.sort(key=lambda x: x["avg_rwmd_total"])

    latex = to_latex_table(rows, caption=caption, label=label)

    out_path = os.path.join(result_dir, output_tex_name)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(latex)

    print(f"✅ 已輸出 LaTeX 表格：{out_path}")


# ============================================================
# main
# ============================================================
if __name__ == "__main__":
    base_dir = "outputs/0123_200green_synset_flat"

    # 你的兩個結果資料夾
    algo_dir = os.path.join(base_dir, "algo")
    cross_dir = os.path.join(base_dir, "crossdomain")

    # 1) 同領域不同演算法（algo/）
    export_one_dir_to_latex(
        result_dir=algo_dir,
        output_tex_name="rwmd_results_table_algo.tex",
        caption="RWMD (symmetric) results for comparing different watermark algorithms within the same domain.",
        label="tab:rwmd_algo_in_domain"
    )

    # 2) 同演算法跨領域（crossdomain/）
    export_one_dir_to_latex(
        result_dir=cross_dir,
        output_tex_name="rwmd_results_table_crossdomain.tex",
        caption="RWMD (symmetric) results for comparing different domains under the same watermark algorithm.",
        label="tab:rwmd_cross_domain"
    )
