import math
import itertools
import pandas as pd
import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon, spearmanr

# =========================================
# Config
# =========================================
INPUT_CSV = "rq2_crossdomain_master.csv"
ALGORITHMS = ["KGW", "SWEET", "Unigram", "EXP", "SynthID"]
TARGET_ALGO = "SWEET"
ALPHA = 0.05
# 你可以選擇分析 nostop 或是 full
ANALYSIS_SETTING = "nostop" 

# =========================================
# Helpers
# =========================================
def holm_correction(pvals_dict):
    if not pvals_dict: return {}
    items = sorted(pvals_dict.items(), key=lambda x: x[1])
    m = len(items)
    adjusted = {}
    prev = 0.0
    for i, (name, p) in enumerate(items):
        adj = (m - i) * p
        adj = max(adj, prev)
        adj = min(adj, 1.0)
        adjusted[name] = adj
        prev = adj
    return adjusted

def rank_biserial_from_diffs(diffs):
    # 這裡保留你原本的邏輯，已修正 list conversion 以防 pandas Series 報錯
    diffs = np.array(diffs)
    nonzero = diffs[diffs != 0]
    if len(nonzero) == 0: return 0.0
    
    ranks = pd.Series(np.abs(nonzero)).rank()
    w_pos = ranks[nonzero > 0].sum()
    w_neg = ranks[nonzero < 0].sum()
    
    denom = w_pos + w_neg
    return (w_pos - w_neg) / denom if denom != 0 else 0.0

def safe_wilcoxon(x, y, alternative="two-sided"):
    diffs = [a - b for a, b in zip(x, y)]
    nonzero_diffs = [d for d in diffs if abs(d) > 1e-9]

    if len(nonzero_diffs) < 5: # Wilcoxon 需要足夠樣本
        return {"statistic": 0.0, "pvalue": 1.0, "n_nonzero": len(nonzero_diffs), "median_diff": 0.0, "rank_biserial": 0.0}

    res = wilcoxon(x, y, alternative=alternative)
    return {
        "statistic": float(res.statistic),
        "pvalue": float(res.pvalue),
        "n_nonzero": len(nonzero_diffs),
        "median_diff": float(np.median(diffs)),
        "rank_biserial": float(rank_biserial_from_diffs(diffs))
    }

def print_section(title):
    print(f"\n{'='*80}\n{title}\n{'='*80}")

# =========================================
# Load & Clean Data
# =========================================
df = pd.read_csv(INPUT_CSV)

# 過濾 Setting
df = df[df["setting"] == ANALYSIS_SETTING].copy()
df = df[df["algorithm"].isin(ALGORITHMS)].copy()

# 關鍵步驟：Pivot 並丟棄不完整的對齊資料
# 只有在所有指定的 ALGORITHMS 裡都有資料的 Pair 才會被留下
pivot_rwmd = df.pivot(index="pair", columns="algorithm", values="rwmd_sym")
pivot_rwmd = pivot_rwmd.dropna(subset=ALGORITHMS)

n_pairs = len(pivot_rwmd)
print_section(f"Data Summary (Setting: {ANALYSIS_SETTING})")
print(f"Total matched pairs found: {n_pairs}")
if n_pairs == 0:
    print("Error: No complete matches found across all algorithms. Check your CSV and algorithm names.")
    exit()

# =========================================
# H2a: Friedman Omnibus
# =========================================
print_section("H2a: Friedman Test (Global Differences)")
friedman_arrays = [pivot_rwmd[algo].values for algo in ALGORITHMS]
f_stat, f_p = friedmanchisquare(*friedman_arrays)

k = len(ALGORITHMS)
kendalls_w = f_stat / (n_pairs * (k - 1))

print(f"Chi-square: {f_stat:.4f}, p-value: {f_p:.6g}")
print(f"Kendall's W (Effect Size): {kendalls_w:.4f}")

# =========================================
# H2b: Post-hoc (Target vs Others)
# =========================================
print_section(f"H2b: {TARGET_ALGO} vs Others (One-sided: {TARGET_ALGO} < Competitor)")

h2b_raw_p = {}
h2b_results = {}

for comp in ALGORITHMS:
    if comp == TARGET_ALGO: continue
    
    res = safe_wilcoxon(pivot_rwmd[TARGET_ALGO], pivot_rwmd[comp], alternative="less")
    key = f"{TARGET_ALGO} < {comp}"
    h2b_raw_p[key] = res["pvalue"]
    h2b_results[key] = res

h2b_adj_p = holm_correction(h2b_raw_p)

for key, res in h2b_results.items():
    print(f"{key:20} | adj_p: {h2b_adj_p[key]:.6g} | r_rb: {res['rank_biserial']:.4f} | Med_Diff: {res['median_diff']:.4f}")

# =========================================
# H2c: Correlations
# =========================================
print_section("H2c: Spearman Correlation (RWMD vs Linguistic Divergence)")
components = ["d_token", "d_synset", "d_lca"]
for comp in components:
    # 這裡使用完整的資料集 (不限於配對) 來算相關性
    valid_df = df.dropna(subset=["rwmd_sym", comp])
    rho, p = spearmanr(valid_df["rwmd_sym"], valid_df[comp])
    print(f"RWMD vs {comp:10} | rho: {rho:.4f}, p: {p:.6g} (n={len(valid_df)})")

print("\nAnalysis Complete.")