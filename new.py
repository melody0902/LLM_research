import numpy as np
import pandas as pd
from scipy.stats import kendalltau

df = pd.read_csv("/home/soslab/Desktop/Melody/signature/llm-watermark-research/analysis_tables/token_count_distribution.csv")

# 移除 TOTAL 列
df = df[df["count"] != "TOTAL"].copy()
df["count"] = df["count"].astype(int)
df["num_tokens"] = df["num_tokens"].astype(int)

# --- 計算每個 domain × algorithm 的 coverage curve ---
# 展開：每個 token 的出現次數 = 一筆資料
# (count=5, num_tokens=3) → 3 個 token 各出現 5 次
def expand_and_sort(group):
    counts = np.repeat(group["count"].values, group["num_tokens"].values)
    counts = np.sort(counts)[::-1]  # 由大到小
    cumsum = np.cumsum(counts)
    total = cumsum[-1]
    coverage = cumsum / total
    return coverage

groups = df.groupby(["domain", "algorithm"])
curves = {}
for (domain, algo), group in groups:
    curves[(domain, algo)] = expand_and_sort(group)

# --- 用 AUC 作為每條 curve 的單一代表值 ---
# 統一取前 K 個 token（取各組最短長度）
K = min(len(v) for v in curves.values())

auc_scores = {}
for (domain, algo), curve in curves.items():
    auc_scores[(domain, algo)] = np.mean(curve[:K])  # AUC ≈ mean coverage over top-K

# --- 整理成 domain × algorithm 矩陣 ---
domains = df["domain"].unique()
algorithms = df["algorithm"].unique()

matrix = pd.DataFrame(index=domains, columns=algorithms, dtype=float)
for (domain, algo), auc in auc_scores.items():
    matrix.loc[domain, algo] = auc

print("AUC matrix:")
print(matrix.round(4))

# --- 計算 Kendall's W ---
def kendalls_w(matrix):
    """
    matrix: rows = raters (domains), columns = subjects (algorithms)
    """
    m, n = matrix.shape  # m = domains, n = algorithms

    # 每個 domain 內對 algorithm 排名
    ranks = matrix.rank(axis=1)  # 每列內排名

    # 各 algorithm 的排名總和
    R = ranks.sum(axis=0)
    R_mean = R.mean()

    # S = sum of squared deviations
    S = np.sum((R - R_mean) ** 2)

    # Kendall's W
    W = 12 * S / (m ** 2 * (n ** 3 - n))

    # 近似卡方檢定
    chi2 = m * (n - 1) * W
    df_chi2 = n - 1
    from scipy.stats import chi2 as chi2_dist
    p_value = 1 - chi2_dist.cdf(chi2, df_chi2)

    return W, chi2, df_chi2, p_value

W, chi2, df_chi2, p = kendalls_w(matrix)
print(f"\nKendall's W = {W:.4f}")
print(f"Chi-square = {chi2:.4f}, df = {df_chi2}, p = {p:.4f}")

# --- 印出各 domain 的演算法排名 ---
rank_matrix = matrix.rank(axis=1, ascending=False).astype(int)
print("\nAlgorithm rankings per domain (1 = most uneven):")
print(rank_matrix)