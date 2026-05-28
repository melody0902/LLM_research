import json
import os
import math
import itertools
import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
import fasttext
import fasttext.util
import numpy as np

# ============================================================
# 0. NLTK setup
# ============================================================

# Auto-download required NLTK resources (safe)
try:
    wn.ensure_loaded()
except Exception:
    nltk.download("wordnet")
    nltk.download("omw-1.4")

try:
    stopwords.words("english")
except Exception:
    nltk.download("stopwords")



# 載入一次就好，放在 module level
fasttext.util.download_model('en', if_exists='ignore')
FT_MODEL = fasttext.load_model('cc.en.300.bin')

def get_ft_vector(token):
    return FT_MODEL.get_word_vector(token.strip().lower())

def cosine_distance(v1, v2):
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm == 0:
        return 1.0
    return 1.0 - float(np.dot(v1, v2) / norm)

def calculate_pair_metrics_ft(entry1, entry2):
    """Embedding-based ground cost，替換 WordNet 版本"""
    t1 = entry1["token"]
    t2 = entry2["token"]

    # token exact match 仍然是 0
    if t1.strip().lower() == t2.strip().lower():
        return {
            "d_token": 0.0,
            "d_embed": 0.0,
            "total_divergence": 0.0
        }

    v1 = get_ft_vector(t1)
    v2 = get_ft_vector(t2)
    d_embed = cosine_distance(v1, v2)

    return {
        "d_token": 1.0,
        "d_embed": round(d_embed, 4),
        "total_divergence": round(d_embed, 4)
    }


# ============================================================
# 1. Utility functions
# ============================================================

def safe_get_synset(name):
    """Safely resolve a WordNet synset name into a synset object."""
    if not name:
        return None
    try:
        return wn.synset(name)
    except Exception:
        return None


def calculate_pair_metrics(entry1, entry2, gamma=2.0):
    """
    Compute component divergences between two entries.
    Returns:
        d_token, d_synset, d_lca, total_divergence
    """
    t1, s1_name = entry1["token"], entry1["synset"]
    t2, s2_name = entry2["token"], entry2["synset"]

    syn1 = safe_get_synset(s1_name)
    syn2 = safe_get_synset(s2_name)

    # d_token: token/synset exact match => 0 else 1
    is_token_same = (t1.strip().lower() == t2.strip().lower())
    is_synset_same = (s1_name is not None) and (s1_name == s2_name)
    d_token = 0.0 if (is_token_same or is_synset_same) else 1.0

    # If synset missing, treat as maximal semantic divergence
    if not syn1 or not syn2:
        d_synset = 1.0
        d_lca = 1.0
        total_div = (d_token * (d_synset + gamma * d_lca)) / (1 + gamma)
        return {
            "d_token": d_token,
            "d_synset": d_synset,
            "d_lca": d_lca,
            "total_divergence": total_div
        }

    # -------------------------
    # d_synset: synset shortest path distance (normalized)
    # -------------------------
    dist = syn1.shortest_path_distance(syn2, simulate_root=True)
    if dist is None:
        d_synset = 1.0
    elif dist == 0:
        d_synset = 0.0
    else:
        TREE_DIAMETER = 30
        d_synset = min(math.log(1 + dist) / math.log(1 + TREE_DIAMETER), 1.0)

    # -------------------------
    # d_lca: hypernym divergence based on LCP depth
    # -------------------------
    if syn1 == syn2:
        d_lca = 0.0
    else:
        d_lca = 1.0
        common_hypernyms = syn1.lowest_common_hypernyms(syn2)
        if common_hypernyms:
            lcp_depth = max(common_hypernyms, key=lambda s: s.max_depth()).max_depth()
            max_depth = max(syn1.max_depth(), syn2.max_depth())
            if max_depth > 0:
                d_lca = 1.0 - (lcp_depth / max_depth)

    total_div = (d_token * (d_synset + (gamma * d_lca))) / (1 + gamma)

    return {
        "d_token": d_token,
        "d_synset": d_synset,
        "d_lca": d_lca,
        "total_divergence": total_div
    }


# ============================================================
# 2. Preprocess profile (case merge + optional stopword removal)
# ============================================================

def preprocess_profile(data, exclude_stopwords=False):
    """
    Preprocess token frequency profile:
      - lowercasing
      - merging duplicate tokens
      - optional stopword filtering
    """
    stop_words = set(stopwords.words("english")) if exclude_stopwords else set()
    grouped = {}

    for entry in data:
        raw_token = entry.get("token", "").strip()
        token_lower = raw_token.lower()

        # filter stopwords and non-alnum tokens if requested
        if exclude_stopwords:
            if (token_lower in stop_words) or (not token_lower.isalpha()):
                continue

        count = entry.get("count", 0)
        synset = entry.get("synset")

        if token_lower not in grouped:
            grouped[token_lower] = {
                "token": token_lower,
                "synset": synset,
                "count": count
            }
        else:
            grouped[token_lower]["count"] += count

    return sorted(grouped.values(), key=lambda x: x["count"], reverse=True)


# ============================================================
# 3. RWMD core (bidirectional + symmetric)
# ============================================================

def calculate_rwmd_metrics(data_a, data_b, gamma=2.0, 
                           sym_mode="mean", mode="wordnet"):
    """
    Compute RWMD-like divergence between two profiles, including:
      - directional A->B alignment
      - directional B->A alignment
      - symmetric aggregation (mean/max/rms/min)

    Returns:
      {
        "a2b": {...},
        "b2a": {...},
        "symmetric": {...},
        "direction_gap": float
      }
    """
    total_count_a = sum(d["count"] for d in data_a)
    total_count_b = sum(d["count"] for d in data_b)

    def empty_schema():
        empty = {
            "avg_d_token": 1.0,
            "avg_d_synset": 1.0,
            "avg_d_lca": 1.0,
            "avg_rwmd_total": 1.0,
        }
        return {
            "a2b": empty,
            "b2a": empty,
            "symmetric": {"mode": sym_mode, **empty},
            "direction_gap": 0.0
        }

    if total_count_a == 0 or total_count_b == 0:
        return empty_schema()

    dist_matrix = []
    for d1 in data_a:
        if mode == "fasttext":
            row = [calculate_pair_metrics_ft(d1, d2) for d2 in data_b]
        else:
            row = [calculate_pair_metrics(d1, d2, gamma) for d2 in data_b]
        dist_matrix.append(row)

    def get_weighted_aligned_metrics(source_data, source_total, is_forward):
        sums = {"token": 0.0, "synset": 0.0, "lca": 0.0, "total": 0.0}

        for i in range(len(source_data)):
            prob = source_data[i]["count"] / source_total

            if is_forward:
                best_match = min(dist_matrix[i], key=lambda x: x["total_divergence"])
            else:
                best_match = min(
                    [dist_matrix[row][i] for row in range(len(data_a))],
                    key=lambda x: x["total_divergence"]
                )

            sums["token"] += prob * best_match["d_token"]
            sums["synset"] += prob * best_match["d_synset"]
            sums["lca"] += prob * best_match["d_lca"]
            sums["total"] += prob * best_match["total_divergence"]

        return sums

    m_a2b = get_weighted_aligned_metrics(data_a, total_count_a, True)
    m_b2a = get_weighted_aligned_metrics(data_b, total_count_b, False)

    def sym(x, y):
        if sym_mode == "max":
            return max(x, y)
        elif sym_mode == "mean":
            return (x + y) / 2
        elif sym_mode == "rms":
            return math.sqrt((x * x + y * y) / 2)
        elif sym_mode == "min":
            return min(x, y)
        else:
            raise ValueError(f"Unknown sym_mode: {sym_mode}")

    a2b_out = {
        "avg_d_token": round(m_a2b["token"], 4),
        "avg_d_synset": round(m_a2b["synset"], 4),
        "avg_d_lca": round(m_a2b["lca"], 4),
        "avg_rwmd_total": round(m_a2b["total"], 4),
    }

    b2a_out = {
        "avg_d_token": round(m_b2a["token"], 4),
        "avg_d_synset": round(m_b2a["synset"], 4),
        "avg_d_lca": round(m_b2a["lca"], 4),
        "avg_rwmd_total": round(m_b2a["total"], 4),
    }

    sym_out = {
        "mode": sym_mode,
        "avg_d_token": round(sym(m_a2b["token"], m_b2a["token"]), 4),
        "avg_d_synset": round(sym(m_a2b["synset"], m_b2a["synset"]), 4),
        "avg_d_lca": round(sym(m_a2b["lca"], m_b2a["lca"]), 4),
        "avg_rwmd_total": round(sym(m_a2b["total"], m_b2a["total"]), 4),
    }

    direction_gap = round(abs(m_a2b["total"] - m_b2a["total"]), 4)

    return {
        "a2b": a2b_out,
        "b2a": b2a_out,
        "symmetric": sym_out,
        "direction_gap": direction_gap
    }


# ============================================================
# 4. Compare and save report
# ============================================================

def compare_profiles(path_a, path_b, output_path, exclude_stop, 
                     top_k=300, gamma=2.0, sym_mode="mean", mode="wordnet"):
    if not (os.path.exists(path_a) and os.path.exists(path_b)):
        return False

    with open(path_a, "r", encoding="utf-8") as f:
        data_a = json.load(f)
    with open(path_b, "r", encoding="utf-8") as f:
        data_b = json.load(f)

    clean_a = preprocess_profile(data_a, exclude_stopwords=exclude_stop)[:top_k]
    clean_b = preprocess_profile(data_b, exclude_stopwords=exclude_stop)[:top_k]

    metrics = calculate_rwmd_metrics(clean_a, clean_b, 
                                     gamma=gamma, sym_mode=sym_mode, mode=mode)

    report = {
        "file_A": path_a,
        "file_B": path_b,
        "exclude_stopwords": exclude_stop,
        "top_k": top_k,
        "gamma": gamma,
        "mode": mode,
        "sym_mode": sym_mode,
        "results": metrics
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    return True


# ============================================================
# 修正 get_weighted_aligned_metrics 讓它支援兩種 mode
# ============================================================

def calculate_rwmd_metrics(data_a, data_b, gamma=2.0,
                           sym_mode="mean", mode="wordnet"):

    total_count_a = sum(d["count"] for d in data_a)
    total_count_b = sum(d["count"] for d in data_b)

    def empty_schema():
        empty = {"avg_d_token": 1.0, "avg_rwmd_total": 1.0}
        if mode == "wordnet":
            empty.update({"avg_d_synset": 1.0, "avg_d_lca": 1.0})
        else:
            empty.update({"avg_d_embed": 1.0})
        return {
            "a2b": empty, "b2a": empty,
            "symmetric": {"mode": sym_mode, **empty},
            "direction_gap": 0.0
        }

    if total_count_a == 0 or total_count_b == 0:
        return empty_schema()

    dist_matrix = []
    for d1 in data_a:
        if mode == "fasttext":
            row = [calculate_pair_metrics_ft(d1, d2) for d2 in data_b]
        else:
            row = [calculate_pair_metrics(d1, d2, gamma) for d2 in data_b]
        dist_matrix.append(row)

    def get_weighted_aligned_metrics(source_data, source_total, is_forward):
        sums = {"token": 0.0, "total": 0.0}
        if mode == "wordnet":
            sums.update({"synset": 0.0, "lca": 0.0})
        else:
            sums.update({"embed": 0.0})

        for i in range(len(source_data)):
            prob = source_data[i]["count"] / source_total

            if is_forward:
                best = min(dist_matrix[i], key=lambda x: x["total_divergence"])
            else:
                best = min(
                    [dist_matrix[row][i] for row in range(len(data_a))],
                    key=lambda x: x["total_divergence"]
                )

            sums["token"] += prob * best["d_token"]
            sums["total"] += prob * best["total_divergence"]

            if mode == "wordnet":
                sums["synset"] += prob * best["d_synset"]
                sums["lca"]    += prob * best["d_lca"]
            else:
                sums["embed"]  += prob * best["d_embed"]

        return sums

    m_a2b = get_weighted_aligned_metrics(data_a, total_count_a, True)
    m_b2a = get_weighted_aligned_metrics(data_b, total_count_b, False)

    def sym(x, y):
        if sym_mode == "max":   return max(x, y)
        elif sym_mode == "mean": return (x + y) / 2
        elif sym_mode == "rms":  return math.sqrt((x*x + y*y) / 2)
        elif sym_mode == "min":  return min(x, y)
        else: raise ValueError(f"Unknown sym_mode: {sym_mode}")

    def make_out(m):
        out = {
            "avg_d_token": round(m["token"], 4),
            "avg_rwmd_total": round(m["total"], 4),
        }
        if mode == "wordnet":
            out["avg_d_synset"] = round(m["synset"], 4)
            out["avg_d_lca"]    = round(m["lca"], 4)
        else:
            out["avg_d_embed"]  = round(m["embed"], 4)
        return out

    a2b_out = make_out(m_a2b)
    b2a_out = make_out(m_b2a)

    sym_keys = ["token", "total"]
    if mode == "wordnet":
        sym_keys += ["synset", "lca"]
    else:
        sym_keys += ["embed"]

    sym_out = {"mode": sym_mode}
    key_map = {"token": "avg_d_token", "total": "avg_rwmd_total",
               "synset": "avg_d_synset", "lca": "avg_d_lca", "embed": "avg_d_embed"}
    for k in sym_keys:
        sym_out[key_map[k]] = round(sym(m_a2b[k], m_b2a[k]), 4)

    return {
        "a2b": a2b_out,
        "b2a": b2a_out,
        "symmetric": sym_out,
        "direction_gap": round(abs(m_a2b["total"] - m_b2a["total"]), 4)
    }


# ============================================================
# 5. Main
# ============================================================

if __name__ == "__main__":
    base_dir = "outputs/0427_1000green_synset_flat"
    out_dir_in_domain   = f"{base_dir}/500algo"
    out_dir_cross_domain = f"{base_dir}/500crossdomain"

    algorithms = ["KGW", "SWEET", "Unigram", "EXP", "SynthID"]
    domains    = ["ai", "bio", "med", "mis", "security"]
    sym_mode   = "mean"
    top_k      = 500

    # # ── γ ablation（wordnet mode，cross-domain，nostop）──
    # for gamma in [0.0, 0.5, 1.0, 2.0]:
    #     for algo in algorithms:
    #         for domain_a, domain_b in itertools.combinations(domains, 2):
    #             file_a = f"{base_dir}/rewritten_{domain_a}_{algo}_synset_profile.json"
    #             file_b = f"{base_dir}/rewritten_{domain_b}_{algo}_synset_profile.json"
    #             compare_profiles(
    #                 file_a, file_b,
    #                 output_path=f"{out_dir_cross_domain}/nostop_{algo}_{domain_a}_vs_{domain_b}_gamma{gamma}.json",
    #                 exclude_stop=True,
    #                 top_k=top_k, gamma=gamma, sym_mode=sym_mode, mode="wordnet"
    #             )

    # # ── fasttext ablation（cross-domain，nostop）──
    # for algo in algorithms:
    #     for domain_a, domain_b in itertools.combinations(domains, 2):
    #         file_a = f"{base_dir}/rewritten_{domain_a}_{algo}_synset_profile.json"
    #         file_b = f"{base_dir}/rewritten_{domain_b}_{algo}_synset_profile.json"
    #         compare_profiles(
    #             file_a, file_b,
    #             output_path=f"{out_dir_cross_domain}/nostop_{algo}_{domain_a}_vs_{domain_b}_fasttext.json",
    #             exclude_stop=True,
    #             top_k=top_k, gamma=2.0, sym_mode=sym_mode, mode="fasttext"
    #         )

    # ── 原本的 in-domain 和 cross-domain（保持 gamma=2.0，wordnet）──
    gamma = 2.0

    # for domain in domains:
    #     for algo_a, algo_b in itertools.combinations(algorithms, 2):
    #         file_a = f"{base_dir}/rewritten_{domain}_{algo_a}_synset_profile.json"
    #         file_b = f"{base_dir}/rewritten_{domain}_{algo_b}_synset_profile.json"
    #         # for exclude_stop, prefix in [(False, "full"), (True, "nostop")]:
    #         #     compare_profiles(
    #         #         file_a, file_b,
    #         #         output_path=f"{out_dir_in_domain}/{prefix}_{domain}_{algo_a}_vs_{algo_b}.json",
    #         #         exclude_stop=exclude_stop,
    #         #         top_k=top_k, gamma=gamma, sym_mode=sym_mode, mode="wordnet"
    #         #     )
    #         compare_profiles(
    #             file_a, file_b,
    #             output_path=f"{out_dir_in_domain}/nostop_{domain}_{algo_a}_vs_{algo_b}.json",
    #             exclude_stop=True,
    #             top_k=top_k, gamma=2.0, sym_mode=sym_mode, mode="wordnet"
    #         )

    for algo in algorithms:
        for domain_a, domain_b in itertools.combinations(domains, 2):
            file_a = f"{base_dir}/rewritten_{domain_a}_{algo}_synset_profile.json"
            file_b = f"{base_dir}/rewritten_{domain_b}_{algo}_synset_profile.json"
            # for exclude_stop, prefix in [(False, "full"), (True, "nostop")]:
            #     compare_profiles(
            #         file_a, file_b,
            #         output_path=f"{out_dir_cross_domain}/{prefix}_{algo}_{domain_a}_vs_{domain_b}.json",
            #         exclude_stop=exclude_stop,
            #         top_k=top_k, gamma=gamma, sym_mode=sym_mode, mode="wordnet"
            #     )
            compare_profiles(
                file_a, file_b,
                output_path=f"{out_dir_cross_domain}/nostop_{algo}_{domain_a}_vs_{domain_b}.json",
                exclude_stop=True,
                top_k=top_k, gamma=2.0, sym_mode=sym_mode, mode="wordnet"
            )


    print("All experiments finished.")