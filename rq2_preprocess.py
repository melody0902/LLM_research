import os
import json
import glob
import csv

# ==============================
# config
# ==============================
BASE_DIR = "outputs/0409_500green_synset_flat/1000crossdomain"
# BASE_DIR = "outputs/0123_200green_synset_flat/1000crossdomain"
OUTPUT_CSV = "rq2_crossdomain_master0428_1000.csv"

# ==============================
# helper
# ==============================
def parse_filename(path):
    """
    解析檔名以提取實驗設定
    範例: nostop_KGW_ai_vs_med.json -> ('nostop', 'KGW', 'ai', 'med', 'ai__med')
    """
    base = os.path.basename(path).replace(".json", "")
    parts = base.split("_")
    
    # 根據你的檔名結構調整索引
    setting = parts[0]   # 'nostop' or 'full'
    algo = parts[1]      # 'KGW', 'SWEET', etc.
    domain_a = parts[2]
    domain_b = parts[4]
    pair = f"{domain_a}__{domain_b}"

    return setting, algo, domain_a, domain_b, pair

# ==============================
# main
# ==============================
rows = []
files = glob.glob(os.path.join(BASE_DIR, "*.json"))

print(f"Found {len(files)} files in {BASE_DIR}")

for file_path in files:
    try:
        setting, algo, domain_a, domain_b, pair = parse_filename(file_path)

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        results = data["results"]

        row = {
            "setting": setting, # 重要：修正與統計腳本的對接
            "algorithm": algo,
            "domain_a": domain_a,
            "domain_b": domain_b,
            "pair": pair,
            "rwmd_sym": results["symmetric"]["avg_rwmd_total"],
            "d_token": results["symmetric"]["avg_d_token"],
            "d_synset": results["symmetric"]["avg_d_synset"],
            "d_lca": results["symmetric"]["avg_d_lca"],
            "rwmd_a2b": results["a2b"]["avg_rwmd_total"],
            "rwmd_b2a": results["b2a"]["avg_rwmd_total"],
            "direction_gap": results.get("direction_gap", 0) # 使用 get 防止 key 缺失
        }
        rows.append(row)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

# ==============================
# save CSV
# ==============================
fieldnames = ["setting", "algorithm", "domain_a", "domain_b", "pair", 
              "rwmd_sym", "d_token", "d_synset", "d_lca", 
              "rwmd_a2b", "rwmd_b2a", "direction_gap"]

with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"\nSuccessfully saved CSV to: {OUTPUT_CSV}")
print(f"Total rows: {len(rows)}")