import json
import glob
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_heatmaps(compare_dir, output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)
    all_res = []
    
    # 讀取所有 JSON
    for f in glob.glob(f"{compare_dir}/*.json"):
        with open(f, 'r') as j:
            data = json.load(j)
            fname = os.path.basename(f).replace(".json", "")
            parts = fname.split("_") # [type, domain, algoA, vs, algoB]
            
            all_res.append({
                "Type": parts[0],
                "Domain": parts[1],
                "AlgoA": parts[2],
                "AlgoB": parts[4],
                "Total_Div": data["results"]["avg_rwmd_total"]
            })
            # 為了熱圖對稱，補上 B vs A
            all_res.append({
                "Type": parts[0], "Domain": parts[1],
                "AlgoA": parts[4], "AlgoB": parts[2],
                "Total_Div": data["results"]["avg_rwmd_total"]
            })

    df = pd.DataFrame(all_res)

    # 為每個領域產出一張熱圖
    for domain in df['Domain'].unique():
        for t in df['Type'].unique():
            sub = df[(df['Domain'] == domain) & (df['Type'] == t)]
            pivot = sub.pivot(index="AlgoA", columns="AlgoB", values="Total_Div").fillna(0)
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(pivot, annot=True, cmap="YlGnBu", fmt=".3f")
            plt.title(f"Semantic Divergence - Domain: {domain.upper()} ({t.upper()})")
            plt.savefig(f"{output_dir}/heatmap_{domain}_{t}.png", dpi=300, bbox_inches='tight')
            plt.close()

if __name__ == "__main__":
    plot_heatmaps("outputs/1221/0116compare")
    print("可視化圖表已生成至 plots/ 資料夾。")