import os
import json
from collections import Counter
from transformers import AutoTokenizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# ====== 基本設定 ======
model_name = "facebook/opt-1.3b"  # 需和簽名生成時一致
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 參數設定
algorithms = ["KGW", "SWEET", "Unigram"]
domains = ["ai", "bio", "security", "med", "law", "mis", "edu"]
ns = [2, 3, 4]

# 輸入和輸出路徑
input_dir = "ngram/signature_sets"
output_dir = "ngram/wordcloud_images"
os.makedirs(output_dir, exist_ok=True)

# ====== 主迴圈 ======
for algo in algorithms:
    for domain in domains:
        for n in ns:
            algo_lower = algo.lower()
            input_file = f"{input_dir}/{algo_lower}_{domain}_sig_n{n}.json"
            output_file = f"{output_dir}/{algo_lower}_{domain}_n{n}_wordcloud.png"

            if not os.path.exists(input_file):
                print(f"❌ 找不到檔案: {input_file}，跳過")
                continue

            print(f"✅ 處理中: {input_file}")

            # 讀取 JSON
            with open(input_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            signatures = data.get("signatures", [])

            # 解碼 n-gram
            decoded_ngrams = []
            for ngram in signatures:
                try:
                    tokens = tokenizer.decode(ngram)
                    decoded_ngrams.append(tokens)
                except Exception as e:
                    print(f"解碼失敗: {ngram} - {e}")

            # 統計頻率
            ngram_counts = Counter(decoded_ngrams)

            if not ngram_counts:
                print(f"⚠️ 沒有可用的 n-gram: {input_file}")
                continue

            # 產生 WordCloud
            wc = WordCloud(width=1600, height=800, background_color="white")
            wc.generate_from_frequencies(ngram_counts)

            # 儲存圖片
            plt.figure(figsize=(16, 8))
            plt.imshow(wc, interpolation="bilinear")
            plt.axis("off")
            plt.title(f"{algo} - {domain} - n={n}")
            plt.savefig(output_file, dpi=300, bbox_inches="tight")
            plt.close()

            print(f"💾 已儲存 WordCloud: {output_file}")

print("🎉 所有圖片已完成輸出！")
