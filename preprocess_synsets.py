import json
import os
import glob
import nltk
from nltk.corpus import wordnet as wn


try:
    wn.ensure_loaded()
except:
    nltk.download("wordnet")
    nltk.download("omw-1.4")


def get_synset_name(word):
    if not word:
        return None

    clean_word = word.strip().lower().replace("ġ", "")
    if not clean_word:
        return None

    synsets = wn.synsets(clean_word)
    if synsets:
        return synsets[0].name()
    return None


def generate_synset_profile(input_path, output_path):
    print(f"處理檔案: {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    processed_data = []
    items = data if isinstance(data, list) else []

    for item in items:
        token_text = ""
        count = 1

        if isinstance(item, dict):
            token_text = item.get("token") or item.get("word")
            count = item.get("count", 1)
        elif isinstance(item, str):
            token_text = item

        # 只保留純字母
        if not token_text or not token_text.replace(" ", "").isalpha():
            continue

        syn_name = get_synset_name(token_text)

        processed_data.append({
            "token": token_text,
            "synset": syn_name,
            "count": count
        })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)

    print(f" 已儲存: {output_path} (共 {len(processed_data)} 筆)\n")


def extract_domain_algo(filename):
    """
    從檔名推測 domain + algo
    例：rewritten_ai_SynthID_wm_token_freq.json
        -> domain=ai, algo=SynthID
    """
    base = os.path.basename(filename)

    # 去掉前綴 rewritten_
    if base.startswith("rewritten_"):
        base = base.replace("rewritten_", "", 1)

    # 去掉後綴
    base = base.replace("_wm_token_freq.json", "")
    base = base.replace("_token_freq.json", "")

    parts = base.split("_")
    if len(parts) >= 2:
        domain = parts[0]
        algo = parts[1]
        return domain, algo

    # fallback
    return "unknown", "unknown"


def batch_generate_synset_profiles_flat(step1_output_dir, step2_output_dir):
    """
     從 step1_output_dir 掃描所有 *_token_freq.json
     synset_profile.json 全部輸出到 step2_output_dir (不建立子資料夾)
    """
    pattern = os.path.join(step1_output_dir, "**", "*_token_freq.json")
    input_files = glob.glob(pattern, recursive=True)

    if not input_files:
        print(f"找不到任何 token_freq.json，請確認 Step1 output_dir: {step1_output_dir}")
        return

    print(f"🔍 找到 {len(input_files)} 個 token_freq.json\n")

    os.makedirs(step2_output_dir, exist_ok=True)

    for input_path in input_files:
        domain, algo = extract_domain_algo(input_path)

        #  扁平化輸出：全部放同一個資料夾
        output_file = f"rewritten_{domain}_{algo}_synset_profile.json"
        output_path = os.path.join(step2_output_dir, output_file)

        generate_synset_profile(input_path, output_path)


# ============================================================
# main
# ============================================================
if __name__ == "__main__":
    step1_output_dir = "outputs/0517_200green"
    step2_output_dir = "outputs/0517_200green_synset_flat"

    batch_generate_synset_profiles_flat(step1_output_dir, step2_output_dir)

