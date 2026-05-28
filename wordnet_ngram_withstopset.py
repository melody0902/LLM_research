import os
import json
import re
from collections import Counter

import nltk
from nltk.corpus import wordnet as wn, stopwords
from nltk import pos_tag
from nltk.wsd import lesk  # ✅ 新增：必須引入 Lesk
from transformers import AutoTokenizer

# ====== 確保 NLTK 可用 ======
def ensure_nltk_resources():
    resources = [
        "stopwords",
        "punkt",
        "averaged_perceptron_tagger",
        "averaged_perceptron_tagger_eng",
        "wordnet",     # ✅ 確保 wordnet 本體
        "omw-1.4"      # ✅ 確保多語言支援 (有時需要)
    ]
    for r in resources:
        try:
            nltk.data.find(r)
        except LookupError:
            print(f"⬇️ 下載資源: {r}")
            nltk.download(r, quiet=True)

ensure_nltk_resources()

# ====== 初始化 ======
stop_words = set(stopwords.words("english"))
model_name = 'meta-llama/Llama-3.1-8B-Instruct'
tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)

# 輸入/輸出資料夾
input_dir = "ngram/signature_sets"
output_dir = "ngram/stopset"
os.makedirs(output_dir, exist_ok=True)

# ====== Helper Function: POS Tag 轉換 ======
# ✅ 新增：這是 Lesk 和 Head Word 判斷所需要的
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wn.ADJ
    elif treebank_tag.startswith('V'):
        return wn.VERB
    elif treebank_tag.startswith('N'):
        return wn.NOUN
    elif treebank_tag.startswith('R'):
        return wn.ADV
    else:
        return None

# ====== 核心功能 ======
def decode_signatures(ngram_data):
    """
    輸入: JSON 讀進來的 list，可能是純 ids [1, 2] 或 dict [{'ngram': [1,2], ...}]
    輸出: decoded_phrases (原本的 phrase list)
    """
    decoded_phrases = []

    for item in ngram_data:
        # ✅ 相容性處理：判斷輸入格式
        ids = []
        if isinstance(item, dict) and "ngram" in item:
            ids = item["ngram"]
        elif isinstance(item, list):
            ids = item
        else:
            continue # 格式不符跳過

        try:
            text = tokenizer.decode(ids).strip()
            # 移除 BPE artifact (如 "Ġ")
            cleaned = re.sub(r"[Ġ]+", " ", text).strip()
            phrase = cleaned.lower()
            if phrase:
                decoded_phrases.append(phrase)
        except Exception:
            continue

    return decoded_phrases

# ✅ 整合後的 Synset 取得函式 (含 N=1 處理 & Lesk)
def get_representative_synset(tokens):
    """
    為 N-gram (N>=1) 找出最合適的 WordNet Synset。
    策略：
    1. [N=1] 單字模式：回傳 MFS (synsets[0])。
    2. [N>1] 片語模式：MWE -> Head Word + Lesk。
    """
    if not tokens:
        return None

    # 清理 token
    clean_tokens = [t.strip().lower() for t in tokens if t.strip()]
    if not clean_tokens:
        return None

    # === 情況 A：單字 (Unigram) ===
    if len(clean_tokens) == 1:
        word = clean_tokens[0]
        synsets = wn.synsets(word)
        # 如果只有一個字，沒有上下文，統計上選最常用的意思 (MFS) 是最佳解
        return synsets[0] if synsets else None

    # === 情況 B：多字 (Bigram, Trigram...) ===
    # 1. MWE 檢查 (例如 credit_card)
    mwe_query = "_".join(clean_tokens)
    mwe_synsets = wn.synsets(mwe_query)
    if mwe_synsets:
        return mwe_synsets[0]

    # 2. 尋找 Head Word (詞性標註)
    tagged_tokens = pos_tag(clean_tokens)
    head_word = None
    head_pos = None
    
    # 策略：找最右邊的名詞 -> 最右邊的動詞 -> 最後一個字
    for word, tag in reversed(tagged_tokens):
        wn_pos = get_wordnet_pos(tag)
        if wn_pos == wn.NOUN:
            head_word = word
            head_pos = wn.NOUN
            break
    
    if not head_word:
        for word, tag in reversed(tagged_tokens):
            wn_pos = get_wordnet_pos(tag)
            if wn_pos == wn.VERB:
                head_word = word
                head_pos = wn.VERB
                break
                
    if not head_word:
        head_word = clean_tokens[-1]
        # 嘗試取得最後一個字的 POS，若無法對應則為 None
        head_pos = get_wordnet_pos(tagged_tokens[-1][1])

    # 3. Lesk 消歧義 (利用 N-gram 內部當作上下文)
    # 例如 ["river", "bank"] -> Lesk 會判斷 bank 是河岸
    best_synset = lesk(clean_tokens, head_word, pos=head_pos)

    if best_synset:
        return best_synset
    
    # 4. 退回查字典 (Fallback)
    fallback = wn.synsets(head_word, pos=head_pos)
    return fallback[0] if fallback else None

# Stop set：過於泛化的上位詞 (保持不變)
STOP_SYNSETS = {
    "entity.n.01", "physical_entity.n.01", "abstraction.n.06", "object.n.01",
    "thing.n.12", "psychological_feature.n.01", "relation.n.01",
    "attribute.n.02", "group.n.01", "act.n.02", "activity.n.01",
    "process.n.06", "state.n.02", "event.n.01", "state.n.01", 
    "condition.n.01", "situation.n.01", "case.n.01"
}

def get_concept_levels(syn, levels=4):
    """往上追 hypernym path，跳過 Stop set"""
    if not syn:
        return {}
    paths = syn.hypernym_paths()
    if not paths:
        return {}

    path = max(paths, key=len)
    candidates = [c for c in reversed(path) if c.name() not in STOP_SYNSETS]

    result = {}
    for idx, c in enumerate(candidates[:levels]):
        try:
            definition = c.definition()
        except:
            definition = "N/A"
        result[f"level{idx+1}"] = {
            "name": c.name(),
            "definition": definition
        }
    return result

# ====== 主流程 ======
def process_file(file_path, file_name):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print(f"❌ 讀取錯誤: {file_name}")
        return

    # 若 JSON 是 list (signature sets)，data 本身就是 list
    # 若 JSON 是 dict (其他格式)，嘗試取 key
    ngram_data = data.get("signatures", data) if isinstance(data, dict) else data

    # 1. 解碼 (取得 Phrase 字串)
    decoded_phrases = decode_signatures(ngram_data)

    synset_results = []
    concept_levels = {"level1": [], "level2": [], "level3": [], "level4": []}

    print(f"  - 處理 {len(decoded_phrases)} 個 Phrases...")

    # 2. 針對每個 Phrase 找 Synset
    for phrase in decoded_phrases:
        phrase_tokens = phrase.split()
        
        # ⭐ 使用整合後的函式
        syn = get_representative_synset(phrase_tokens)
        
        # 3. 取得概念層級
        levels = get_concept_levels(syn, levels=4)
        if levels:
            for k, v in levels.items():
                concept_levels[k].append(v)
                if k == "level1" and "name" in v: # 統計 Level 1 概念
                    synset_results.append(v["name"])

    if not synset_results:
        print(f"⚠️ {file_name} 沒有找到代表 synset")
        return

    counts = Counter(synset_results)
    top_list = []
    for name, count in counts.most_common(5): # 取前 5 名
        try:
            syn = wn.synset(name)
            definition = syn.definition()
        except:
            definition = "N/A"
        top_list.append({
            "name": name,
            "count": count,
            "definition": definition
        })

    result = {
        "file": file_name,
        "n": data.get("n", "unknown") if isinstance(data, dict) else "unknown",
        "decoded_sample": decoded_phrases[:10], # 存前10個當樣本
        "top_concepts": top_list,
        # concept_levels 若資料量大建議不要全存，或只存統計
        "concept_counts": dict(counts.most_common(20)) 
    }

    result_path = os.path.join(output_dir, file_name.replace(".json", "_analysis.json"))
    with open(result_path, "w", encoding="utf-8") as out_f:
        json.dump(result, out_f, ensure_ascii=False, indent=2)

    print(f"✅ {file_name} 已完成，Top Concept: {top_list[0]['name'] if top_list else 'None'}")


# ====== 執行 ======
if __name__ == "__main__":
    files = [f for f in os.listdir(input_dir) if f.endswith(".json")]
    for file in sorted(files):
        print(f"\n--- 分析檔案：{file} ---")
        process_file(os.path.join(input_dir, file), file)