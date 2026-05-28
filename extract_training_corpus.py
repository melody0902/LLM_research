import json

datasets = [
    ("dataset/zhtw/mydatasets/ai/output_data_combined_iclr_abstracts.json", "ai"),
    ("dataset/zhtw/mydatasets/bio/output_data_combined_BIO2_abstracts.json", "bio"),
    ("dataset/zhtw/mydatasets/med/output_data_combined_MIE_abstracts.json", "med"),
]

output_path = "training_corpus.txt"


def extract_text(sample):
    """
    從每筆 JSON 中萃取可用文本
    """
    for key in ["natural_text", "text", "prompt", "content", "abstract"]:
        if key in sample and isinstance(sample[key], str):
            return sample[key]
    return ""


def clean_text(t):
    """
    清理文本格式
    """
    if not isinstance(t, str):
        return ""
    t = t.replace("\n", " ").replace("\r", " ")
    return " ".join(t.split()).strip()


def load_jsonl(path):
    """
    強制以 JSONL 逐行讀取
    """
    items = []
    print(f"📘 強制 JSONL 讀取：{path}")

    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                items.append(obj)
            except json.JSONDecodeError:
                print(f"⚠️ 第 {i+1} 行無法解析，略過：{line[:50]}...")
                continue

    print(f"📄 找到 {len(items)} 筆資料")
    return items


def main():
    all_lines = []

    for path, domain in datasets:
        data = load_jsonl(path)

        for entry in data:
            text = extract_text(entry)
            text = clean_text(text)
            if text:
                all_lines.append(text)

    print(f"✏️ 最終取得 {len(all_lines)} 行訓練資料")
    
    with open(output_path, "w", encoding="utf-8") as f:
        for line in all_lines:
            f.write(line + "\n")

    print(f"✅ 已輸出資料到：{output_path}")


if __name__ == "__main__":
    main()
