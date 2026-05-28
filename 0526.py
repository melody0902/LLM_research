import os
import re
import json
from collections import Counter

output_dir = "/home/soslab/Desktop/Melody/signature/llm-watermark-research/outputs/0517_200green"

algorithms = ["KGW", "SWEET", "Unigram", "EXP", "SynthID"]
domains = ["bio", "med", "mis", "security"]
model_safe_name = "meta-llama__Llama-3.1-8B-Instruct"

save_dir = os.path.join(output_dir, "favored_vs_plain_ratio")
os.makedirs(save_dir, exist_ok=True)


def tokenize_plain(text):
    return re.findall(
        r"\b[a-zA-Z0-9]+(?:'[a-zA-Z0-9]+)?\b",
        text.lower()
    )


for algo in algorithms:
    for domain in domains:
        base_name = f"rewritten_{domain}_{algo}_{model_safe_name}"

        wm_freq_path = os.path.join(
            output_dir,
            f"{base_name}_wm_token_freq.json"
        )

        wm_tokens_path = os.path.join(
            output_dir,
            f"{base_name}_wm_tokens.json"
        )

        if not os.path.exists(wm_freq_path):
            print(f"Skip missing wm freq: {wm_freq_path}")
            continue

        if not os.path.exists(wm_tokens_path):
            print(f"Skip missing wm tokens: {wm_tokens_path}")
            continue

        with open(wm_freq_path, "r", encoding="utf-8") as f:
            favored_data = json.load(f)

        total_favored = sum(x["count"] for x in favored_data)

        with open(wm_tokens_path, "r", encoding="utf-8") as f:
            plain_data = json.load(f)

        plain_counter = Counter()
        total_plain = 0

        for item in plain_data:
            words = tokenize_plain(item.get("plain", ""))
            plain_counter.update(words)
            total_plain += len(words)

        results = []

        for item in favored_data:
            token = item["token"].strip().lower()
            favored_count = item["count"]

            plain_count = plain_counter[token]

            favored_ratio = favored_count / total_favored if total_favored > 0 else 0
            plain_ratio = plain_count / total_plain if total_plain > 0 else 0

            enrichment = (
                favored_ratio / plain_ratio
                if plain_ratio > 0
                else None
            )

            results.append({
                "domain": domain,
                "algorithm": algo,
                "token_id": item.get("token_id"),
                "token": token,
                "favored_count": favored_count,
                "plain_count": plain_count,
                "favored_ratio": favored_ratio,
                "plain_ratio": plain_ratio,
                "enrichment": enrichment
            })

        results = sorted(
            results,
            key=lambda x: x["enrichment"] if x["enrichment"] is not None else -1,
            reverse=True
        )

        out_path = os.path.join(
            save_dir,
            f"favored_vs_plain_ratio_{domain}_{algo}.json"
        )

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"Saved: {out_path}")