import json
import os
import glob
import re
import itertools
import nltk
import numpy as np

from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords


# ============================================================
# 0. Setup
# ============================================================
try:
    wn.ensure_loaded()
except:
    nltk.download("wordnet")
    nltk.download("omw-1.4")

try:
    stopwords.words("english")
except:
    nltk.download("stopwords")


# ============================================================
# 1. Filename parser
# ============================================================
def parse_filename(filename):
    base = os.path.basename(filename)

    pattern = re.compile(
        r"^rewritten_(?P<domain>[^_]+)_(?P<algo>[^_]+)_(?P<model>.+)_wm_token_freq\.json$"
    )

    m = pattern.match(base)
    if not m:
        return "unknown", "unknown", "unknown"

    domain = m.group("domain")
    algo = m.group("algo")
    model = m.group("model").replace("__", "/")

    return domain, algo, model


# ============================================================
# 2. Token → Synset
# ============================================================
def get_synset(word):
    if not word:
        return None
    word = word.strip().lower()
    synsets = wn.synsets(word)
    return synsets[0] if synsets else None


def generate_synset_profile(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    result = []
    for item in data:
        if isinstance(item, dict):
            token = item.get("token") or item.get("word")
            count = item.get("count", 1)
        else:
            token = item
            count = 1

        if not token or not token.replace(" ", "").isalpha():
            continue

        syn = get_synset(token)

        result.append({
            "token": token,
            "synset": syn.name() if syn else None,
            "count": count
        })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return output_path


# ============================================================
# 3. WordNet distance (替代 fastText)
# ============================================================
def wordnet_distance(a, b):
    if a["token"].lower() == b["token"].lower():
        return 0.0

    syn_a = wn.synset(a["synset"]) if a["synset"] else None
    syn_b = wn.synset(b["synset"]) if b["synset"] else None

    if syn_a and syn_b:
        sim = syn_a.wup_similarity(syn_b)
        if sim is not None:
            return 1.0 - sim  # similarity → distance

    return 1.0  # fallback 最大距離


# ============================================================
# 4. RWMD
# ============================================================
def pair_dist(a, b):
    return wordnet_distance(a, b)


def rwmd(A, B):
    totalA = sum(x["count"] for x in A)
    totalB = sum(x["count"] for x in B)

    def align(src, tgt, total):
        s = 0
        for x in src:
            p = x["count"] / total
            best = min(pair_dist(x, y) for y in tgt)
            s += p * best
        return s

    a2b = align(A, B, totalA)
    b2a = align(B, A, totalB)

    return {
        "a2b": round(a2b, 4),
        "b2a": round(b2a, 4),
        "mean": round((a2b + b2a)/2, 4)
    }


# ============================================================
# 5. Pipeline
# ============================================================
def run_pipeline(step1_dir, synset_dir, report_dir):

    files = glob.glob(os.path.join(step1_dir, "**", "*_token_freq.json"), recursive=True)

    synset_files = []

    print(f"Found {len(files)} token files")

    # Step 1: generate synsets
    for f in files:
        domain, algo, model = parse_filename(f)
        safe_model = model.replace("/", "__")

        out = os.path.join(
            synset_dir,
            f"rewritten_{domain}_{algo}_{safe_model}_synset_profile.json"
        )

        syn_path = generate_synset_profile(f, out)
        synset_files.append(syn_path)

    # Step 2: compare all pairs
    os.makedirs(report_dir, exist_ok=True)

    for f1, f2 in itertools.combinations(synset_files, 2):

        with open(f1) as a, open(f2) as b:
            A = json.load(a)
            B = json.load(b)

        result = rwmd(A[:300], B[:300])

        name = os.path.basename(f1) + "_VS_" + os.path.basename(f2)
        out = os.path.join(report_dir, name + ".json")

        with open(out, "w") as f:
            json.dump(result, f, indent=2)

    print("Pipeline done")


# ============================================================
# main
# ============================================================
if __name__ == "__main__":
    run_pipeline(
        step1_dir="outputs/0315_200green",
        synset_dir="outputs/0421/synset",
        report_dir="outputs/0421/report"
    )