# import json
# import nltk
# from nltk.corpus import wordnet as wn

# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger_eng')
# nltk.download('punkt_tab')

# CONTENT_POS = {
#     'NN', 'NNS', 'NNP', 'NNPS',
#     'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',
#     'JJ', 'JJR', 'JJS',
#     'RB', 'RBR', 'RBS'
# }

# def wordnet_coverage(jsonl_path, domain_name):
#     texts = []
#     with open(jsonl_path, 'r') as f:
#         for line in f:
#             obj = json.loads(line)
#             texts.append(obj["prompt"])

#     unique_words = set()
#     covered = set()

#     for text in texts:
#         tokens = nltk.word_tokenize(text.lower())
#         tagged = nltk.pos_tag(tokens)
#         for word, pos in tagged:
#             if pos in CONTENT_POS and word.isalpha():
#                 unique_words.add(word)
#                 if wn.synsets(word):
#                     covered.add(word)

#     ratio = len(covered) / len(unique_words) if unique_words else 0
#     print(f"[{domain_name}] {len(covered)}/{len(unique_words)} = {ratio:.2%}")
#     return ratio

# datasets = [
#     ("dataset/zhtw/mydatasets/ai/output_data_combined_iclr_abstracts_merged_prompt.jsonl", "ai"),
#     ("dataset/zhtw/mydatasets/bio/output_data_combined_BIO2_abstracts_merged_prompt.jsonl", "bio"),
#     ("dataset/zhtw/mydatasets/med/output_data_combined_MIE_abstracts_merged_prompt.jsonl", "med"),
#     ("dataset/zhtw/mydatasets/mis/combined_icis_merged_prompt.jsonl", "mis"),
#     ("dataset/zhtw/mydatasets/Security/output_data_combined_SP_abstracts_merged_prompt.jsonl", "security"),
# ]

# for path, domain in datasets:
#     wordnet_coverage(path, domain)



import json
import math
import torch
import numpy as np
from tqdm import tqdm
import os

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from watermark.auto_watermark import AutoWatermark
from utils.transformers_config import TransformersConfig
from evaluation.baseline_detectors import BaselineDetectorFactory
from watermark.synthid.detector import get_detector

# ============================================================
# 1. Global settings
# ============================================================

ALGORITHMS = ["KGW", "SWEET", "Unigram", "EXP", "SynthID"]
BASE_ALGO_FOR_PLAIN = "KGW"
DOMAIN = "bio"  # 只跑一個 domain 做對比驗證

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
MODEL_TAG = MODEL_NAME.replace("/", "__")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TOP_K_TOKENS = 200

TEXT_JSON_TMPL = (
    "outputs/0517_200green/"
    "rewritten_{domain}_{alg}_{model_tag}_wm_tokens.json"
)
TOKEN_SET_TMPL = "outputs/0305_200test/rewritten_{domain}_{alg}_wm_token_freq.json"

OUTPUT_PATH = (
    f"outputs/test/detect/"
    f"fullprecision_subset_vs_baseline_{DOMAIN}_{MODEL_TAG}_top{TOP_K_TOKENS}.json"
)


# ============================================================
# 2. Transformers config (full-precision, no quantization)
# ============================================================

def get_transformers_config():
    print(f"Using model: {MODEL_NAME} (full-precision bfloat16, no quantization)")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",       # 改這行，自動分配 GPU+CPU
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    return TransformersConfig(
        model=model,
        tokenizer=tokenizer,
        vocab_size=len(tokenizer),
        device=DEVICE,
        max_new_tokens=200,
        do_sample=False,
    )


# ============================================================
# 3. Helpers
# ============================================================

def load_texts(path, key):
    with open(path, "r", encoding="utf-8") as f:
        return [x[key] for x in json.load(f)]


def load_token_subset(path, top_k=None):
    with open(path, "r", encoding="utf-8") as f:
        freq = json.load(f)
    if top_k is not None:
        freq = freq[:top_k]
    return {x["token_id"] for x in freq}


# ============================================================
# 3.1 EXP evidence computation
# ============================================================

def exp_pvalue_and_avgscore(wm, text, eps=1e-12):
    import scipy.stats

    tokenizer = wm.config.generation_tokenizer
    prefix_len = wm.config.prefix_length
    vocab_size = wm.config.vocab_size

    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]

    total_score = 0.0
    num_scored = 0

    for i in range(prefix_len, len(ids)):
        t = ids[i].item()
        wm.utils.seed_rng(ids[:i])
        r = torch.rand(vocab_size, generator=wm.utils.rng)[t].item()
        r = min(max(r, 0.0), 1.0 - eps)
        total_score += -math.log(1.0 - r)
        num_scored += 1

    avg_score = (total_score / num_scored) if num_scored > 0 else 0.0
    p_value = scipy.stats.gamma.sf(total_score, num_scored, loc=0, scale=1) if num_scored > 0 else 1.0

    return {
        "p_value": float(p_value),
        "avg_score": float(avg_score),
        "total_score": float(total_score),
        "num_scored": int(num_scored),
    }


# ============================================================
# 4. Metric + threshold sweep
# ============================================================

def find_best_threshold_both(wm_scores, plain_scores):
    scores = wm_scores + plain_scores
    labels = [1] * len(wm_scores) + [0] * len(plain_scores)

    best = None

    for direction in ["gt", "lt"]:
        best_f1 = -1
        best_metrics = None

        for thr in sorted(set(scores)):
            preds = [1 if s > thr else 0 for s in scores] if direction == "gt" \
                else [1 if s < thr else 0 for s in scores]

            TP = sum(p == 1 and y == 1 for p, y in zip(preds, labels))
            FP = sum(p == 1 and y == 0 for p, y in zip(preds, labels))
            FN = sum(p == 0 and y == 1 for p, y in zip(preds, labels))
            TN = sum(p == 0 and y == 0 for p, y in zip(preds, labels))

            TPR = TP / (TP + FN) if (TP + FN) else 0.0
            precision = TP / (TP + FP) if (TP + FP) else 0.0
            FPR = FP / (FP + TN) if (FP + TN) else 0.0
            F1 = 2 * precision * TPR / (precision + TPR) if (precision + TPR) else 0.0

            if F1 > best_f1:
                best_f1 = F1
                best_metrics = {
                    "TPR": TPR, "F1": F1,
                    "precision": precision, "FPR": FPR,
                    "threshold": thr, "direction": direction,
                }

        if best is None or best_metrics["F1"] > best["F1"]:
            best = best_metrics

    return best


# ============================================================
# 5. Subset-aware detectors
# ============================================================

def _get_gamma_from_wm(wm):
    for name in ["gamma", "greenlist_ratio", "greenlist_fraction", "watermark_gamma"]:
        if hasattr(wm.config, name):
            g = getattr(wm.config, name)
            if g is not None:
                return float(g)
    raise AttributeError("Cannot find gamma in wm.config.")


def _zscore(obs, n, p, eps=1e-12):
    if n <= 0:
        return 0.0
    return float((obs - n * p) / math.sqrt(n * p * (1.0 - p) + eps))


class SubsetAwareKGWDetector:
    def __init__(self, wm, S):
        self.wm = wm
        self.utils = wm.utils
        self.tokenizer = wm.config.generation_tokenizer
        self.prefix_len = wm.config.prefix_length
        self.S = set(S)
        self.gamma = _get_gamma_from_wm(wm)

    def detect(self, text):
        ids = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(self.wm.config.device)
        obs, n = 0, 0
        for i in range(self.prefix_len, len(ids)):
            t = ids[i].item()
            if t not in self.S:
                continue
            n += 1
            if t in self.utils.get_greenlist_ids(ids[:i]):
                obs += 1
        return _zscore(obs, n, self.gamma)


class SubsetAwareSWEETDetector(SubsetAwareKGWDetector):
    def __init__(self, wm, S):
        super().__init__(wm, S)
        self.model = wm.config.generation_model
        self.entropy_threshold = wm.config.entropy_threshold

    def detect(self, text):
        ids = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(self.wm.config.device)
        entropy = self.utils.calculate_entropy(self.model, ids)
        obs, n = 0, 0
        for i in range(self.prefix_len, len(ids)):
            if entropy[i] <= self.entropy_threshold:
                continue
            t = ids[i].item()
            if t not in self.S:
                continue
            n += 1
            if t in self.utils.get_greenlist_ids(ids[:i]):
                obs += 1
        return _zscore(obs, n, self.gamma)


class SubsetAwareUnigramDetector:
    def __init__(self, wm, S):
        self.mask = wm.utils.mask
        self.tokenizer = wm.config.generation_tokenizer
        self.S = set(S)
        self.gamma = _get_gamma_from_wm(wm)

    def detect(self, text):
        ids = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0].tolist()
        obs, n = 0, 0
        for t in ids:
            if t not in self.S:
                continue
            n += 1
            if self.mask[t]:
                obs += 1
        return _zscore(obs, n, self.gamma)


class SubsetAwareEXPDetector:
    def __init__(self, wm, S):
        self.wm = wm
        self.utils = wm.utils
        self.tokenizer = wm.config.generation_tokenizer
        self.prefix_len = wm.config.prefix_length
        self.vocab_size = wm.config.vocab_size
        self.S = set(S)

    def detect(self, text, eps=1e-12):
        ids = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        score_sum, n = 0.0, 0
        for i in range(self.prefix_len, len(ids)):
            t = ids[i].item()
            if t not in self.S:
                continue
            self.utils.seed_rng(ids[:i])
            r = torch.rand(self.vocab_size, generator=self.utils.rng)[t].item()
            r = min(max(r, 0.0), 1.0 - eps)
            score_sum += -math.log(1.0 - r)
            n += 1
        return score_sum / n if n > 0 else 0.0


class SubsetAwareSynthIDDetector:
    def __init__(self, wm, token_subset, detector_name="mean"):
        self.wm = wm
        self.lp = wm.logits_processor
        self.cfg = wm.config
        self.tokenizer = self.cfg.generation_tokenizer
        self.S = set(token_subset)
        self.detector = get_detector(detector_name, self.lp)

    def detect(self, text):
        device = self.lp.device
        input_ids = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
        g_values = self.lp.compute_g_values(input_ids)
        eos_mask = self.lp.compute_eos_token_mask(
            input_ids=input_ids,
            eos_token_id=self.tokenizer.eos_token_id
        )[:, self.cfg.ngram_len - 1:]

        if self.cfg.watermark_mode == "non-distortionary":
            context_mask = self.lp.compute_context_repetition_mask(input_ids)
            mask_orig = eos_mask * context_mask
        else:
            mask_orig = eos_mask

        token_ids = input_ids[0, self.cfg.ngram_len - 1:].cpu().numpy()
        mask_subset = np.array([[1 if t in self.S else 0 for t in token_ids]])
        mask_final = mask_orig.cpu().numpy() * mask_subset

        return float(self.detector.detect(g_values.cpu().numpy(), mask_final)[0])


SUBSET_DETECTORS = {
    "KGW": SubsetAwareKGWDetector,
    "SWEET": SubsetAwareSWEETDetector,
    "Unigram": SubsetAwareUnigramDetector,
    "EXP": SubsetAwareEXPDetector,
    "SynthID": SubsetAwareSynthIDDetector,
}


# ============================================================
# 6. Main
# ============================================================

def main():
    print("=" * 80)
    print(f"Full-Precision Detection (no quantization) — domain: {DOMAIN}")
    print("=" * 80)

    cfg = get_transformers_config()

    plain_texts = load_texts(
        TEXT_JSON_TMPL.format(domain=DOMAIN, alg=BASE_ALGO_FOR_PLAIN, model_tag=MODEL_TAG, top_k=TOP_K_TOKENS),
        "plain"
    )

    results = {}

    for alg in ALGORITHMS:
        print("-" * 80)
        print(f"Algorithm: {alg}")

        wm_texts = load_texts(
            TEXT_JSON_TMPL.format(domain=DOMAIN, alg=alg, model_tag=MODEL_TAG, top_k=TOP_K_TOKENS),
            "rewritten"
        )

        token_subset = load_token_subset(
            TOKEN_SET_TMPL.format(domain=DOMAIN, alg=alg, model_tag=MODEL_TAG, top_k=TOP_K_TOKENS),
            TOP_K_TOKENS
        )

        wm = AutoWatermark.load(alg, f"config/{alg}.json", cfg)

        if alg == "SynthID":
            baseline_detector_fn = BaselineDetectorFactory(wm).build().detect
        else:
            baseline_detector_fn = lambda text, _wm=wm: _wm.detect_watermark(text, return_dict=True)["score"]

        subset_detector = SUBSET_DETECTORS[alg](wm, token_subset)

        wm_base, plain_base = [], []
        wm_subset, plain_subset = [], []
        wm_base_avg, plain_base_avg = [], []  # EXP only

        for w, p in tqdm(zip(wm_texts, plain_texts), total=len(plain_texts)):
            wm_base.append(baseline_detector_fn(w))
            plain_base.append(baseline_detector_fn(p))

            wm_subset.append(subset_detector.detect(w))
            plain_subset.append(subset_detector.detect(p))

            if alg == "EXP":
                wm_base_avg.append(exp_pvalue_and_avgscore(wm, w)["avg_score"])
                plain_base_avg.append(exp_pvalue_and_avgscore(wm, p)["avg_score"])

        entry = {
            "subset_size": len(token_subset),
            "baseline": find_best_threshold_both(wm_base, plain_base),
            "subset": find_best_threshold_both(wm_subset, plain_subset),
        }
        if alg == "EXP":
            entry["baseline_avg_score"] = find_best_threshold_both(wm_base_avg, plain_base_avg)

        results[alg] = entry
        print(results[alg])

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("=" * 80)
    print(f"Saved to {OUTPUT_PATH}")
    print("=" * 80)


if __name__ == "__main__":
    main()