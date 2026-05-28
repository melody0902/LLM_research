import json
import math
import torch
from tqdm import tqdm
import os

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from watermark.auto_watermark import AutoWatermark
from utils.transformers_config import TransformersConfig


# ============================================================
# 1. Global settings
# ============================================================

ALGORITHMS = ["KGW", "SWEET", "Unigram", "EXP"]
BASE_ALGO_FOR_PLAIN = "KGW"
DOMAIN = "ai"  # e.g. "bi" / "ai"

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
    f"subset_vs_baseline_{DOMAIN}_{MODEL_TAG}_avoid_top{TOP_K_TOKENS}.json"
)


# ============================================================
# 2. Transformers config
# ============================================================

def get_transformers_config():
    print(f"Using model: {MODEL_NAME}")

    nf4 = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
        quantization_config=nf4,
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
# 3.1 EXP evidence computation (baseline avg_score)
# ============================================================

def exp_pvalue_and_avgscore(wm, text, eps=1e-12):
    """
    EXP stats computed externally:
    - total_score = sum_i -log(1-r_i)
    - avg_score   = total_score / num_scored
    - p_value     = P(Gamma(k=num_scored, θ=1) >= total_score)
    """
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
        total_score += -math.log(1.0 - r)  # == log(1/(1-r))
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
    labels = [1]*len(wm_scores) + [0]*len(plain_scores)

    best = None

    for direction in ["gt", "lt"]:
        best_f1 = -1
        best_metrics = None

        for thr in sorted(set(scores)):
            if direction == "gt":
                preds = [1 if s > thr else 0 for s in scores]
            else:  # "lt"
                preds = [1 if s < thr else 0 for s in scores]

            TP = sum(p == 1 and y == 1 for p, y in zip(preds, labels))
            FP = sum(p == 1 and y == 0 for p, y in zip(preds, labels))
            FN = sum(p == 0 and y == 1 for p, y in zip(preds, labels))
            TN = sum(p == 0 and y == 0 for p, y in zip(preds, labels))

            TPR = TP / (TP + FN) if (TP + FN) else 0.0
            precision = TP / (TP + FP) if (TP + FP) else 0.0
            FPR = FP / (FP + TN) if (FP + TN) else 0.0
            F1 = 2*precision*TPR/(precision+TPR) if (precision+TPR) else 0.0

            if F1 > best_f1:
                best_f1 = F1
                best_metrics = {
                    "TPR": TPR,
                    "F1": F1,
                    "precision": precision,
                    "FPR": FPR,
                    "threshold": thr,
                    "direction": direction,
                }

        if best is None or best_metrics["F1"] > best["F1"]:
            best = best_metrics

    return best


# # ============================================================
# # 5. Subset-aware detectors
# # ============================================================

# class SubsetAwareKGWDetector:
#     """
#     subset score = (# positions i where token in S AND token in greenlist(prefix_i)) / (# positions i where token in S)
#     """
#     def __init__(self, wm, S):
#         self.wm = wm
#         self.utils = wm.utils
#         self.tokenizer = wm.config.generation_tokenizer
#         self.prefix_len = wm.config.prefix_length
#         self.S = set(S)

#     def detect(self, text):
#         ids = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
#         ids = ids.to(self.wm.config.device)

#         obs, total = 0, 0
#         for i in range(self.prefix_len, len(ids)):
#             t = ids[i].item()
#             if t not in self.S:
#                 continue
#             total += 1
#             if t in self.utils.get_greenlist_ids(ids[:i]):
#                 obs += 1
#         return obs / total if total > 0 else 0.0


# class SubsetAwareSWEETDetector(SubsetAwareKGWDetector):
#     """
#     SWEET subset score = same as KGW but only on high-entropy positions (entropy[i] > threshold)
#     """
#     def __init__(self, wm, S):
#         super().__init__(wm, S)
#         self.model = wm.config.generation_model
#         self.entropy_threshold = wm.config.entropy_threshold

#     def detect(self, text):
#         ids = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
#         ids = ids.to(self.wm.config.device)

#         entropy = self.utils.calculate_entropy(self.model, ids)

#         obs, total = 0, 0
#         for i in range(self.prefix_len, len(ids)):
#             if entropy[i] <= self.entropy_threshold:
#                 continue
#             t = ids[i].item()
#             if t not in self.S:
#                 continue
#             total += 1
#             if t in self.utils.get_greenlist_ids(ids[:i]):
#                 obs += 1
#         return obs / total if total > 0 else 0.0


# class SubsetAwareUnigramDetector:
#     """
#     Unigram subset score = (# tokens t in S with mask[t]=1) / (# tokens t in S)
#     (does not use prefix_len in this implementation)
#     """
#     def __init__(self, wm, S):
#         self.mask = wm.utils.mask
#         self.tokenizer = wm.config.generation_tokenizer
#         self.S = set(S)

#     def detect(self, text):
#         ids = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0].tolist()

#         obs, total = 0, 0
#         for t in ids:
#             if t not in self.S:
#                 continue
#             total += 1
#             if self.mask[t]:
#                 obs += 1
#         return obs / total if total > 0 else 0.0


# class SubsetAwareEXPDetector:
#     """
#     EXP subset score = average_i[-log(1-r_i)] over positions where token in S (and i >= prefix_len)
#     """
#     def __init__(self, wm, S):
#         self.wm = wm
#         self.utils = wm.utils
#         self.tokenizer = wm.config.generation_tokenizer
#         self.prefix_len = wm.config.prefix_length
#         self.vocab_size = wm.config.vocab_size
#         self.S = set(S)

#     def detect(self, text):
#         ids = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]

#         score, total = 0.0, 0
#         for i in range(self.prefix_len, len(ids)):
#             t = ids[i].item()
#             if t not in self.S:
#                 continue
#             self.utils.seed_rng(ids[:i])
#             r = torch.rand(self.vocab_size, generator=self.utils.rng)[t].item()
#             score += -math.log(1.0 - r)  # == log(1/(1-r))
#             total += 1
#         return score / total if total > 0 else 0.0

# ============================================================
# 5. Subset-aware detectors (BASELINE Z-SCORE STYLE)
#     - score uses z-score like baseline
#     - BUT n = #subset tokens (scorable positions that are in S)
# ============================================================

def _get_gamma_from_wm(wm):
    """
    Try best-effort to fetch the greenlist/mask expected rate (gamma) from watermark config.
    Adjust the attribute names here if your config uses different keys.
    """
    for name in ["gamma", "greenlist_ratio", "greenlist_fraction", "watermark_gamma"]:
        if hasattr(wm.config, name):
            g = getattr(wm.config, name)
            if g is not None:
                return float(g)
    raise AttributeError(
        "Cannot find gamma in wm.config. Tried: gamma / greenlist_ratio / greenlist_fraction / watermark_gamma. "
        "Please check your watermark config fields."
    )


def _zscore(obs, n, p, eps=1e-12):
    # Binomial z-score
    if n <= 0:
        return 0.0
    var = n * p * (1.0 - p)
    return float((obs - n * p) / math.sqrt(var + eps))


class SubsetAwareKGWDetector:
    """
    Baseline-style z-score, but n = # positions (i>=prefix_len) where token in S.
    obs = # of those positions that are also in greenlist(prefix_i).
    """

    def __init__(self, wm, S):
        self.wm = wm
        self.utils = wm.utils
        self.tokenizer = wm.config.generation_tokenizer
        self.prefix_len = wm.config.prefix_length
        self.S = set(S)
        self.gamma = _get_gamma_from_wm(wm)

    def detect(self, text):
        ids = self.tokenizer(
            text, return_tensors="pt", add_special_tokens=False
        )["input_ids"][0].to(self.wm.config.device)

        obs = 0
        n = 0

        for i in range(self.prefix_len, len(ids)):
            t = ids[i].item()
            if t not in self.S:
                continue

            n += 1
            if t in self.utils.get_greenlist_ids(ids[:i]):
                obs += 1

        return _zscore(obs, n, self.gamma)


class SubsetAwareSWEETDetector(SubsetAwareKGWDetector):
    """
    Same as baseline SWEET: only score high-entropy positions,
    but n still counts only subset tokens among those positions.
    """

    def __init__(self, wm, S):
        super().__init__(wm, S)
        self.model = wm.config.generation_model
        self.entropy_threshold = wm.config.entropy_threshold

    def detect(self, text):
        ids = self.tokenizer(
            text, return_tensors="pt", add_special_tokens=False
        )["input_ids"][0].to(self.wm.config.device)

        entropy = self.utils.calculate_entropy(self.model, ids)

        obs = 0
        n = 0

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
    """
    Baseline-style z-score for Unigram:
    - obs = # tokens in S with mask[t]==1
    - n   = # tokens in S
    p(gamma) uses wm.config.* (same gamma meaning: expected mask-on rate)
    """

    def __init__(self, wm, S):
        self.wm = wm
        self.mask = wm.utils.mask
        self.tokenizer = wm.config.generation_tokenizer
        self.S = set(S)
        self.gamma = _get_gamma_from_wm(wm)

    def detect(self, text):
        ids = self.tokenizer(
            text, return_tensors="pt", add_special_tokens=False
        )["input_ids"][0].tolist()

        obs = 0
        n = 0

        for t in ids:
            if t not in self.S:
                continue
            n += 1
            if self.mask[t]:
                obs += 1

        return _zscore(obs, n, self.gamma)


class SubsetAwareEXPDetector:
    """
    EXP 不太是二項(success/failure)→ z-score 的形式通常不適用。
    這裡給一個「baseline風格」的做法：只在 subset tokens 上計算 EXP 的 avg evidence
    （也就是用 subset token 當 total token）。
    如果你 baseline 的 EXP score 不是 avg evidence，而是 p-value/其他統計量，
    再把這裡換成一致的那個即可。
    """

    def __init__(self, wm, S):
        self.wm = wm
        self.utils = wm.utils
        self.tokenizer = wm.config.generation_tokenizer
        self.prefix_len = wm.config.prefix_length
        self.vocab_size = wm.config.vocab_size
        self.S = set(S)

    def detect(self, text, eps=1e-12):
        ids = self.tokenizer(
            text, return_tensors="pt", add_special_tokens=False
        )["input_ids"][0]

        score_sum = 0.0
        n = 0

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


SUBSET_DETECTORS = {
    "KGW": SubsetAwareKGWDetector,
    "SWEET": SubsetAwareSWEETDetector,
    "Unigram": SubsetAwareUnigramDetector,
    "EXP": SubsetAwareEXPDetector,
}

# ============================================================
# 6. Main
# ============================================================

def main():
    print("=" * 80)
    print("Subset vs Baseline Detection (Non-SynthID) - ALL ALGS")
    print("=" * 80)

    cfg = get_transformers_config()

    plain_texts = load_texts(
        TEXT_JSON_TMPL.format(
            domain=DOMAIN,
            alg=BASE_ALGO_FOR_PLAIN,
            model_tag=MODEL_TAG,
            top_k=TOP_K_TOKENS,
        ),
        "plain"
    )

    results = {}

    for alg in ALGORITHMS:
        print("-" * 80)
        print(f"Algorithm: {alg}")

        wm_texts = load_texts(
            TEXT_JSON_TMPL.format(
                domain=DOMAIN,
                alg=alg,
                model_tag=MODEL_TAG,
                top_k=TOP_K_TOKENS,
            ),
            "rewritten"
        )

        token_subset = load_token_subset(
            TOKEN_SET_TMPL.format(
                domain=DOMAIN,
                alg=alg,
                model_tag=MODEL_TAG,
                top_k=TOP_K_TOKENS,
            ),
            TOP_K_TOKENS
        )

        wm = AutoWatermark.load(alg, f"config/{alg}.json", cfg)

        baseline_detector = wm.detect_watermark
        subset_detector = SUBSET_DETECTORS[alg](wm, token_subset)

        wm_base, plain_base = [], []
        wm_subset, plain_subset = [], []

        # EXP extra baseline: avg evidence
        wm_base_avg, plain_base_avg = [], []

        for w, p in tqdm(zip(wm_texts, plain_texts), total=len(plain_texts)):
            # ===== baseline score from library =====
            wm_base.append(baseline_detector(w, return_dict=True)["score"])
            plain_base.append(baseline_detector(p, return_dict=True)["score"])

            # ===== subset score =====
            wm_subset.append(subset_detector.detect(w))
            plain_subset.append(subset_detector.detect(p))

            # ===== optional: EXP avg_score baseline =====
            if alg == "EXP":
                wm_stats = exp_pvalue_and_avgscore(wm, w)
                pl_stats = exp_pvalue_and_avgscore(wm, p)
                wm_base_avg.append(wm_stats["avg_score"])
                plain_base_avg.append(pl_stats["avg_score"])

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