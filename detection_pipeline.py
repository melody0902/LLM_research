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
        data = json.load(f)

    texts = []
    for i, x in enumerate(data):
        text = normalize_text(x.get(key, ""))
        texts.append(text)

    return texts


def load_token_subset(path, top_k=None):
    with open(path, "r", encoding="utf-8") as f:
        freq = json.load(f)
    if top_k is not None:
        freq = freq[:top_k]
    return {x["token_id"] for x in freq}

def normalize_text(x):
    if x is None:
        return ""
    if not isinstance(x, str):
        x = str(x)
    return x.strip()


def token_len(tokenizer, text):
    ids = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=False
    )["input_ids"][0]
    return int(ids.numel())


def is_too_short_for_alg(text, tokenizer, wm, alg):
    text = normalize_text(text)

    if not text:
        return True

    n_tokens = token_len(tokenizer, text)

    if n_tokens == 0:
        return True

    # KGW / SWEET / EXP need positions after prefix_length.
    if alg in {"KGW", "SWEET", "EXP"}:
        prefix_len = getattr(wm.config, "prefix_length", 0)
        return n_tokens <= prefix_len

    # Unigram can technically score any non-empty token sequence.
    if alg == "Unigram":
        return False

    # SynthID-like algorithms usually need at least ngram_len tokens.
    if hasattr(wm.config, "ngram_len"):
        return n_tokens < wm.config.ngram_len

    return False


def safe_baseline_score(wm, alg, text):
    text = normalize_text(text)

    if is_too_short_for_alg(text, wm.config.generation_tokenizer, wm, alg):
        return None

    try:
        result = wm.detect_watermark(text, return_dict=True)
        return float(result["score"])
    except Exception as e:
        print(f"[skip baseline] alg={alg}, reason={repr(e)}, text={repr(text[:80])}")
        return None


def safe_subset_score(detector, text):
    text = normalize_text(text)

    if not text:
        return None

    try:
        return float(detector.detect(text))
    except Exception as e:
        print(f"[skip subset] reason={repr(e)}, text={repr(text[:80])}")
        return None
        
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
    if len(wm_scores) == 0 or len(plain_scores) == 0:
        return {
            "TPR": 0.0,
            "F1": 0.0,
            "precision": 0.0,
            "FPR": 0.0,
            "threshold": None,
            "direction": None,
            "note": "No valid scores available after skipping empty/too-short/error cases.",
        }

    scores = wm_scores + plain_scores
    labels = [1] * len(wm_scores) + [0] * len(plain_scores)

    best = None

    for direction in ["gt", "lt"]:
        best_f1 = -1
        best_metrics = None

        for thr in sorted(set(scores)):
            if direction == "gt":
                preds = [1 if s > thr else 0 for s in scores]
            else:
                preds = [1 if s < thr else 0 for s in scores]

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
                    "TPR": TPR,
                    "F1": F1,
                    "precision": precision,
                    "FPR": FPR,
                    "threshold": thr,
                    "direction": direction,
                }

        if best_metrics is not None and (best is None or best_metrics["F1"] > best["F1"]):
            best = best_metrics

    return best

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


# # ============================================================
# # 5. Subset-aware detectors
# # ============================================================

class SubsetAwareKGWDetector:
    """
    Baseline-style z-score, but n = # positions (i>=prefix_len) where token in S.
    obs = # of those positions that are also in greenlist(prefix_i).
    KGW does not use entropy.
    """

    def __init__(self, wm, S):
        self.wm = wm
        self.utils = wm.utils
        self.tokenizer = wm.config.generation_tokenizer
        self.prefix_len = wm.config.prefix_length
        self.S = set(S)
        self.gamma = _get_gamma_from_wm(wm)

    def detect(self, text):
        text = normalize_text(text)
        if not text:
            return 0.0

        ids = self.tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=False
        )["input_ids"][0].to(self.wm.config.device)

        if ids.numel() == 0 or ids.numel() <= self.prefix_len:
            return 0.0

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
        text = normalize_text(text)
        if not text:
            return 0.0

        ids = self.tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=False
        )["input_ids"][0].to(self.wm.config.device)

        if ids.numel() == 0 or ids.numel() <= self.prefix_len:
            return 0.0

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
    """

    def __init__(self, wm, S):
        self.wm = wm
        self.mask = wm.utils.mask
        self.tokenizer = wm.config.generation_tokenizer
        self.S = set(S)
        self.gamma = _get_gamma_from_wm(wm)

    def detect(self, text):
        text = normalize_text(text)
        if not text:
            return 0.0

        ids = self.tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=False
        )["input_ids"][0].tolist()

        if len(ids) == 0:
            return 0.0

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
    EXP subset score = average evidence over subset tokens at scorable positions.
    """

    def __init__(self, wm, S):
        self.wm = wm
        self.utils = wm.utils
        self.tokenizer = wm.config.generation_tokenizer
        self.prefix_len = wm.config.prefix_length
        self.vocab_size = wm.config.vocab_size
        self.S = set(S)

    def detect(self, text, eps=1e-12):
        text = normalize_text(text)
        if not text:
            return 0.0

        ids = self.tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=False
        )["input_ids"][0]

        if ids.numel() == 0 or ids.numel() <= self.prefix_len:
            return 0.0

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

        skipped_pairs = 0

        n_pairs = min(len(wm_texts), len(plain_texts))
        for idx, (w, p) in enumerate(tqdm(zip(wm_texts[:n_pairs], plain_texts[:n_pairs]), total=n_pairs)):
            w = normalize_text(w)
            p = normalize_text(p)
        
            if is_too_short_for_alg(w, cfg.tokenizer, wm, alg) or is_too_short_for_alg(p, cfg.tokenizer, wm, alg):
                skipped_pairs += 1
                print(f"[skip pair] alg={alg}, idx={idx}, reason=empty_or_too_short")
                continue
        
            wm_b = safe_baseline_score(wm, alg, w)
            pl_b = safe_baseline_score(wm, alg, p)
            wm_s = safe_subset_score(subset_detector, w)
            pl_s = safe_subset_score(subset_detector, p)
        
            if wm_b is None or pl_b is None or wm_s is None or pl_s is None:
                skipped_pairs += 1
                continue
        
            wm_base.append(wm_b)
            plain_base.append(pl_b)
            wm_subset.append(wm_s)
            plain_subset.append(pl_s)
        
            if alg == "EXP":
                wm_stats = exp_pvalue_and_avgscore(wm, w)
                pl_stats = exp_pvalue_and_avgscore(wm, p)
                wm_base_avg.append(wm_stats["avg_score"])
                plain_base_avg.append(pl_stats["avg_score"])

        entry = {
            "subset_size": len(token_subset),
            "num_total_pairs": n_pairs,
            "num_valid_pairs": len(wm_base),
            "num_skipped_pairs": skipped_pairs,
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
