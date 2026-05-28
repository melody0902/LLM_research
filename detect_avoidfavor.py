import json
import math
import os
import torch
from tqdm import tqdm

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from watermark.auto_watermark import AutoWatermark
from utils.transformers_config import TransformersConfig

ALGORITHMS = ["KGW", "SWEET", "Unigram", "EXP"]
# ALGORITHMS = ["KGW"]
BASE_ALGO_FOR_PLAIN = "KGW"
DOMAIN = "ai"
# DOMAIN = ["ai", "bio", "med", "mis"]
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TOP_K_TOKENS = 200

REWRITE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
REWRITE_MODEL_TAG = REWRITE_MODEL.replace("/", "__")

PLAIN_JSON = f"outputs/0123_200green/rewritten_{DOMAIN}_{BASE_ALGO_FOR_PLAIN}_{REWRITE_MODEL_TAG}_wm_tokens.json"
REWRITTEN_AVOID_JSON_TMPL = (
    "outputs/rewrite_avoid_favor_multi_0517/"
    "rewritten_{domain}_{alg}_{model_tag}_avoid_top200.json"
)

TOKEN_SET_TMPL = "outputs/0305_200test/rewritten_{domain}_{alg}_wm_token_freq.json"

OUTPUT_PATH = (
    f"outputs/test/detect_avoid0517/"
    f"subset_vs_baseline_{DOMAIN}_{REWRITE_MODEL_TAG}_top{TOP_K_TOKENS}.json"
)


def get_transformers_config():
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


def load_texts(path, key):
    with open(path, "r", encoding="utf-8") as f:
        return [x[key] for x in json.load(f)]


def load_token_subset(path, top_k=None):
    with open(path, "r", encoding="utf-8") as f:
        freq = json.load(f)
    if top_k is not None:
        freq = freq[:top_k]
    return {x["token_id"] for x in freq}


def find_best_threshold_both(wm_scores, plain_scores):
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

        if best is None or best_metrics["F1"] > best["F1"]:
            best = best_metrics

    return best


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
    var = n * p * (1.0 - p)
    return float((obs - n * p) / math.sqrt(var + eps))


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
        self.wm = wm
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


SUBSET_DETECTORS = {
    "KGW": SubsetAwareKGWDetector,
    "SWEET": SubsetAwareSWEETDetector,
    "Unigram": SubsetAwareUnigramDetector,
    "EXP": SubsetAwareEXPDetector,
}


def main():
    cfg = get_transformers_config()

    results = {}

    for alg in ALGORITHMS:
        print(f"Running {alg}")

        rewrite_path = REWRITTEN_AVOID_JSON_TMPL.format(
            domain=DOMAIN,
            alg=alg,
            model_tag=REWRITE_MODEL_TAG,
        )

        wm_texts = load_texts(
            rewrite_path,
            "rewrite_watermarked_avoid_set"
        )

        plain_texts = load_texts(
            rewrite_path,
            "rewrite_unwatermarked_favor_set"
        )

        token_subset = load_token_subset(
            TOKEN_SET_TMPL.format(domain=DOMAIN, alg=alg),
            TOP_K_TOKENS
        )

        wm = AutoWatermark.load(alg, f"config/{alg}.json", cfg)
        baseline_detector = wm.detect_watermark
        subset_detector = SUBSET_DETECTORS[alg](wm, token_subset)

        wm_base, plain_base = [], []
        wm_subset, plain_subset = [], []

        for w, p in tqdm(zip(wm_texts, plain_texts), total=len(plain_texts)):
            wm_base.append(baseline_detector(w, return_dict=True)["score"])
            plain_base.append(baseline_detector(p, return_dict=True)["score"])

            wm_subset.append(subset_detector.detect(w))
            plain_subset.append(subset_detector.detect(p))

        results[alg] = {
            "subset_size": len(token_subset),

            "baseline_wm_avoid_vs_unwm_favor": find_best_threshold_both(
                wm_base,
                plain_base
            ),

            "subset_wm_avoid_vs_unwm_favor": find_best_threshold_both(
                wm_subset,
                plain_subset
            ),
        }

        print(results[alg])

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()