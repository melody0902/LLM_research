import json
import math
import os

import numpy as np
import torch
from tqdm import tqdm

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from watermark.auto_watermark import AutoWatermark
from utils.transformers_config import TransformersConfig
from evaluation.baseline_detectors import BaselineDetectorFactory
from watermark.synthid.detector import get_detector


# ============================================================
# 1. Global settings
# ============================================================

ALGORITHM = "SynthID"
DOMAIN = "ai"
BASE_ALGO_FOR_PLAIN = "KGW"

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
    f"synthid_subset_vs_baseline_{DOMAIN}_{MODEL_TAG}_avoid_top{TOP_K_TOKENS}.json"
)


# ============================================================
# 2. Transformers config
# ============================================================

def get_transformers_config():
    print(f"Using model: {MODEL_NAME}")
    print(f"Using model tag: {MODEL_TAG}")

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

def normalize_text(x):
    if x is None:
        return ""
    if not isinstance(x, str):
        x = str(x)
    return x.strip()


def load_texts(path, key):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    texts = []
    for i, x in enumerate(data):
        if not isinstance(x, dict):
            print(f"[load_texts] idx={i}, non-dict item skipped as empty")
            texts.append("")
            continue

        text = normalize_text(x.get(key, ""))
        texts.append(text)

    return texts


def load_token_subset(path, top_k=None):
    with open(path, "r", encoding="utf-8") as f:
        freq = json.load(f)

    if top_k is not None:
        freq = freq[:top_k]

    return {int(x["token_id"]) for x in freq}


def token_len(tokenizer, text):
    text = normalize_text(text)
    if not text:
        return 0

    ids = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=False,
    )["input_ids"][0]

    return int(ids.numel())


def is_too_short_for_synthid(text, tokenizer, wm):
    text = normalize_text(text)
    if not text:
        return True

    n_tokens = token_len(tokenizer, text)
    if n_tokens == 0:
        return True

    ngram_len = int(getattr(wm.config, "ngram_len", 1))
    return n_tokens < ngram_len


def score_to_float(score):
    """
    Some detectors return a Python float, numpy scalar, torch scalar,
    or a dict containing score-like fields.
    """
    if isinstance(score, dict):
        for key in ["score", "z_score", "prediction", "p_value"]:
            if key in score:
                return float(score[key])
        raise ValueError(f"Cannot find score field in dict: {score.keys()}")

    if isinstance(score, torch.Tensor):
        if score.numel() == 0:
            raise ValueError("Empty tensor score")
        return float(score.detach().cpu().reshape(-1)[0].item())

    arr = np.asarray(score)
    if arr.size == 0:
        raise ValueError("Empty score")
    return float(arr.reshape(-1)[0])


def safe_baseline_score(baseline_detector, text):
    text = normalize_text(text)
    if not text:
        return None

    try:
        return score_to_float(baseline_detector.detect(text))
    except Exception as e:
        print(f"[skip baseline] reason={repr(e)}, text={repr(text[:80])}")
        return None


def safe_subset_score(subset_detector, text):
    text = normalize_text(text)
    if not text:
        return None

    try:
        return score_to_float(subset_detector.detect(text))
    except Exception as e:
        print(f"[skip subset] reason={repr(e)}, text={repr(text[:80])}")
        return None


# ============================================================
# 4. Metric + threshold sweep
#    Supports gt / lt, because SynthID score direction may differ.
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

    scores = [float(x) for x in wm_scores + plain_scores if x is not None and math.isfinite(float(x))]
    if len(scores) == 0:
        return {
            "TPR": 0.0,
            "F1": 0.0,
            "precision": 0.0,
            "FPR": 0.0,
            "threshold": None,
            "direction": None,
            "note": "No finite scores available.",
        }

    labels = [1] * len(wm_scores) + [0] * len(plain_scores)
    all_scores = [float(x) for x in wm_scores + plain_scores]

    best = None

    for direction in ["gt", "lt"]:
        best_f1 = -1.0
        best_metrics = None

        for thr in sorted(set(scores)):
            if direction == "gt":
                preds = [1 if s > thr else 0 for s in all_scores]
            else:
                preds = [1 if s < thr else 0 for s in all_scores]

            TP = sum(p == 1 and y == 1 for p, y in zip(preds, labels))
            FP = sum(p == 1 and y == 0 for p, y in zip(preds, labels))
            FN = sum(p == 0 and y == 1 for p, y in zip(preds, labels))
            TN = sum(p == 0 and y == 0 for p, y in zip(preds, labels))

            TPR = TP / (TP + FN) if (TP + FN) > 0 else 0.0
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
            FPR = FP / (FP + TN) if (FP + TN) > 0 else 0.0
            F1 = (2 * precision * TPR / (precision + TPR)) if (precision + TPR) > 0 else 0.0

            if F1 > best_f1:
                best_f1 = F1
                best_metrics = {
                    "TPR": float(TPR),
                    "F1": float(F1),
                    "precision": float(precision),
                    "FPR": float(FPR),
                    "threshold": float(thr),
                    "direction": direction,
                }

        if best is None or best_metrics["F1"] > best["F1"]:
            best = best_metrics

    return best


# ============================================================
# 5. Subset-aware SynthID detector
# ============================================================

class SubsetAwareSynthIDDetector:
    def __init__(self, wm, token_subset, detector_name="mean"):
        self.wm = wm
        self.lp = wm.logits_processor
        self.cfg = wm.config
        self.tokenizer = self.cfg.generation_tokenizer
        self.S = set(int(x) for x in token_subset)
        self.detector = get_detector(detector_name, self.lp)

    def detect(self, text):
        text = normalize_text(text)
        if not text:
            return 0.0

        device = self.lp.device
        input_ids = self.tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=False,
        )["input_ids"].to(device)

        if input_ids.numel() == 0:
            return 0.0

        seq_len = int(input_ids.shape[1])
        ngram_len = int(getattr(self.cfg, "ngram_len", 1))

        # SynthID needs at least ngram_len tokens to form one scorable position.
        if seq_len < ngram_len:
            return 0.0

        g_values = self.lp.compute_g_values(input_ids)

        eos_mask = self.lp.compute_eos_token_mask(
            input_ids=input_ids,
            eos_token_id=self.tokenizer.eos_token_id,
        )[:, ngram_len - 1:]

        if self.cfg.watermark_mode == "non-distortionary":
            context_mask = self.lp.compute_context_repetition_mask(input_ids)
            mask_orig = eos_mask * context_mask
        else:
            mask_orig = eos_mask

        token_ids = input_ids[0, ngram_len - 1:].detach().cpu().numpy()
        if token_ids.size == 0:
            return 0.0

        mask_subset = np.array(
            [[1 if int(t) in self.S else 0 for t in token_ids]],
            dtype=np.float32,
        )

        mask_final = mask_orig.detach().cpu().numpy().astype(np.float32) * mask_subset

        # No token from the selected subset is scorable.
        if mask_final.size == 0 or float(mask_final.sum()) <= 0.0:
            return 0.0

        score = self.detector.detect(
            g_values.detach().cpu().numpy(),
            mask_final,
        )[0]

        score = float(score)
        if not math.isfinite(score):
            return 0.0

        return score


# ============================================================
# 6. Main
# ============================================================

def main():
    print("=" * 80)
    print("SynthID Baseline vs Subset Detection")
    print("=" * 80)

    cfg = get_transformers_config()

    wm = AutoWatermark.load("SynthID", "config/SynthID.json", cfg)
    baseline_detector = BaselineDetectorFactory(wm).build()

    wm_texts = load_texts(
        TEXT_JSON_TMPL.format(
            domain=DOMAIN,
            alg=ALGORITHM,
            model_tag=MODEL_TAG,
            top_k=TOP_K_TOKENS,
        ),
        "rewritten",
    )

    plain_texts = load_texts(
        TEXT_JSON_TMPL.format(
            domain=DOMAIN,
            alg=BASE_ALGO_FOR_PLAIN,
            model_tag=MODEL_TAG,
            top_k=TOP_K_TOKENS,
        ),
        "plain",
    )

    token_subset = load_token_subset(
        TOKEN_SET_TMPL.format(
            domain=DOMAIN,
            alg=ALGORITHM,
            model_tag=MODEL_TAG,
            top_k=TOP_K_TOKENS,
        ),
        TOP_K_TOKENS,
    )

    subset_detector = SubsetAwareSynthIDDetector(wm, token_subset)

    print("Baseline detector:", type(baseline_detector).__name__)
    print("Subset detector:", type(subset_detector).__name__)

    wm_scores_base, plain_scores_base = [], []
    wm_scores_subset, plain_scores_subset = [], []

    skipped_pairs = 0
    n_pairs = min(len(wm_texts), len(plain_texts))

    for idx, (w, p) in enumerate(tqdm(zip(wm_texts[:n_pairs], plain_texts[:n_pairs]), total=n_pairs)):
        w = normalize_text(w)
        p = normalize_text(p)

        if is_too_short_for_synthid(w, cfg.tokenizer, wm) or is_too_short_for_synthid(p, cfg.tokenizer, wm):
            skipped_pairs += 1
            print(f"[skip pair] idx={idx}, reason=empty_or_too_short")
            continue

        wm_b = safe_baseline_score(baseline_detector, w)
        pl_b = safe_baseline_score(baseline_detector, p)
        wm_s = safe_subset_score(subset_detector, w)
        pl_s = safe_subset_score(subset_detector, p)

        if wm_b is None or pl_b is None or wm_s is None or pl_s is None:
            skipped_pairs += 1
            print(f"[skip pair] idx={idx}, reason=detector_error")
            continue

        wm_scores_base.append(wm_b)
        plain_scores_base.append(pl_b)
        wm_scores_subset.append(wm_s)
        plain_scores_subset.append(pl_s)

    results = {
        "algorithm": ALGORITHM,
        "domain": DOMAIN,
        "model": MODEL_NAME,
        "model_tag": MODEL_TAG,
        "subset_size": len(token_subset),
        "num_total_pairs": n_pairs,
        "num_valid_pairs": len(wm_scores_base),
        "num_skipped_pairs": skipped_pairs,
        "baseline": find_best_threshold_both(
            wm_scores_base,
            plain_scores_base,
        ),
        "subset": find_best_threshold_both(
            wm_scores_subset,
            plain_scores_subset,
        ),
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(results)
    print(f"Saved to {OUTPUT_PATH}")
    print("=" * 80)


if __name__ == "__main__":
    main()
