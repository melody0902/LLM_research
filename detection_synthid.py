import json
import torch
import numpy as np
from tqdm import tqdm
import os

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from watermark.auto_watermark import AutoWatermark
from utils.transformers_config import TransformersConfig
from evaluation.baseline_detectors import BaselineDetectorFactory
from watermark.synthid.detector import get_detector


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
# 4. Metric + threshold sweep
# ============================================================

def find_best_threshold(wm_scores, plain_scores):
    scores = wm_scores + plain_scores
    labels = [1] * len(wm_scores) + [0] * len(plain_scores)

    best_f1 = -1.0
    best_thr = None
    best_metrics = None

    for thr in sorted(set(scores)):
        preds = [1 if s > thr else 0 for s in scores]

        TP = sum(p == 1 and y == 1 for p, y in zip(preds, labels))
        FP = sum(p == 1 and y == 0 for p, y in zip(preds, labels))
        FN = sum(p == 0 and y == 1 for p, y in zip(preds, labels))
        TN = sum(p == 0 and y == 0 for p, y in zip(preds, labels))

        TPR = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        FPR = FP / (FP + TN) if (FP + TN) > 0 else 0.0
        F1 = 2 * precision * TPR / (precision + TPR) if (precision + TPR) > 0 else 0.0

        if F1 > best_f1:
            best_f1 = F1
            best_thr = thr
            best_metrics = {
                "TPR": TPR,
                "F1": F1,
                "precision": precision,
                "FPR": FPR,
            }

    best_metrics["threshold"] = best_thr
    return best_metrics


# ============================================================
# 5. Subset-aware SynthID detector
# ============================================================

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

        input_ids = self.tokenizer(
            text, return_tensors="pt", add_special_tokens=False
        )["input_ids"].to(device)

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

        score = self.detector.detect(
            g_values.cpu().numpy(),
            mask_final
        )[0]

        return float(score)


# ============================================================
# 6. Main
# ============================================================

def main():
    print("=" * 80)
    print("SynthID Baseline vs Subset Detection")
    print("=" * 80)

    cfg = get_transformers_config()
    wm = AutoWatermark.load(ALGORITHM, f"config/{ALGORITHM}.json", cfg)

    baseline_detector = BaselineDetectorFactory(wm).build()

    plain_texts = load_texts(
        TEXT_JSON_TMPL.format(
            domain=DOMAIN,
            alg=BASE_ALGO_FOR_PLAIN,
            model_tag=MODEL_TAG,
            top_k=TOP_K_TOKENS,
        ),
        "plain"
    )

    wm_texts = load_texts(
        TEXT_JSON_TMPL.format(
            domain=DOMAIN,
            alg=ALGORITHM,
            model_tag=MODEL_TAG,
            top_k=TOP_K_TOKENS,
        ),
        "rewritten"
    )

    token_subset = load_token_subset(
        TOKEN_SET_TMPL.format(
            domain=DOMAIN,
            alg=ALGORITHM,
            model_tag=MODEL_TAG,
            top_k=TOP_K_TOKENS,
        ),
        TOP_K_TOKENS
    )

    subset_detector = SubsetAwareSynthIDDetector(wm, token_subset)

    print("Baseline detector:", type(baseline_detector).__name__)
    print("Subset detector:", type(subset_detector).__name__)

    wm_scores_base, plain_scores_base = [], []
    wm_scores_subset, plain_scores_subset = [], []

    for w, p in tqdm(zip(wm_texts, plain_texts), total=len(plain_texts)):
        wm_scores_base.append(baseline_detector.detect(w))
        plain_scores_base.append(baseline_detector.detect(p))

        wm_scores_subset.append(subset_detector.detect(w))
        plain_scores_subset.append(subset_detector.detect(p))

    results = {
        "algorithm": ALGORITHM,
        "domain": DOMAIN,
        "model_name": MODEL_NAME,
        "model_tag": MODEL_TAG,
        "subset_size": len(token_subset),
        "baseline": find_best_threshold(wm_scores_base, plain_scores_base),
        "subset": find_best_threshold(wm_scores_subset, plain_scores_subset),
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(results)
    print(f"Saved to {OUTPUT_PATH}")
    print("=" * 80)


if __name__ == "__main__":
    main()