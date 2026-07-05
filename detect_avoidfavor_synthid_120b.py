# ============================================================
# detect_avoidfavor_synthid_120b.py
#
# 120B detection for SynthID.
# Reads avoid/favor output from:
#   outputs/rewrite_avoid_favor_multi_120b/
# Reads token subset from:
#   outputs/wm_tokens_120b_0_200/
# ============================================================

import argparse
import json
import os

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from watermark.auto_watermark import AutoWatermark
from utils.transformers_config import TransformersConfig
from evaluation.baseline_detectors import BaselineDetectorFactory
from watermark.synthid.detector import get_detector


DEFAULT_MODEL_NAME = "openai/gpt-oss-120b"
DEFAULT_DOMAINS = ["ai", "bio", "med", "mis", "security"]

DEFAULT_SAMPLE_MODE = "sequential"
DEFAULT_SAMPLE_SEED = 30
DEFAULT_MAX_SAMPLES = 200

ALGORITHM = "SynthID"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOP_K_TOKENS = 200

REWRITE_AVOID_DIR = "outputs/rewrite_avoid_favor_multi_120b"
TOKEN_DIR = "outputs/wm_tokens_120b_0_200"
OUTPUT_DIR = "outputs/test/detect_avoidfavor_120b"


def sanitize_model_name(model_name: str) -> str:
    return model_name.replace("/", "__").replace(" ", "_")


def get_rewrite_avoid_path(domain, model_name, sample_mode, sample_seed, max_samples, top_k):
    safe = sanitize_model_name(model_name)
    return os.path.join(
        REWRITE_AVOID_DIR,
        f"rewrite_avoid_favor_{domain}_{ALGORITHM}_{safe}_{sample_mode}_seed{sample_seed}_n{max_samples}_top{top_k}.json",
    )


def get_token_path(domain, model_name, sample_mode, sample_seed, max_samples):
    safe = sanitize_model_name(model_name)
    return os.path.join(
        TOKEN_DIR,
        f"rewritten_{domain}_{ALGORITHM}_{safe}_{sample_mode}_seed{sample_seed}_n{max_samples}_wm_token_freq.json",
    )


def get_output_path(domain, model_name, sample_mode, sample_seed, max_samples, top_k, tag=None):
    safe = sanitize_model_name(model_name)
    suffix = f"_{tag}" if tag else ""
    return os.path.join(
        OUTPUT_DIR,
        f"detect_avoidfavor_synthid_120b_{domain}_{safe}_{sample_mode}_seed{sample_seed}_n{max_samples}_top{top_k}{suffix}.json",
    )


def get_transformers_config(
    model_name,
    load_in_4bit=False,
    load_in_8bit=False,
    torch_dtype="bfloat16",
    max_memory=None,
):
    print(f"Using detector model: {model_name}")

    if torch_dtype == "float16":
        dtype = torch.float16
    elif torch_dtype == "float32":
        dtype = torch.float32
    else:
        dtype = torch.bfloat16

    model_kwargs = dict(
        device_map="auto",
        dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    if max_memory:
        memory_map = {}
        for item in max_memory.split(","):
            k, v = item.split(":", 1)
            if k.strip().lower() == "cpu":
                memory_map["cpu"] = v.strip()
            else:
                memory_map[int(k.strip())] = v.strip()
        model_kwargs["max_memory"] = memory_map

    if "gpt-oss" in model_name.lower():
        if load_in_4bit:
            print("[warning] gpt-oss already uses MXFP4; ignoring --load_in_4bit")
            load_in_4bit = False
        if load_in_8bit:
            print("[warning] gpt-oss already uses MXFP4; ignoring --load_in_8bit")
            load_in_8bit = False

    if load_in_4bit and load_in_8bit:
        raise ValueError("Choose only one: load_in_4bit or load_in_8bit.")

    if load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )

    if load_in_8bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    if model.get_output_embeddings() is not None:
        real_vocab_size = model.get_output_embeddings().out_features
    else:
        real_vocab_size = model.config.vocab_size

    return TransformersConfig(
        model=model,
        tokenizer=tokenizer,
        vocab_size=real_vocab_size,
        device=DEVICE,
        max_new_tokens=200,
        do_sample=False,
    )


def load_texts(path, key):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    texts = []

    for item in data:
        text = item.get(key, "")
        if isinstance(text, str) and text.strip():
            texts.append(text.strip())

    return texts


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
        best_f1 = -1.0
        best_metrics = None

        for thr in sorted(set(scores)):
            if direction == "gt":
                preds = [1 if s > thr else 0 for s in scores]
            else:
                preds = [1 if s < thr else 0 for s in scores]

            tp = sum(p == 1 and y == 1 for p, y in zip(preds, labels))
            fp = sum(p == 1 and y == 0 for p, y in zip(preds, labels))
            fn = sum(p == 0 and y == 1 for p, y in zip(preds, labels))
            tn = sum(p == 0 and y == 0 for p, y in zip(preds, labels))

            tpr = tp / (tp + fn) if (tp + fn) else 0.0
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            fpr = fp / (fp + tn) if (fp + tn) else 0.0
            f1 = 2 * precision * tpr / (precision + tpr) if (precision + tpr) else 0.0

            if f1 > best_f1:
                best_f1 = f1
                best_metrics = {
                    "TPR": tpr,
                    "F1": f1,
                    "precision": precision,
                    "FPR": fpr,
                    "threshold": thr,
                    "direction": direction,
                }

        if best is None or best_metrics["F1"] > best["F1"]:
            best = best_metrics

    return best


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
            text,
            return_tensors="pt",
            add_special_tokens=False,
        )["input_ids"].to(device)

        g_values = self.lp.compute_g_values(input_ids)

        eos_mask = self.lp.compute_eos_token_mask(
            input_ids=input_ids,
            eos_token_id=self.tokenizer.eos_token_id,
        )[:, self.cfg.ngram_len - 1:]

        if self.cfg.watermark_mode == "non-distortionary":
            context_mask = self.lp.compute_context_repetition_mask(input_ids)
            mask_orig = eos_mask * context_mask
        else:
            mask_orig = eos_mask

        token_ids = input_ids[0, self.cfg.ngram_len - 1:].detach().cpu().numpy()
        mask_subset = np.array([[1 if int(t) in self.S else 0 for t in token_ids]])

        mask_final = mask_orig.detach().cpu().numpy() * mask_subset

        score = self.detector.detect(
            g_values.detach().cpu().numpy(),
            mask_final,
        )[0]

        return float(score)


def baseline_detect_score(baseline_detector, text):
    score = baseline_detector.detect(text)
    return float(score)


def run_domain(args, cfg, domain):
    print("=" * 80)
    print(f"SynthID 120B detection: domain={domain}")
    print("=" * 80)

    rewrite_path = get_rewrite_avoid_path(
        domain=domain,
        model_name=args.model_name,
        sample_mode=args.sample_mode,
        sample_seed=args.sample_seed,
        max_samples=args.max_samples,
        top_k=args.top_k_tokens,
    )

    token_path = get_token_path(
        domain=domain,
        model_name=args.model_name,
        sample_mode=args.sample_mode,
        sample_seed=args.sample_seed,
        max_samples=args.max_samples,
    )

    if not os.path.exists(rewrite_path):
        print(f"[SKIP] missing rewrite file: {rewrite_path}")
        return

    if not os.path.exists(token_path):
        print(f"[SKIP] missing token file: {token_path}")
        return

    wm = AutoWatermark.load(ALGORITHM, "config/SynthID.json", cfg)
    baseline_detector = BaselineDetectorFactory(wm).build()

    wm_texts = load_texts(rewrite_path, "rewrite_watermarked_avoid_set")
    plain_texts = load_texts(rewrite_path, "rewrite_unwatermarked_favor_set")

    n = min(len(wm_texts), len(plain_texts))
    wm_texts = wm_texts[:n]
    plain_texts = plain_texts[:n]

    if args.test_limit is not None:
        wm_texts = wm_texts[:args.test_limit]
        plain_texts = plain_texts[:args.test_limit]
        n = min(len(wm_texts), len(plain_texts))

    if n == 0:
        print(f"[SKIP] no valid texts for {domain}")
        return

    token_subset = load_token_subset(token_path, args.top_k_tokens)
    subset_detector = SubsetAwareSynthIDDetector(
        wm,
        token_subset,
        detector_name=args.detector_name,
    )

    print("Baseline detector:", type(baseline_detector).__name__)
    print("Subset detector:", type(subset_detector).__name__)
    print("Rewrite model:", args.model_name)
    print("Num pairs:", n)
    print("Subset size:", len(token_subset))

    wm_scores_base, plain_scores_base = [], []
    wm_scores_subset, plain_scores_subset = [], []

    for w, p in tqdm(zip(wm_texts, plain_texts), total=n):
        wm_scores_base.append(baseline_detect_score(baseline_detector, w))
        plain_scores_base.append(baseline_detect_score(baseline_detector, p))

        wm_scores_subset.append(subset_detector.detect(w))
        plain_scores_subset.append(subset_detector.detect(p))

    results = {
        "algorithm": ALGORITHM,
        "domain": domain,
        "rewrite_model": args.model_name,
        "sample_mode": args.sample_mode,
        "sample_seed": args.sample_seed,
        "max_samples": args.max_samples,
        "top_k_tokens": args.top_k_tokens,
        "detector_name": args.detector_name,
        "num_pairs": n,
        "positive_text_key": "rewrite_watermarked_avoid_set",
        "negative_text_key": "rewrite_unwatermarked_favor_set",
        "subset_size": len(token_subset),
        "baseline_wm_avoid_vs_unwm_favor": find_best_threshold_both(
            wm_scores_base,
            plain_scores_base,
        ),
        "subset_wm_avoid_vs_unwm_favor": find_best_threshold_both(
            wm_scores_subset,
            plain_scores_subset,
        ),
        "raw_scores": {
            "baseline_positive": wm_scores_base,
            "baseline_negative": plain_scores_base,
            "subset_positive": wm_scores_subset,
            "subset_negative": plain_scores_subset,
        } if args.save_raw_scores else None,
    }

    output_path = get_output_path(
        domain=domain,
        model_name=args.model_name,
        sample_mode=args.sample_mode,
        sample_seed=args.sample_seed,
        max_samples=args.max_samples,
        top_k=args.top_k_tokens,
        tag=args.output_tag,
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(results)
    print(f"Saved to {output_path}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--domains", type=str, nargs="+", default=DEFAULT_DOMAINS)
    parser.add_argument("--sample_mode", type=str, default=DEFAULT_SAMPLE_MODE, choices=["sequential", "random"])
    parser.add_argument("--sample_seed", type=int, default=DEFAULT_SAMPLE_SEED)
    parser.add_argument("--max_samples", type=int, default=DEFAULT_MAX_SAMPLES)
    parser.add_argument("--top_k_tokens", type=int, default=TOP_K_TOKENS)
    parser.add_argument("--detector_name", type=str, default="mean")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
    )
    parser.add_argument("--max_memory", type=str, default=None)
    parser.add_argument("--test_limit", type=int, default=None)
    parser.add_argument("--output_tag", type=str, default=None)
    parser.add_argument("--save_raw_scores", action="store_true")

    args = parser.parse_args()

    cfg = get_transformers_config(
        model_name=args.model_name,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        torch_dtype=args.torch_dtype,
        max_memory=args.max_memory,
    )

    for domain in args.domains:
        run_domain(args, cfg, domain)


if __name__ == "__main__":
    main()
