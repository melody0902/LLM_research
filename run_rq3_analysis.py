import os
import json
import math
from itertools import combinations

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from scipy.stats import friedmanchisquare, wilcoxon, binomtest
from sklearn.metrics import f1_score, confusion_matrix

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
# 0. User settings
# ============================================================

ALGORITHMS = ["KGW", "SWEET", "Unigram", "EXP", "SynthID"]

# 改成你的所有 domains
DOMAINS = ["ai", "bio", "med", "mis", "security"]

SETTING_NAME = "nostop"          # e.g. nostop / full
BASE_ALGO_FOR_PLAIN = "KGW"
TOP_K_TOKENS = 1000
N_BOOT = 2000
BOOT_SEED = 42
ALPHA = 0.05

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TEXT_JSON_TMPL = "outputs/0123_200green/rewritten_{domain}_{alg}_wm_tokens.json"
TOKEN_SET_TMPL = "outputs/0305_200test/rewritten_{domain}_{alg}_wm_token_freq.json"

OUT_DIR = f"outputs/0305_200test/rq3_analysis/all_domains_top{TOP_K_TOKENS}_{SETTING_NAME}"
SAMPLE_CSV = os.path.join(OUT_DIR, "rq3_sample_level.csv")
DOMAIN_CSV = os.path.join(OUT_DIR, "rq3_domain_level.csv")
SUMMARY_JSON = os.path.join(OUT_DIR, "rq3_summary.json")


# ============================================================
# 1. Transformers config
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
# 2. Helpers
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


def _get_gamma_from_wm(wm):
    for name in ["gamma", "greenlist_ratio", "greenlist_fraction", "watermark_gamma"]:
        if hasattr(wm.config, name):
            g = getattr(wm.config, name)
            if g is not None:
                return float(g)
    raise AttributeError(
        "Cannot find gamma in wm.config. Tried gamma / greenlist_ratio / "
        "greenlist_fraction / watermark_gamma."
    )


def _zscore(obs, n, p, eps=1e-12):
    if n <= 0:
        return 0.0
    var = n * p * (1.0 - p)
    return float((obs - n * p) / math.sqrt(var + eps))


def safe_load_domain_inputs(domain, alg, base_algo_for_plain):
    wm_path = TEXT_JSON_TMPL.format(domain=domain, alg=alg)
    plain_path = TEXT_JSON_TMPL.format(domain=domain, alg=base_algo_for_plain)
    token_path = TOKEN_SET_TMPL.format(domain=domain, alg=alg)

    if not os.path.exists(wm_path):
        raise FileNotFoundError(f"Missing watermark text file: {wm_path}")
    if not os.path.exists(plain_path):
        raise FileNotFoundError(f"Missing plain text file: {plain_path}")
    if not os.path.exists(token_path):
        raise FileNotFoundError(f"Missing token subset file: {token_path}")

    wm_texts = load_texts(wm_path, "rewritten")
    plain_texts = load_texts(plain_path, "plain")
    token_subset = load_token_subset(token_path, TOP_K_TOKENS)

    if len(wm_texts) != len(plain_texts):
        raise ValueError(
            f"Length mismatch in domain={domain}, alg={alg}: "
            f"wm_texts={len(wm_texts)} vs plain_texts={len(plain_texts)}"
        )

    return wm_texts, plain_texts, token_subset


# ============================================================
# 3. EXP helper
# ============================================================

def exp_avg_score(wm, text, eps=1e-12):
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

    return float(total_score / num_scored) if num_scored > 0 else 0.0


# ============================================================
# 4. Threshold sweep
# ============================================================

def find_best_threshold_both(wm_scores, plain_scores):
    scores = list(wm_scores) + list(plain_scores)
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

            TP = sum((p == 1 and y == 1) for p, y in zip(preds, labels))
            FP = sum((p == 1 and y == 0) for p, y in zip(preds, labels))
            FN = sum((p == 0 and y == 1) for p, y in zip(preds, labels))
            TN = sum((p == 0 and y == 0) for p, y in zip(preds, labels))

            TPR = TP / (TP + FN) if (TP + FN) else 0.0
            precision = TP / (TP + FP) if (TP + FP) else 0.0
            FPR = FP / (FP + TN) if (FP + TN) else 0.0
            F1 = 2 * precision * TPR / (precision + TPR) if (precision + TPR) else 0.0

            if F1 > best_f1:
                best_f1 = F1
                best_metrics = {
                    "TPR": float(TPR),
                    "F1": float(F1),
                    "Precision": float(precision),
                    "FPR": float(FPR),
                    "Threshold": float(thr),
                    "Direction": direction,
                }

        if best is None or best_metrics["F1"] > best["F1"]:
            best = best_metrics

    return best


def apply_threshold(score, threshold, direction):
    if direction == "gt":
        return int(score > threshold)
    return int(score < threshold)


# ============================================================
# 5. Subset-aware detectors
# ============================================================

class SubsetAwareKGWDetector:
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


SUBSET_DETECTORS = {
    "KGW": SubsetAwareKGWDetector,
    "SWEET": SubsetAwareSWEETDetector,
    "Unigram": SubsetAwareUnigramDetector,
    "EXP": SubsetAwareEXPDetector,
    "SynthID": SubsetAwareSynthIDDetector,
}


# ============================================================
# 6. Metrics / bootstrap / stats
# ============================================================

def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = f1_score(y_true, y_pred, zero_division=0)
    fpr = fp / (fp + tn) if (fp + tn) else 0.0

    return {
        "TPR": float(tpr),
        "F1": float(f1),
        "Precision": float(precision),
        "FPR": float(fpr),
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "TN": int(tn),
    }


def bootstrap_ci(y_true, y_pred, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    n = len(y_true)

    rows = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        rows.append(compute_metrics(y_true[idx], y_pred[idx]))

    out = {}
    for key in ["TPR", "F1", "Precision", "FPR"]:
        vals = np.array([r[key] for r in rows], dtype=float)
        out[key] = {
            "low": float(np.percentile(vals, 2.5)),
            "high": float(np.percentile(vals, 97.5)),
        }
    return out


def mcnemar_exact_from_preds(pred_a, pred_b):
    pred_a = np.asarray(pred_a).astype(int)
    pred_b = np.asarray(pred_b).astype(int)

    b = int(((pred_a == 1) & (pred_b == 0)).sum())
    c = int(((pred_a == 0) & (pred_b == 1)).sum())

    n = b + c
    if n == 0:
        pval = 1.0
    else:
        pval = binomtest(min(b, c), n=n, p=0.5, alternative="two-sided").pvalue

    return {
        "b": b,
        "c": c,
        "n_discordant": n,
        "p_value": float(pval),
    }


def holm_correction(pvals_dict):
    if not pvals_dict:
        return {}

    items = sorted(pvals_dict.items(), key=lambda x: x[1])
    m = len(items)

    adjusted = {}
    prev = 0.0
    for i, (name, p) in enumerate(items, start=1):
        adj = min(1.0, (m - i + 1) * p)
        adj = max(adj, prev)
        adjusted[name] = adj
        prev = adj
    return adjusted


# ============================================================
# 7. Detector builders
# ============================================================

def get_baseline_score_fn(alg, wm):
    if alg == "SynthID":
        baseline_detector = BaselineDetectorFactory(wm).build()
        return lambda text: float(baseline_detector.detect(text))
    elif alg == "EXP":
        return lambda text: float(exp_avg_score(wm, text))
    else:
        return lambda text: float(wm.detect_watermark(text, return_dict=True)["score"])


def get_subset_detector(alg, wm, token_subset):
    cls = SUBSET_DETECTORS[alg]
    return cls(wm, token_subset)


# ============================================================
# 8. Main pipeline
# ============================================================

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 80)
    print("RQ3 multi-domain analysis pipeline")
    print("=" * 80)

    cfg = get_transformers_config()

    all_sample_rows = []
    domain_algorithm_metrics = []

    summary = {
        "setting": SETTING_NAME,
        "domains": DOMAINS,
        "top_k_tokens": TOP_K_TOKENS,
        "algorithms": {},
        "domain_algorithm_metrics": [],
        "friedman_tests": {},
        "posthoc_wilcoxon_holm": {},
        "descriptive_degradation": {},
    }

    # --------------------------------------------------------
    # A. Run detectors for each domain x algorithm
    # --------------------------------------------------------
    for domain in DOMAINS:
        print("=" * 80)
        print(f"Domain: {domain}")
        print("=" * 80)

        for alg in ALGORITHMS:
            print("-" * 80)
            print(f"Algorithm: {alg}")

            wm_texts, plain_texts, token_subset = safe_load_domain_inputs(
                domain=domain,
                alg=alg,
                base_algo_for_plain=BASE_ALGO_FOR_PLAIN,
            )

            wm = AutoWatermark.load(alg, f"config/{alg}.json", cfg)
            baseline_score_fn = get_baseline_score_fn(alg, wm)
            subset_detector = get_subset_detector(alg, wm, token_subset)

            wm_base, plain_base = [], []
            wm_subset, plain_subset = [], []
            algo_rows = []

            for idx, (w, p) in enumerate(
                tqdm(zip(wm_texts, plain_texts), total=len(plain_texts))
            ):
                wb = baseline_score_fn(w)
                pb = baseline_score_fn(p)
                ws = float(subset_detector.detect(w))
                ps = float(subset_detector.detect(p))

                wm_base.append(wb)
                plain_base.append(pb)
                wm_subset.append(ws)
                plain_subset.append(ps)

                algo_rows.append({
                    "setting": SETTING_NAME,
                    "domain": domain,
                    "algorithm": alg,
                    "sample_id": idx,
                    "y_true": 1,
                    "text_type": "wm",
                    "baseline_score": wb,
                    "subset_score": ws,
                })
                algo_rows.append({
                    "setting": SETTING_NAME,
                    "domain": domain,
                    "algorithm": alg,
                    "sample_id": idx,
                    "y_true": 0,
                    "text_type": "plain",
                    "baseline_score": pb,
                    "subset_score": ps,
                })

            baseline_best = find_best_threshold_both(wm_base, plain_base)
            subset_best = find_best_threshold_both(wm_subset, plain_subset)

            for row in algo_rows:
                row["baseline_pred"] = apply_threshold(
                    row["baseline_score"],
                    baseline_best["Threshold"],
                    baseline_best["Direction"],
                )
                row["subset_pred"] = apply_threshold(
                    row["subset_score"],
                    subset_best["Threshold"],
                    subset_best["Direction"],
                )

            df_algo = pd.DataFrame(algo_rows)

            base_metrics = compute_metrics(df_algo["y_true"], df_algo["baseline_pred"])
            subset_metrics = compute_metrics(df_algo["y_true"], df_algo["subset_pred"])

            base_ci = bootstrap_ci(
                df_algo["y_true"].values,
                df_algo["baseline_pred"].values,
                n_boot=N_BOOT,
                seed=BOOT_SEED,
            )
            subset_ci = bootstrap_ci(
                df_algo["y_true"].values,
                df_algo["subset_pred"].values,
                n_boot=N_BOOT,
                seed=BOOT_SEED,
            )
            mcnemar = mcnemar_exact_from_preds(
                df_algo["baseline_pred"].values,
                df_algo["subset_pred"].values,
            )

            degradation_row = {
                "setting": SETTING_NAME,
                "domain": domain,
                "algorithm": alg,
                "subset_size": len(token_subset),
                "baseline_tpr": base_metrics["TPR"],
                "subset_tpr": subset_metrics["TPR"],
                "degradation_tpr": base_metrics["TPR"] - subset_metrics["TPR"],
                "baseline_f1": base_metrics["F1"],
                "subset_f1": subset_metrics["F1"],
                "degradation_f1": base_metrics["F1"] - subset_metrics["F1"],
                "baseline_precision": base_metrics["Precision"],
                "subset_precision": subset_metrics["Precision"],
                "degradation_precision": base_metrics["Precision"] - subset_metrics["Precision"],
                "baseline_fpr": base_metrics["FPR"],
                "subset_fpr": subset_metrics["FPR"],
                "degradation_fpr": subset_metrics["FPR"] - base_metrics["FPR"],
            }

            domain_algorithm_metrics.append(degradation_row)

            summary["algorithms"].setdefault(alg, {})
            summary["algorithms"][alg][domain] = {
                "subset_size": len(token_subset),
                "baseline_threshold": baseline_best,
                "subset_threshold": subset_best,
                "baseline_metrics": base_metrics,
                "subset_metrics": subset_metrics,
                "baseline_ci": base_ci,
                "subset_ci": subset_ci,
                "mcnemar": mcnemar,
            }

            all_sample_rows.extend(algo_rows)

    # --------------------------------------------------------
    # B. Save sample-level and domain-level data
    # --------------------------------------------------------
    df_samples = pd.DataFrame(all_sample_rows)
    df_samples.to_csv(SAMPLE_CSV, index=False, encoding="utf-8-sig")
    print(f"Saved sample-level results to: {SAMPLE_CSV}")

    df_domain_metrics = pd.DataFrame(domain_algorithm_metrics)
    df_domain_metrics.to_csv(DOMAIN_CSV, index=False, encoding="utf-8-sig")
    print(f"Saved domain-level degradation results to: {DOMAIN_CSV}")

    summary["domain_algorithm_metrics"] = domain_algorithm_metrics

    # --------------------------------------------------------
    # C. Friedman + post-hoc Wilcoxon
    # --------------------------------------------------------
    primary_metrics = ["degradation_f1"]
    secondary_metrics = ["degradation_tpr", "degradation_precision", "degradation_fpr"]

    for metric in primary_metrics + secondary_metrics:
        pivot = df_domain_metrics.pivot(index="domain", columns="algorithm", values=metric)
        pivot = pivot.dropna(subset=ALGORITHMS)

        if len(pivot) < 2:
            summary["friedman_tests"][metric] = {
                "error": "Need at least 2 matched domains for Friedman test.",
                "n_domains": int(len(pivot)),
            }
            summary["posthoc_wilcoxon_holm"][metric] = {}
            continue

        arrays = [pivot[alg].values for alg in ALGORITHMS]
        stat, p = friedmanchisquare(*arrays)

        summary["friedman_tests"][metric] = {
            "chi_square": float(stat),
            "p_value": float(p),
            "n_domains": int(len(pivot)),
        }

        pairwise_stats = {}

        if p < ALPHA:
            raw_pvals = {}

            for a, b in combinations(ALGORITHMS, 2):
                x = pivot[a].values
                y = pivot[b].values

                try:
                    w_stat, w_p = wilcoxon(
                        x, y,
                        zero_method="wilcox",
                        alternative="two-sided"
                    )
                except ValueError:
                    w_stat, w_p = 0.0, 1.0

                pair_name = f"{a}__vs__{b}"

                raw_pvals[pair_name] = float(w_p)
                pairwise_stats[pair_name] = {
                    "wilcoxon_stat": float(w_stat),
                    "raw_p_value": float(w_p),
                    "median_diff": float(np.median(x - y)),
                }

            adj = holm_correction(raw_pvals)
            for name in pairwise_stats:
                pairwise_stats[name]["holm_adj_p_value"] = float(adj[name])

        summary["posthoc_wilcoxon_holm"][metric] = pairwise_stats

    # --------------------------------------------------------
    # D. Descriptive degradation summary across domains
    # --------------------------------------------------------
    for alg in ALGORITHMS:
        g = df_domain_metrics[df_domain_metrics["algorithm"] == alg].copy()

        summary["descriptive_degradation"][alg] = {
            "mean_degradation_tpr": float(g["degradation_tpr"].mean()),
            "mean_degradation_f1": float(g["degradation_f1"].mean()),
            "mean_degradation_precision": float(g["degradation_precision"].mean()),
            "mean_degradation_fpr": float(g["degradation_fpr"].mean()),
            "std_degradation_f1": float(g["degradation_f1"].std(ddof=1)) if len(g) > 1 else 0.0,
            "n_domains": int(len(g)),
        }

    # --------------------------------------------------------
    # E. Save summary
    # --------------------------------------------------------
    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Saved summary to: {SUMMARY_JSON}")
    print("=" * 80)
    print("Done.")
    print("=" * 80)


if __name__ == "__main__":
    main()