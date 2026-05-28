import numpy as np
import torch


# ============================================================
# 1. Base interface
# ============================================================

class BaseTextDetector:
    """
    Unified interface for baseline (non-subset) detectors.
    """

    def detect(self, text: str) -> float:
        raise NotImplementedError


# ============================================================
# 2. Generic wrapper for non-SynthID algorithms
# ============================================================

class SimpleWatermarkDetector(BaseTextDetector):
    """
    Wraps watermark.detect_watermark(text) for KGW / SWEET / EXP / Unigram.
    """

    def __init__(self, watermark):
        self.wm = watermark

    def detect(self, text: str) -> float:
        result = self.wm.detect_watermark(text, return_dict=True)
        return float(result["score"])


# ============================================================
# 3. SynthID baseline detector (NO subset)
# ============================================================

class BaselineSynthIDDetector(BaseTextDetector):
    """
    Canonical SynthID detection:
    - compute g_values
    - compute algorithm-defined mask
    - apply built-in SynthID detector
    """

    def __init__(self, watermark):
        self.wm = watermark
        self.tokenizer = watermark.config.generation_tokenizer
        self.logits_processor = watermark.logits_processor
        self.detector = watermark.detector  # mean / weighted / bayesian

        self.ngram_len = watermark.config.ngram_len
        self.device = watermark.config.device

    def detect(self, text: str) -> float:
        # ----------------------------------------------------
        # 1. tokenize
        # ----------------------------------------------------
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=False
        )["input_ids"].to(self.device)

        # ----------------------------------------------------
        # 2. compute g-values
        # ----------------------------------------------------
        g_values = self.logits_processor.compute_g_values(encoded)
        # shape: [1, seq_len - (n-1), depth]

        # ----------------------------------------------------
        # 3. build original SynthID mask
        # ----------------------------------------------------
        eos_mask = self.logits_processor.compute_eos_token_mask(
            encoded,
            self.tokenizer.eos_token_id
        )[:, self.ngram_len - 1:]

        if self.wm.config.watermark_mode == "non-distortionary":
            repetition_mask = self.logits_processor.compute_context_repetition_mask(
                encoded
            )
            mask = eos_mask * repetition_mask
        else:
            mask = eos_mask

        # ----------------------------------------------------
        # 4. detect
        # ----------------------------------------------------
        score = self.detector.detect(
            g_values.cpu().numpy(),
            mask.cpu().numpy()
        )[0]

        return float(score)


# ============================================================
# 4. Factory
# ============================================================

class BaselineDetectorFactory:
    """
    Factory to build baseline detectors for all watermark algorithms.
    """

    def __init__(self, watermark):
        self.wm = watermark
        self.alg = watermark.config.algorithm_name

    def build(self) -> BaseTextDetector:
        if self.alg in {"KGW", "SWEET", "Unigram", "EXP"}:
            return SimpleWatermarkDetector(self.wm)

        if self.alg == "SynthID":
            return BaselineSynthIDDetector(self.wm)

        raise ValueError(f"Unsupported algorithm: {self.alg}")
