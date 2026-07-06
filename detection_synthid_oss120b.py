"""OSS 120B variant for detection_synthid.py.

Place this file next to detection_synthid.py and run:
    python detection_synthid_oss120b.py

Important:
    This file does NOT reuse the original 8B data paths.
    Edit the OSS_120B_* path constants below to match your 120B generated files.
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import detection_synthid as base
from utils.transformers_config import TransformersConfig


# ============================================================
# OSS 120B settings
# ============================================================

OSS_120B_MODEL_NAME = "openai/gpt-oss-120b"
OSS_120B_MODEL_TAG = OSS_120B_MODEL_NAME.replace("/", "__")

OSS_120B_TEXT_JSON_TMPL = (
    "/home/soslab/Desktop/Melody/signature/llm-watermark-research/"
    "outputs/wm_tokens_120b_0_200/"
    "rewritten_{domain}_{alg}_{model_tag}_sequential_seed30_n200_wm_tokens.json"
)

OSS_120B_TOKEN_SET_TMPL = (
    "/home/soslab/Desktop/Melody/signature/llm-watermark-research/"
    "outputs/wm_tokens_120b_0_200/"
    "rewritten_{domain}_{alg}_{model_tag}_sequential_seed30_n200_wm_token_freq.json"
)

OSS_120B_OUTPUT_PATH = (
    f"outputs/test/detect/oss120b/"
    f"synthid_subset_vs_baseline_{base.DOMAIN}_{OSS_120B_MODEL_TAG}_avoid_top{base.TOP_K_TOKENS}.json"
)


# ============================================================
# Override base detection_synthid.py globals
# ============================================================

base.MODEL_NAME = OSS_120B_MODEL_NAME
base.MODEL_TAG = OSS_120B_MODEL_TAG
base.TEXT_JSON_TMPL = OSS_120B_TEXT_JSON_TMPL
base.TOKEN_SET_TMPL = OSS_120B_TOKEN_SET_TMPL
base.OUTPUT_PATH = OSS_120B_OUTPUT_PATH


def _check_required_files():
    """Fail early if the OSS 120B input files are not where this script expects."""
    missing = []

    plain_path = base.TEXT_JSON_TMPL.format(
        domain=base.DOMAIN,
        alg=base.BASE_ALGO_FOR_PLAIN,
        model_tag=base.MODEL_TAG,
        top_k=base.TOP_K_TOKENS,
    )

    wm_path = base.TEXT_JSON_TMPL.format(
        domain=base.DOMAIN,
        alg=base.ALGORITHM,
        model_tag=base.MODEL_TAG,
        top_k=base.TOP_K_TOKENS,
    )

    token_path = base.TOKEN_SET_TMPL.format(
        domain=base.DOMAIN,
        alg=base.ALGORITHM,
        model_tag=base.MODEL_TAG,
        top_k=base.TOP_K_TOKENS,
    )

    for path in [plain_path, wm_path, token_path]:
        if not os.path.exists(path):
            missing.append(path)

    if missing:
        raise FileNotFoundError(
            "OSS 120B input files were not found. "
            "Please update OSS_120B_TEXT_JSON_TMPL and OSS_120B_TOKEN_SET_TMPL.\n"
            + "\n".join(f"  - {p}" for p in missing)
        )


def get_transformers_config():
    print(f"Using model: {base.MODEL_NAME}")
    print(f"Using model tag: {base.MODEL_TAG}")
    print(f"Using text path template: {base.TEXT_JSON_TMPL}")
    print(f"Using token-set path template: {base.TOKEN_SET_TMPL}")
    print(f"Output path: {base.OUTPUT_PATH}")

    model = AutoModelForCausalLM.from_pretrained(
        base.MODEL_NAME,
        device_map="auto",
        torch_dtype="auto",
        low_cpu_mem_usage=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(base.MODEL_NAME)

    return TransformersConfig(
        model=model,
        tokenizer=tokenizer,
        vocab_size=len(tokenizer),
        device="cuda" if torch.cuda.is_available() else "cpu",
        max_new_tokens=200,
        do_sample=False,
    )


# Make base.main() use the OSS 120B loader above.
base.get_transformers_config = get_transformers_config


if __name__ == "__main__":
    _check_required_files()
    base.main()
