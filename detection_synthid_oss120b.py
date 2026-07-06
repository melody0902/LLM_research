"""OSS 120B variant for detection_synthid.py.

Place this file next to detection_synthid.py and run:
    python detection_synthid_oss120b.py
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import detection_synthid as base
from utils.transformers_config import TransformersConfig


OSS_120B_MODEL_NAME = "openai/gpt-oss-120b"


# Override the original 8B model settings.
base.MODEL_NAME = OSS_120B_MODEL_NAME
base.MODEL_TAG = base.MODEL_NAME.replace("/", "__")
base.OUTPUT_PATH = (
    f"outputs/test/detect/"
    f"synthid_subset_vs_baseline_{base.DOMAIN}_{base.MODEL_TAG}_avoid_top{base.TOP_K_TOKENS}.json"
)


def get_transformers_config():
    print(f"Using model: {base.MODEL_NAME}")
    print(f"Using model tag: {base.MODEL_TAG}")

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
    base.main()
