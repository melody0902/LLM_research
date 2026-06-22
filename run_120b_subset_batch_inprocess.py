# ============================================================
# run_120b_subset_batch_inprocess.py
# In-process batch runner for 120B sequential watermark experiments
# Loads the 120B model only once.
#
# This version:
# - does NOT random select samples
# - uses start_index=0 and max_samples=200
# - runs samples 0~199 for each dataset
# - reloads watermark per algorithm/domain job for safer state isolation
# ============================================================

import os
import json
import traceback
from datetime import datetime

from rewrite_and_collect_watermark_tokens_120b import (
    get_transformers_config,
    rewrite_and_collect_120b,
)

from watermark.auto_watermark import AutoWatermark


algorithms = ["KGW", "SWEET", "Unigram", "EXP", "SynthID"]
# algorithms = ["SynthID"]

datasets = [
    ("dataset/zhtw/mydatasets/ai/output_data_combined_iclr_abstracts_merged_prompt.jsonl", "ai"),
    ("dataset/zhtw/mydatasets/bio/output_data_combined_BIO2_abstracts_merged_prompt.jsonl", "bio"),
    ("dataset/zhtw/mydatasets/med/output_data_combined_MIE_abstracts_merged_prompt.jsonl", "med"),
    ("dataset/zhtw/mydatasets/mis/combined_icis_merged_prompt.jsonl", "mis"),
    ("dataset/zhtw/mydatasets/Security/output_data_combined_SP_abstracts_merged_prompt.jsonl", "security"),
]

MODEL_NAME = "openai/gpt-oss-120b"

START_INDEX = 0
MAX_SAMPLES = 200

# 之前 smoke test 200 會截斷，所以建議 260。
# 如果你想更保守可以改回 230。
MAX_NEW_TOKENS_CAP = 260

OUTPUT_DIR = "outputs/wm_tokens_120b_0_200"

TORCH_DTYPE = "bfloat16"
MAX_MEMORY = "0:76GiB,cpu:200GiB"

# Important:
# False = sequential 0~199
# True = random subset
RANDOM_SAMPLE = False

SKIP_PLAIN = True
USE_PLAIN_CACHE = True

# For gpt-oss, do not use bitsandbytes 4bit/8bit.
LOAD_IN_4BIT = False
LOAD_IN_8BIT = False


def check_paths():
    print("Current working directory:", os.getcwd())

    missing = []

    if not os.path.exists("rewrite_and_collect_watermark_tokens_120b.py"):
        missing.append(("script", "rewrite_and_collect_watermark_tokens_120b.py"))

    for dataset_path, domain in datasets:
        if not os.path.exists(dataset_path):
            missing.append((domain, dataset_path))

    if missing:
        print("\nMissing files:")
        for name, path in missing:
            print(f"  {name}: {path}")
        raise FileNotFoundError("One or more required paths do not exist.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    check_paths()

    failed_jobs = []
    completed_jobs = 0
    total_jobs = len(algorithms) * len(datasets)

    print("=" * 80)
    print("Loading model once")
    print("START:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print(f"MODEL: {MODEL_NAME}")
    print("=" * 80)

    cfg = get_transformers_config(
        model_name=MODEL_NAME,
        load_in_4bit=LOAD_IN_4BIT,
        load_in_8bit=LOAD_IN_8BIT,
        torch_dtype=TORCH_DTYPE,
        max_memory=MAX_MEMORY,
    )

    print("=" * 80)
    print("Model loaded")
    print("Total jobs:", total_jobs)
    print(f"Sequential samples: start_index={START_INDEX}, max_samples={MAX_SAMPLES}")
    print("=" * 80)

    for algorithm in algorithms:
        for dataset_path, domain in datasets:
            print("\n" + "#" * 80)
            print(f"Loading watermark: {algorithm} / {domain}")
            print("#" * 80)

            # Reload watermark for every algorithm/domain job.
            # This is safer than reusing one wm across datasets.
            wm = AutoWatermark.load(
                algorithm,
                f"config/{algorithm}.json",
                cfg,
            )

            print("\n" + "=" * 80)
            print("START:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            print(f"MODEL: {MODEL_NAME}")
            print(f"ALGORITHM: {algorithm}")
            print(f"DOMAIN: {domain}")
            print(f"DATASET: {dataset_path}")
            print(f"START_INDEX: {START_INDEX}")
            print(f"MAX_SAMPLES: {MAX_SAMPLES}")
            print(f"RANDOM_SAMPLE: {RANDOM_SAMPLE}")
            print("=" * 80)

            try:
                rewrite_and_collect_120b(
                    algorithm=algorithm,
                    dataset_path=dataset_path,
                    max_samples=MAX_SAMPLES,
                    domain=domain,
                    model_name=MODEL_NAME,
                    output_dir=OUTPUT_DIR,
                    start_index=START_INDEX,
                    sample_seed=30,
                    random_sample=RANDOM_SAMPLE,
                    cfg=cfg,
                    wm=wm,
                    use_plain_cache=USE_PLAIN_CACHE,
                    skip_plain=SKIP_PLAIN,
                    max_new_tokens_cap=MAX_NEW_TOKENS_CAP,
                    use_chat_template=False,
                    load_in_4bit=LOAD_IN_4BIT,
                    load_in_8bit=LOAD_IN_8BIT,
                    torch_dtype=TORCH_DTYPE,
                    max_memory=MAX_MEMORY,
                )

                completed_jobs += 1
                return_code = 0

            except Exception as e:
                return_code = 1
                print("[ERROR] Job failed:")
                traceback.print_exc()

                failed_jobs.append({
                    "model_name": MODEL_NAME,
                    "algorithm": algorithm,
                    "domain": domain,
                    "dataset_path": dataset_path,
                    "start_index": START_INDEX,
                    "max_samples": MAX_SAMPLES,
                    "random_sample": RANDOM_SAMPLE,
                    "error": repr(e),
                })

            print("=" * 80)
            print("END:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            print("RETURN CODE:", return_code)
            print("=" * 80)

    print("\n" + "=" * 80)
    print("BATCH SUMMARY")
    print("=" * 80)
    print(f"Completed jobs: {completed_jobs}/{total_jobs}")
    print(f"Failed jobs: {len(failed_jobs)}")

    if failed_jobs:
        failed_path = os.path.join(OUTPUT_DIR, "failed_jobs.json")
        with open(failed_path, "w", encoding="utf-8") as f:
            json.dump(failed_jobs, f, ensure_ascii=False, indent=2)

        print(f"Saved failed jobs to: {failed_path}")


if __name__ == "__main__":
    main()