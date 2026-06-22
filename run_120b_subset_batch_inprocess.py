# ============================================================
# run_120b_subset_batch_inprocess.py
# In-process batch runner for 120B random-subset watermark experiments
# Loads the 120B model only once.
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

datasets = [
    ("dataset/zhtw/mydatasets/ai/output_data_combined_iclr_abstracts_merged_prompt.jsonl", "ai"),
    ("dataset/zhtw/mydatasets/bio/output_data_combined_BIO2_abstracts_merged_prompt.jsonl", "bio"),
    ("dataset/zhtw/mydatasets/med/output_data_combined_MIE_abstracts_merged_prompt.jsonl", "med"),
    ("dataset/zhtw/mydatasets/mis/combined_icis_merged_prompt.jsonl", "mis"),
    ("dataset/zhtw/mydatasets/Security/output_data_combined_SP_abstracts_merged_prompt.jsonl", "security"),
]

MODEL_NAME = "openai/gpt-oss-120b"

MAX_SAMPLES = 30
SAMPLE_SEED = 30
MAX_NEW_TOKENS_CAP = 260
OUTPUT_DIR = "outputs/wm_tokens_120b_subset"

TORCH_DTYPE = "bfloat16"
MAX_MEMORY = "0:76GiB,cpu:200GiB"

RANDOM_SAMPLE = True
SKIP_PLAIN = True
USE_PLAIN_CACHE = True

# For gpt-oss, do not use bitsandbytes 4bit/8bit.
LOAD_IN_4BIT = False
LOAD_IN_8BIT = False


def check_paths():
    print("Current working directory:", os.getcwd())

    missing = []
    for dataset_path, domain in datasets:
        if not os.path.exists(dataset_path):
            missing.append((domain, dataset_path))

    if missing:
        print("\nMissing dataset files:")
        for domain, path in missing:
            print(f"  domain={domain}: {path}")
        raise FileNotFoundError("One or more dataset paths do not exist.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    check_paths()

    failed_jobs = []
    completed_jobs = 0
    total_jobs = len(algorithms) * len(datasets)

    print("=" * 80)
    print("Loading model once")
    print("START:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
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
    print("=" * 80)

    for algorithm in algorithms:
        print("\n" + "#" * 80)
        print(f"Loading watermark: {algorithm}")
        print("#" * 80)

        wm = AutoWatermark.load(
            algorithm,
            f"config/{algorithm}.json",
            cfg,
        )

        for dataset_path, domain in datasets:
            print("\n" + "=" * 80)
            print("START:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            print(f"MODEL: {MODEL_NAME}")
            print(f"ALGORITHM: {algorithm}")
            print(f"DOMAIN: {domain}")
            print(f"DATASET: {dataset_path}")
            print("=" * 80)

            try:
                rewrite_and_collect_120b(
                    algorithm=algorithm,
                    dataset_path=dataset_path,
                    max_samples=MAX_SAMPLES,
                    domain=domain,
                    model_name=MODEL_NAME,
                    output_dir=OUTPUT_DIR,
                    sample_seed=SAMPLE_SEED,
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