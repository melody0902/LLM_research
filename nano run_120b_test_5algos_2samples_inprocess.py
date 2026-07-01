# ============================================================
# run_120b_test_5algos_2samples_inprocess.py
# Test runner:
# - load openai/gpt-oss-120b once
# - run 5 watermark algorithms
# - run ai domain only
# - run sequential samples 0~1
# - use chat template for gpt-oss
# - write to a separate test output dir
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
]

MODEL_NAME = "openai/gpt-oss-120b"

START_INDEX = 0
MAX_SAMPLES = 2
MAX_NEW_TOKENS_CAP = 260

OUTPUT_DIR = "outputs/wm_tokens_120b_test5algos_2samples"

TORCH_DTYPE = "bfloat16"

# If your Pod has only 125 GB RAM, use this:
MAX_MEMORY = "0:72GiB,cpu:110GiB"

# If your Pod has 251 GB RAM, you can use this instead:
# MAX_MEMORY = "0:72GiB,cpu:220GiB"

RANDOM_SAMPLE = False

SKIP_PLAIN = True
USE_PLAIN_CACHE = True

LOAD_IN_4BIT = False
LOAD_IN_8BIT = False

SAMPLE_SEED = 30


def check_paths():
    print("Current working directory:", os.getcwd())

    missing = []

    if not os.path.exists("rewrite_and_collect_watermark_tokens_120b.py"):
        missing.append(("script", "rewrite_and_collect_watermark_tokens_120b.py"))

    for dataset_path, domain in datasets:
        if not os.path.exists(dataset_path):
            missing.append((domain, dataset_path))

    for algorithm in algorithms:
        config_path = f"config/{algorithm}.json"
        if not os.path.exists(config_path):
            missing.append((algorithm, config_path))

    if missing:
        print("\nMissing files:")
        for name, path in missing:
            print(f"  {name}: {path}")
        raise FileNotFoundError("One or more required paths do not exist.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    check_paths()

    os.environ.setdefault("DEBUG_RAW_CLEAN", "1")

    failed_jobs = []
    completed_jobs = 0
    total_jobs = len(algorithms) * len(datasets)

    print("=" * 80)
    print("Loading model once")
    print("START:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print(f"MODEL: {MODEL_NAME}")
    print(f"MAX_MEMORY: {MAX_MEMORY}")
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
    print(f"Output dir: {OUTPUT_DIR}")
    print("=" * 80)

    for algorithm in algorithms:
        for dataset_path, domain in datasets:
            print("\n" + "#" * 80)
            print(f"Loading watermark: {algorithm} / {domain}")
            print("#" * 80)

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
            print(f"USE_CHAT_TEMPLATE: True")
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
                    sample_seed=SAMPLE_SEED,
                    random_sample=RANDOM_SAMPLE,
                    cfg=cfg,
                    wm=wm,
                    use_plain_cache=USE_PLAIN_CACHE,
                    skip_plain=SKIP_PLAIN,
                    max_new_tokens_cap=MAX_NEW_TOKENS_CAP,
                    use_chat_template=True,
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