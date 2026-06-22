# ============================================================
# run_120b_subset_batch.py
# Batch runner for 120B random-subset watermark experiments
# ============================================================
# tar -czf /workspace/results_output.tar.gz outputs/wm_tokens_120b_subset/
# scp -P 12680 -i ~/.ssh/id_ed25519 root@205.196.17.138:/workspace/results_output.tar.gz ~/Downloads/



import os
import shlex
import subprocess
from datetime import datetime


algorithms = ["KGW", "SWEET", "Unigram", "EXP", "SynthID"]

datasets = [
    ("dataset/zhtw/mydatasets/ai/output_data_combined_iclr_abstracts_merged_prompt.jsonl", "ai"),
    ("dataset/zhtw/mydatasets/bio/output_data_combined_BIO2_abstracts_merged_prompt.jsonl", "bio"),
    ("dataset/zhtw/mydatasets/med/output_data_combined_MIE_abstracts_merged_prompt.jsonl", "med"),
    ("dataset/zhtw/mydatasets/mis/combined_icis_merged_prompt.jsonl", "mis"),
    ("dataset/zhtw/mydatasets/Security/output_data_combined_SP_abstracts_merged_prompt.jsonl", "security"),
]

# Change this to your target 120B model.
models = [
    "openai/gpt-oss-120b",
]

# Recommended 120B subset size.
MAX_SAMPLES = 30
SAMPLE_SEED = 30
MAX_NEW_TOKENS_CAP = 200
OUTPUT_DIR = "outputs/wm_tokens_120b_subset"
SCRIPT = "rewrite_and_collect_watermark_tokens_120b.py"

# For 80GB GPU, start with bf16 and no quantization.
# For limited VRAM, add "--load_in_4bit" or "--load_in_8bit" below.
EXTRA_ARGS = [
    "--random_sample",
    "--torch_dtype", "bfloat16",
    # "--load_in_4bit",
    # "--max_memory", "0:78GiB,cpu:200GiB",
    "--skip_plain",  # Optional: use only if plain baseline is too expensive.
]


def run_command(cmd_list):
    print("\n" + "=" * 80)
    print("START:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("CMD:", " ".join(shlex.quote(x) for x in cmd_list))
    print("=" * 80)

    result = subprocess.run(cmd_list)

    print("=" * 80)
    print("END:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("RETURN CODE:", result.returncode)
    print("=" * 80)

    if result.returncode != 0:
        raise RuntimeError(f"Command failed with return code {result.returncode}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for model_name in models:
        for algo in algorithms:
            for dataset_path, domain in datasets:
                cmd = [
                    "python", SCRIPT,
                    "--model_name", model_name,
                    "--algorithm", algo,
                    "--dataset", dataset_path,
                    "--domain", domain,
                    "--max_samples", str(MAX_SAMPLES),
                    "--sample_seed", str(SAMPLE_SEED),
                    "--max_new_tokens_cap", str(MAX_NEW_TOKENS_CAP),
                    "--output_dir", OUTPUT_DIR,
                ] + EXTRA_ARGS

                run_command(cmd)


if __name__ == "__main__":
    main()
