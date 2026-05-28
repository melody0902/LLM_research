import os
import sys

def run_command(cmd):
    print(f"⚙️ 執行：{cmd}")
    ret = os.system(cmd)
    if ret != 0:
        print(f"錯誤：指令執行失敗 -> {cmd}")
        sys.exit(1)
    print("完成\n")


def batch_rewrite_and_collect():
    algorithms = ["KGW", "SWEET","Unigram", "EXP", "SynthID"] 


    # models = [
    # "meta-llama/Llama-3.1-8B-Instruct"
    # ]
    models = [
        "Qwen/Qwen2.5-7B-Instruct",
        "01-ai/Yi-1.5-9B-Chat"
    ] 

    output_dir = "outputs/0517_200green/new_model"

    # datasets = [
    #     ("dataset/zhtw/mydatasets/ai/output_data_combined_iclr_abstracts_merged_prompt.jsonl", "ai"),
    #     ("dataset/zhtw/mydatasets/bio/output_data_combined_BIO2_abstracts_merged_prompt.jsonl", "bio"),
    #     ("dataset/zhtw/mydatasets/med/output_data_combined_MIE_abstracts_merged_prompt.jsonl", "med"),
    #     ("dataset/zhtw/mydatasets/mis/combined_icis_merged_prompt.jsonl", "mis"),
    #     ("dataset/zhtw/mydatasets/Security/output_data_combined_SP_abstracts_merged_prompt.jsonl", "security")
    # ]

    # datasets = [
    #      ("dataset/zhtw/mydatasets/ai/output_data_combined_iclr_abstracts_merged_prompt.jsonl", "ai"),
    # ]
    datasets = [
        ("dataset/zhtw/mydatasets/bio/output_data_combined_BIO2_abstracts_merged_prompt.jsonl", "bio"),
        ("dataset/zhtw/mydatasets/med/output_data_combined_MIE_abstracts_merged_prompt.jsonl", "med"),
        ("dataset/zhtw/mydatasets/mis/combined_icis_merged_prompt.jsonl", "mis"),
        ("dataset/zhtw/mydatasets/Security/output_data_combined_SP_abstracts_merged_prompt.jsonl", "security")
    ]

   

    for model_name in models:
        for algo in algorithms:
            for dataset_path, domain in datasets:
                cmd = (
                    f'python rewrite_and_collect_watermark_tokens.py '
                    f'--model_name "{model_name}" '
                    f'--algorithm {algo} '
                    f'--dataset "{dataset_path}" '
                    f'--domain {domain} '
                    f'--max_samples 200 '
                    f'--start_index 0 '
                    f'--output_dir "{output_dir}"'
                )
                run_command(cmd)


if __name__ == "__main__":
    batch_rewrite_and_collect()