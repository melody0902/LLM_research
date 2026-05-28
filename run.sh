#!/bin/bash

# Algorithms
algorithms=("KGW" "SWEET" "Unigram")

# Dataset path 和 max_samples 對應
declare -A datasets
datasets["ai"]="dataset/zhtw/mydatasets/ai/output_data_combined_iclr_abstracts.json 100"
datasets["bio"]="dataset/zhtw/mydatasets/bio/combined_BIO2_abstracts.json 100"
datasets["security"]="dataset/zhtw/mydatasets/Security/combined_SP_abstracts.json 100"
datasets["med"]="dataset/zhtw/mydatasets/med/combined_MIE_abstracts.json 100"
datasets["mis"]="dataset/zhtw/mydatasets/mis/combined_icis.json 100"
datasets["edu"]="dataset/zhtw/mydatasets/edu/acm_abstracts.json 100"

# N 值
ns=(2 3 4)

# run 次數
num_runs=10

# 輸出資料夾
outputDir="run_10/signature_sets"
mkdir -p "$outputDir"

# 主迴圈
for algo in "${algorithms[@]}"; do
    for domain in "${!datasets[@]}"; do
        dataset_info=(${datasets[$domain]})
        path=${dataset_info[0]}
        samples=${dataset_info[1]}

        for n in "${ns[@]}"; do
            algoLower=$(echo "$algo" | tr '[:upper:]' '[:lower:]')

            for run_id in $(seq 1 $num_runs); do
                outputFile="$outputDir/${algoLower}_${domain}_sig_n${n}_run${run_id}.json"
                echo "🚀 Running: $algo on $domain (n=$n, run=$run_id, random $samples samples)"

                python script/paraphraser_ngram.py \
                    --generate_signature \
                    --algorithm "$algo" \
                    --dataset "$path" \
                    --max_samples "$samples" \
                    --signature_output "$outputFile" \
                    --n "$n" \
                    --seed "$run_id"
            done
        done
    done
done
