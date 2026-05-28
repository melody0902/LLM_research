# Algorithms
# $algorithms = @("KGW", "SWEET", "UNIGRAM")
$algorithms = @("Unigram")

# Dataset path 和 max_samples 對應
$datasets = @{
    "ai"       = @("dataset/zhtw/mydatasets/ai/output_data_combined_iclr_abstracts.json", 100)
    "bio"      = @("dataset/zhtw/mydatasets/bio/combined_BIO2_abstracts.json", 100)
    "security" = @("dataset/zhtw/mydatasets/Security/combined_SP_abstracts.json", 100)
    "med"      = @("dataset/zhtw/mydatasets/med/combined_MIE_abstracts.json", 100)
    "mis"      = @("dataset/zhtw/mydatasets/mis/combined_icis.json", 100)
    "edu"      = @("dataset/zhtw/mydatasets/edu/acm_abstracts.json", 100)
}

# N 值
$ns = @(2, 3, 4)

# 輸出資料夾
$outputDir = "random/signature_sets"
New-Item -ItemType Directory -Force -Path $outputDir | Out-Null

# 主迴圈
foreach ($algo in $algorithms) {
    foreach ($domain in $datasets.Keys) {
        $path = $datasets[$domain][0]
        $samples = $datasets[$domain][1]

        foreach ($n in $ns) {
            $algoLower = $algo.ToLower()
            $outputFile = "$outputDir/${algoLower}_${domain}_sig_n${n}.json"
            Write-Host "Running: $algo on $domain (n=$n, random $samples samples)"
            python script/paraphraser_ngram.py `
                --generate_signature `
                --algorithm $algo `
                --dataset "$path" `
                --max_samples $samples `
                --signature_output "$outputFile" `
                --n $n
        }
    }
}
