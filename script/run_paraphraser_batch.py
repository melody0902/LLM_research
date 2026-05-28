import os
import subprocess
import shutil

def run_command(cmd, output_file):
    print(f"指令: {cmd}")
    print(f"輸出: {output_file}")
    print("-" * 80)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        subprocess.run(cmd, shell=True, stdout=f)
    print(f"完成: {output_file}")

def generate():
    # model = "llama3.1"
    model = 'opt1.3b'
    algorithm = "SWEET"
    max_samples = 5000
    
    datasets = [
        # ("dataset/zhtw/processed_zhtw_c4.json", "zhc4"),
        ("dataset/c4/processed_c4.json", "enc4")
    ]
    
    deltas = [2, 1, 0.8]
    n_grams = [2]
    # n_grams = [1, 2, 3, 4, 5]
    
    for dataset_path, dataset_name in datasets:
        for delta in deltas:
            for n in n_grams:
                algorithm_lower = algorithm.lower()
                output_dir = f"tables_data_{max_samples}/{model}/{algorithm_lower}/{dataset_name}_d{delta}/{n}-gram"
                watermarked_texts_path = f"{output_dir}/watermarked_texts.json"
                
                cmd = (f"python3 script/paraphraser.py "
                       f"--algorithm {algorithm} "
                       f"--max_samples {max_samples} "
                       f"--output_dir {output_dir} "
                       f"--watermarked_texts_path {watermarked_texts_path} "
                       f"--dataset {dataset_path} "
                       f"--generation_mode=generate "
                       f"--delta={delta} "
                       f"--n={n}")
                
                output_file = f"{output_dir}/res.txt"
                run_command(cmd, output_file)

def detect():
    # model = "llama3.1"
    model = 'opt1.3b'
    algorithm = "Unigram"
    max_samples = 5000
    
    datasets = [
        # ("dataset/zhtw/processed_zhtw_c4.json", "zhc4"),
        ("dataset/c4/processed_c4.json", "enc4")
    ]
    
    deltas = [2, 1, 0.8]
    # n_grams = [2]
    n_grams = [1, 3, 4, 5]
    
    for dataset_path, dataset_name in datasets:
        for delta in deltas:
            for n in n_grams:
                algorithm_lower = algorithm.lower()
                output_dir = f"tables_data_{max_samples}/{model}/{algorithm_lower}/{dataset_name}_d{delta}/{n}-gram"
                watermarked_texts_path = f"tables_data_{max_samples}/{model}/{algorithm_lower}/{dataset_name}_d{delta}/2-gram/watermarked_texts.json"
                
                cmd = (f"python3 script/paraphraser.py "
                       f"--algorithm {algorithm} "
                       f"--max_samples {max_samples} "
                       f"--output_dir {output_dir} "
                       f"--watermarked_texts_path {watermarked_texts_path} "
                       f"--dataset {dataset_path} "
                       f"--generation_mode=load "
                       f"--delta={delta} "
                       f"--n={n}")
                
                output_file = f"{output_dir}/res.txt"
                run_command(cmd, output_file)

def move_watermarked_texts() -> None:
    """將 watermarked_texts.json 從 tables_data_5000 目錄移動到 texts5000 目錄。
    
    保持相同的目錄結構: {model}/{algorithm_lower}/{dataset_name}_d{delta}/{n}-gram/
    """
    model = 'opt1.3b'
    algorithms = ["kgw", "sweet", "unigram"]  # 根據你的算法設定
    max_samples = 5000
    
    datasets = [
        # ("enc4", "enc4")  # (dataset_name, dataset_name)
        # 如果有其他 dataset 可以在這裡添加
        ("mbpp", "mbpp")
    ]
    
    deltas = [2, 1, 0.8, 0.5, 0.1]
    
    for algorithm in algorithms:
        for dataset_name, _ in datasets:
            for delta in deltas:
                # 來源路徑
                source_path = f"tables_data_{max_samples}/{model}/{algorithm}/{dataset_name}_d{delta}/2-gram/watermarked_texts.json"
                
                # 目標路徑
                target_path = f"texts5000/{model}/{algorithm}/{dataset_name}_d{delta}/watermarked_texts.json"
                
                # 檢查來源文件是否存在
                if os.path.exists(source_path):
                    # 確保目標目錄存在
                    os.makedirs(os.path.dirname(target_path), exist_ok=True)
                    
                    # 移動文件
                    try:
                        shutil.move(source_path, target_path)
                        print(f"成功移動: {source_path} -> {target_path}")
                    except Exception as e:
                        print(f"移動失敗: {source_path} -> {target_path}, 錯誤: {e}")
                else:
                    print(f"來源文件不存在: {source_path}")

def code_generation():
    # model = "llama3.1"
    # algorithms = ["KGW", "SWEET", "Unigram"]
    # max_samples = 1000
    # model = 'opt1.3b'
    model = 'llama3.1'
    algorithms = ["EXP"]
    max_samples = 1
    
    datasets = [
        # ("dataset/human_eval/test.jsonl", "he")
        # ("dataset/mbpp/mbpp.jsonl", "mbpp")
        ('dataset/c4/processed_c4.json', 'enc4')
    ]
    
    deltas = [1, 0.8, 0.5]

    
    for dataset_path, dataset_name in datasets:
        for algorithm in algorithms:
            for delta in deltas:
                algorithm_lower = algorithm.lower()
                output_dir = f"tables_data_{max_samples}/{model}/{algorithm_lower}/{dataset_name}_d{delta}"
                watermarked_texts_path = f"{output_dir}/watermarked_texts.json"
                
                cmd = (f"python3 script/paraphraser.py "
                        f"--algorithm {algorithm} "
                        f"--max_samples {max_samples} "
                        f"--output_dir {output_dir} "
                        f"--watermarked_texts_path {watermarked_texts_path} "
                        f"--dataset {dataset_path} "
                        f"--generation_mode=generate "
                        f"--delta={delta} ")
                
                output_file = f"{output_dir}/res.txt"
                run_command(cmd, output_file)

def exp_generate():
    model = 'llama3.1'
    algorithms = ["EXP"]
    max_samples = 1000
    
    datasets = [
        ('dataset/c4/processed_c4.json', 'enc4')
    ]   

    temperatures = [0.3]

    for dataset_path, dataset_name in datasets:
        for algorithm in algorithms:
            for temperature in temperatures:
                algorithm_lower = algorithm.lower()
                output_dir = f"tables_data_{max_samples}/{model}/{algorithm_lower}/{dataset_name}_t{temperature}"
                watermarked_texts_path = f"{output_dir}/watermarked_texts.json"
                
                cmd = (f"python3 script/paraphraser.py "
                    f"--algorithm {algorithm} "
                    f"--max_samples {max_samples} "
                    f"--output_dir {output_dir} "
                    f"--watermarked_texts_path {watermarked_texts_path} "
                    f"--dataset {dataset_path} "
                    f"--generation_mode=generate "
                    f"--temperature={temperature}")
                
                output_file = f"{output_dir}/res.txt"
                run_command(cmd, output_file)

def main():
    # generate()
    # detect()
    # move_watermarked_texts()
    # code_generation()
    exp_generate()

if __name__ == "__main__":
    main() 