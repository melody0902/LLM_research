import yaml
import torch
from pathlib import Path

def load_yaml_config(config_path: str = "config/model_config.yaml") -> dict:
    """載入 YAML 配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 處理特殊值轉換
    if config['model']['load_params']['torch_dtype'] == 'bfloat16':
        config['model']['load_params']['torch_dtype'] = torch.bfloat16
    
    # 處理量化設定中的 dtype
    if 'quantization' in config['model'] and config['model']['quantization']['bnb_4bit_compute_dtype'] == 'bfloat16':
        config['model']['quantization']['bnb_4bit_compute_dtype'] = torch.bfloat16
    
    return config 