import random
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.transformers_config import TransformersConfig
from utils.config_loader import load_yaml_config
from transformers import BitsAndBytesConfig

def load_model_and_config(config_path: str = "config/model_config.yaml") -> tuple[AutoModelForCausalLM, AutoTokenizer, TransformersConfig]:
    """Load model, tokenizer, and create TransformersConfig from a config file.

    Args:
        config_path: Path to the configuration YAML file.

    Returns:
        tuple: (model, tokenizer, transformers_config)
    """
    config = load_yaml_config(config_path)

    seed = 30
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    model_params = config['model']['load_params'].copy()

    if 'quantization' in config['model'] and config['model']['quantization']['enabled']:
        quant_config = config['model']['quantization']
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=quant_config['load_in_4bit'],
            bnb_4bit_quant_type=quant_config['bnb_4bit_quant_type'],
            bnb_4bit_use_double_quant=quant_config['bnb_4bit_use_double_quant'],
            bnb_4bit_compute_dtype=quant_config['bnb_4bit_compute_dtype']
        )
        model_params['quantization_config'] = quantization_config

    try:
        model = AutoModelForCausalLM.from_pretrained(
            config['model']['name'],
            local_files_only=True,
            cache_dir=config['model']['cache_dir'],
            **model_params
        )
    except OSError:
        model = AutoModelForCausalLM.from_pretrained(
            config['model']['name'],
            cache_dir=config['model']['cache_dir'],
            **model_params
        )

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            config['model']['name'],
            local_files_only=True,
            **config['model']['tokenizer_params']
        )
    except OSError:
        tokenizer = AutoTokenizer.from_pretrained(
            config['model']['name'],
            **config['model']['tokenizer_params']
        )

    transformers_config = TransformersConfig(
        model=model,
        tokenizer=tokenizer,
        vocab_size=len(tokenizer.get_vocab()),
        device='cuda' if torch.cuda.is_available() else 'cpu',
        **config['transformers']
    )

    return model, tokenizer, transformers_config