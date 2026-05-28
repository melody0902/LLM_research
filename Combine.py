import os
import torch
import json
import numpy as np
import random

from transformers import AutoModelForCausalLM, AutoTokenizer
from evaluation.dataset import C4Dataset
from watermark.auto_watermark import AutoWatermark
from utils.transformers_config import TransformersConfig

from visualize.visualizer import DiscreteVisualizer
from visualize.font_settings import FontSettings
from visualize.legend_settings import DiscreteLegendSettings
from visualize.page_layout_settings import PageLayoutSettings
from visualize.color_scheme import ColorSchemeForDiscreteVisualization

# =============================
# 1. Set random seed
# =============================
seed = 30
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

# =============================
# 2. Set device and model
# =============================
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_name = 'facebook/opt-1.3b'

print(f"Loading model: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

# =============================
# 3. Load watermark
# =============================
transformers_config = TransformersConfig(
    model=model,
    tokenizer=tokenizer,
    vocab_size=50272,
    device=device,
    max_new_tokens=200,
    do_sample=True,
    min_length=230,
    no_repeat_ngram_size=4
)

my_watermark = AutoWatermark.load(
    'KGW',
    algorithm_config='config/KGW.json',
    transformers_config=transformers_config
)
my_watermark.config.prefix_length = 0

# =============================
# 4. Load signature set
# =============================
signature_path = "signature_sets/kgw_sig.json"
with open(signature_path, "r", encoding="utf-8") as f:
    signature_token_ids = set(json.load(f))
print(f"Loaded {len(signature_token_ids)} signature tokens from {signature_path}")

# =============================
# 5. Custom visualizer: support black tokens
# =============================
# def _highlight_single_token(self, draw, token, value, token_width, show_text, x, y, token_id=None):
#     try:
#         token_ids = self.tokenizer.encode(token, add_special_tokens=False)
#         token_id = token_ids[0] if token_ids else -1
#     except Exception:
#         token_id = -1

#     if hasattr(self, "signature_set") and token_id in self.signature_set:
#         token_color = "black"
#     else:
#         mapping = {
#             0: self.color_scheme.red_token_color,
#             1: self.color_scheme.green_token_color
#         }
#         token_color = mapping.get(value, self.color_scheme.prefix_color)

#     if show_text:
#         draw.text((x, y), token, fill=token_color, font=self.font_settings.font)
#     else:
#         draw.rectangle([(x, y), (x + token_width, y + self.font_settings.font_size)], fill=token_color)

# # 讓 Visualizer 使用你的客製 highlight 函數
# DiscreteVisualizer._highlight_single_token = _highlight_single_token
# 修改視覺化函式，只顯示紅與綠
def _highlight_single_token(self, draw, token, value, token_width, show_text, x, y, token_id=None):
    mapping = {
        0: self.color_scheme.red_token_color,
        1: self.color_scheme.green_token_color
    }
    token_color = mapping.get(value, self.color_scheme.prefix_color)

    if show_text:
        draw.text((x, y), token, fill=token_color, font=self.font_settings.font)
    else:
        draw.rectangle([(x, y), (x + token_width, y + self.font_settings.font_size)], fill=token_color)

DiscreteVisualizer._highlight_single_token = _highlight_single_token


# =============================
# 6. 建立 Visualizer
# =============================
def build_visualizer(tokenizer, signature_token_ids=None):
    visualizer = DiscreteVisualizer(
        color_scheme=ColorSchemeForDiscreteVisualization(
            prefix_color="#292FD7",
            red_token_color="#CC0000",
            green_token_color="#006400"
        ),
        font_settings=FontSettings(font_path="font/msjh.ttf", font_size=20),
        page_layout_settings=PageLayoutSettings(),
        legend_settings=DiscreteLegendSettings()
    )
    visualizer.tokenizer = tokenizer
    if signature_token_ids:
        visualizer.signature_set = signature_token_ids
    return visualizer


#  # 印token id 
# def print_token_info(data, token_ids, signature_token_ids=None):
#     print("Token ：")
#     for token_str, token_id, label in zip(data.decoded_tokens, token_ids, data.highlight_values):
#         if signature_token_ids and token_id in signature_token_ids:
#             label_str = "sigRed（Black）"
#         else:
#             label_str = {1: "Green", 0: "Red", -1: "prefix"}.get(label, "未知")

#         print(f"{token_str:<10} | Token ID: {token_id:<6} | color: {label_str}")



def visualize_and_save(data, tokenizer, signature_token_ids, sample_idx, output_dir="token_images"):
    os.makedirs(output_dir, exist_ok=True)
    visualizer = build_visualizer(tokenizer, signature_token_ids)

    # # 加入列印資訊
    # print_token_info(data, tokenizer, signature_token_ids)

    img = visualizer.visualize(
        data,
        show_text=True,
        visualize_weight=True,
        display_legend=True
    )
    # output_path = os.path.join(output_dir, f"sample_{sample_idx+1}.png")
    output_path = os.path.join(output_dir, f"test.png")
    img.save(output_path)
    print(f" 已儲存圖片：{output_path}")

# =============================
# 8. 選擇要視覺化的文字（可從資料集讀）
# =============================
data_path = 'dataset/zhtw/output_data_combined_abstracts_ai.json'
max_samples = 1

with open(data_path, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i >= max_samples:
            break
        try:
            record = json.loads(line)
            prompt = record.get("text", "")
            if prompt.strip():
                print(f"\n=== 第 {i+1} 筆 ===")
                watermarked_data = my_watermark.get_data_for_visualization(prompt)
                visualize_and_save(watermarked_data, tokenizer, signature_token_ids, i)
                
                # # watermarked_data = my_watermark.get_data_for_visualization(prompt)
                # token_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"][0].tolist()
                # print_token_info(watermarked_data, token_ids, signature_token_ids)


        except json.JSONDecodeError as e:
            print(f" 無法解析第 {i+1} 筆資料：{e}")
