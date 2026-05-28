import json
import torch
import numpy as np
import random
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from watermark.auto_watermark import AutoWatermark
from utils.transformers_config import TransformersConfig
from visualize.font_settings import FontSettings
from visualize.visualizer import DiscreteVisualizer
from visualize.legend_settings import DiscreteLegendSettings
from visualize.page_layout_settings import PageLayoutSettings
from visualize.color_scheme import ColorSchemeForDiscreteVisualization

from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

# 設定隨機種子
seed = 30
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

device = "cuda" if torch.cuda.is_available() else "cpu"


def get_transformes_config():
    model_name = 'facebook/opt-1.3b'
    print(f"使用模型: {model_name}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        local_files_only=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)

    transformers_config = TransformersConfig(
        model=model,
        tokenizer=tokenizer,
        vocab_size=50272,
        device=device,
        max_new_tokens=200,
        min_length=230,
        no_repeat_ngram_size=4,
        do_sample=True,
        eos_token_id=None,
    )
    return model, tokenizer, transformers_config


# 修改視覺化函式以支援 token_id 比對
def _highlight_single_token(self, draw, token, value, token_width, show_text, x, y, token_id=None):
    if token_id is None:
        try:
            token_ids = self.tokenizer(token, add_special_tokens=False).input_ids
            token_id = token_ids[0] if token_ids else -1
        except Exception:
            token_id = -1

    if hasattr(self, "signature_set") and token_id in self.signature_set:
        token_color = "black"
    else:
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




def build_visualizer(tokenizer, signature_set):
    legend = DiscreteLegendSettings()
    legend.show_legend = True
    legend.entries = [
        ("Watermarked Token", "#006400"),      # 綠
        ("Non-Watermarked Token", "#CC0000"),  # 紅
        ("Signature Token", "black")           # 黑
    ]

    visualizer = DiscreteVisualizer(
        color_scheme=ColorSchemeForDiscreteVisualization(
            prefix_color="#3C1BF3",
            red_token_color='#CC0000',
            green_token_color='#006400'
        ),
        font_settings=FontSettings(font_path="font/msjh.ttf", font_size=20),
        page_layout_settings=PageLayoutSettings(),
        legend_settings=legend
    )
    visualizer.tokenizer = tokenizer
    visualizer.signature_set = signature_set
    return visualizer


def standard_visualize(myWatermark, text, delta, tokenizer, signature_set, sample_idx):
    data = myWatermark.get_data_for_visualization(text)
    tokenized = tokenizer(text, add_special_tokens=False)
    tokens = tokenizer.convert_ids_to_tokens(tokenized.input_ids)
    flags = data.highlight_values

    # 顯示對齊 token 和顏色資訊
    for i, (tid, fid) in enumerate(zip(tokenized.input_ids, flags)):
        tok = tokenizer.convert_ids_to_tokens([tid])[0]
        tag = "black" if tid in signature_set else ("green" if fid == 1 else "red")
        # print(f"{i:02d}: {tok:12s} (id: {tid:<6}) flag={fid} → {tag}")

    visualizer = build_visualizer(tokenizer, signature_set)
    img = visualizer.visualize(data, show_text=True, visualize_weight=True, display_legend=True)
    img.save(f"sample_{sample_idx:03d}_combined_d{delta}.png")


if __name__ == "__main__":
    model, tokenizer, transformers_config = get_transformes_config()

    with open(r"C:\Users\user\Desktop\signature\llm-watermark-research\signature_sets\kgw_law_sig.json", "r", encoding="utf-8") as f:
        signature_list = json.load(f)
        signature_set = set(signature_list)

    delta = 2.0
    myWatermark = AutoWatermark.load(
        'KGW',
        algorithm_config='config/KGW.json',
        transformers_config=transformers_config,
        delta=delta
    )
    myWatermark.config.prefix_length = 0

    json_path = r"dataset/zhtw/test.json"
    max_samples = 5

    # with open(json_path, "r", encoding="utf-8") as f:
    #     for i, line in enumerate(f):
    #         if i >= max_samples:
    #             break
    #         try:
    #             record = json.loads(line)
    #             text = record.get("text", "")
    #             if text.strip():
    #                 print(f"\n=== 第 {i+1} 筆 ===")
    #                 standard_visualize(myWatermark, text, delta, tokenizer, signature_set, sample_idx=i + 1)
    #         except json.JSONDecodeError as e:
    #             print(f"無法解析第 {i+1} 筆資料：{e}")


    # 儲存所有 text 以產生 Word Cloud
    all_texts = []

    with open(json_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_samples:
                break
            try:
                record = json.loads(line)
                text = record.get("text", "")
                if text.strip():
                    all_texts.append(text)
                    # print(f"\n=== 第 {i+1} 筆 ===")
                    standard_visualize(myWatermark, text, delta, tokenizer, signature_set, sample_idx=i + 1)
            except json.JSONDecodeError as e:
                print(f"無法解析第 {i+1} 筆資料：{e}")

    # === 製作 Word Cloud ===
    combined_text = " ".join(all_texts)

    # 建立 WordCloud 物件
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        stopwords=set(STOPWORDS),
        font_path="font/msjh.ttf"  # 使用中文字體
    ).generate(combined_text)

    # 顯示 Word Cloud 圖像
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("wordcloud_output.png")  # 如果你想存成圖檔
    plt.show()
