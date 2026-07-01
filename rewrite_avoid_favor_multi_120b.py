# ============================================================
# rewrite_avoid_favor_multi_120b.py
#
# 這支是「avoid/favor 改寫」腳本的 120B 版本。
#
# 邏輯（誰要做什麼）沿用自:
#   rewrite_avoid_favor_multi_0517.py
#     -> build_messages(avoid/favor) / rewrite_once_watermarked_avoid /
#        rewrite_once_unwatermarked_favor / candidate scoring
#
# 模型載入與文字清洗沿用自（因為 gpt-oss-120b 需要特殊處理）:
#   rewrite_and_collect_watermark_tokens_120b.py
#     -> get_transformers_config (不能用 BitsAndBytesConfig 包 gpt-oss，
#        gpt-oss 已經是 MXFP4 量化)
#     -> strip_gpt_oss_meta_prefix / clean_rewritten_text / should_stop_generation
#        (清掉 harmony 格式的 analysis / assistantfinal 等雜訊)
#     -> StopOnMarkersCriteria (生成中即時停止，避免產生一堆 Note: 補充說明)
#     -> reset_synthid_state 安全版 (避免 inference tensor 就地修改報錯)
#
# 檔名規則沿用你目前 120b 產出的檔名格式：
#   rewritten_{domain}_{algorithm}_{model_name}_{mode}_seed{seed}_n{max_samples}_wm_tokens.json
#   rewritten_{domain}_{algorithm}_{model_name}_{mode}_seed{seed}_n{max_samples}_wm_token_freq.json
# ============================================================

import argparse
import json
import os
import random
import re

import numpy as np
import torch
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    LogitsProcessorList,
    StoppingCriteria,
    StoppingCriteriaList,
)

from watermark.auto_watermark import AutoWatermark
from utils.transformers_config import TransformersConfig

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# 預設設定（可被 CLI 參數覆蓋）
# ---------------------------------------------------------------------------

DEFAULT_MODEL_NAME = "openai/gpt-oss-120b"

DEFAULT_ALGORITHMS = ["KGW", "SWEET", "Unigram", "EXP", "SynthID"]
DEFAULT_DOMAINS = ["ai", "bio", "med", "mis", "security"]

DEFAULT_SAMPLE_MODE = "sequential"    # 對應檔名裡的 sequential / random
DEFAULT_SAMPLE_SEED = 30
DEFAULT_MAX_SAMPLES = 200             # 對應檔名裡的 n200

TOP_K_TOKENS = 200
NUM_RETRIES = 5
MAX_NEW_TOKENS = 260

TEST_MODE = False
TEST_LIMIT = 3

INPUT_DIR = "outputs/wm_tokens_120b_0_200"
TOKEN_DIR = "outputs/wm_tokens_120b_0_200"       # 和輸入同一個目錄
OUTPUT_DIR = "outputs/rewrite_avoid_favor_multi_120b"


# ---------------------------------------------------------------------------
# gpt-oss 專用噪音清洗（沿用自 120b 腳本）
# ---------------------------------------------------------------------------

STOP_MARKERS = [
    "\nNote that",
    "\nNote:",
    "\n(Note:",
    "\nNotes:",
    "\nExplanation:",
    "\nOriginal:",
    "\nRewritten:",
    "\nRewritten paragraph:",
    "Note that this",
    "Note: I made",
    "(Note: I made",
    "I made some minor changes",
    "minor changes to make the text flow better",
    "I made minor changes",
    "I have rewritten",
    "Here is the rewritten",
    "Here is a rewritten",
    "The rewritten paragraph is",
]


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sanitize_model_name(model_name: str) -> str:
    return model_name.replace("/", "__").replace(" ", "_")


def remove_repeated_sentences(text: str) -> str:
    if not text:
        return ""

    sentences = re.split(r"(?<=[.!?。！？])\s+", text.strip())
    cleaned = []
    seen = set()

    for sent in sentences:
        normalized = re.sub(r"\s+", " ", sent.strip().lower())
        if not normalized:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        cleaned.append(sent.strip())

    return " ".join(cleaned).strip()


def strip_gpt_oss_meta_prefix(text: str) -> str:
    """
    gpt-oss 可能輸出 harmony 格式的可見文字，例如：
    analysisWe need to rewrite...
    assistantfinalActual answer...

    只保留 assistantfinal / final marker 之後的內容。
    注意：這個函式只在「生成完成後」清洗，不能拿來當生成中的停止條件，
    否則 gpt-oss 一開始就吐 analysis... 會導致生成馬上被截斷成空字串。
    """
    if text is None:
        return ""

    text = text.replace("\\n", "\n").strip()
    if not text:
        return ""

    patterns = [
        r"assistantfinal",
        r"assistant\s+final",
        r"<\|channel\|>\s*final\s*<\|message\|>",
        r"<\|channel\|>\s*final",
        r"<\|final\|>",
    ]

    lowered = text.lower()
    best_idx = -1
    best_pat = None

    for pat in patterns:
        matches = list(re.finditer(pat, lowered, flags=re.IGNORECASE))
        if matches:
            m = matches[-1]
            if m.start() > best_idx:
                best_idx = m.start()
                best_pat = m

    if best_pat is not None:
        text = text[best_pat.end():].strip()

    text = re.sub(
        r"^analysis\s*(we need to|let'?s|we should|i need to).*?(?=[A-Z][a-z])",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    ).strip()

    text = re.sub(r"<\|.*?\|>", "", text).strip()

    return text


def clean_rewritten_text(text: str) -> str:
    if text is None:
        return ""

    text = strip_gpt_oss_meta_prefix(text)
    text = text.replace("\\n", "\n")
    text = text.strip()

    for marker in STOP_MARKERS:
        idx = text.find(marker)
        if idx != -1:
            text = text[:idx].strip()

    note_patterns = [
        r"\n?\(?\s*Note\s*[:：-].*$",
        r"\n?\(?\s*Explanation\s*[:：-].*$",
        r"\n?\(?\s*Original\s*[:：-].*$",
        r"\n?\(?\s*Rewritten\s*(paragraph)?\s*[:：-].*$",
        r"\n?\(?\s*I made some minor changes.*$",
        r"\n?\(?\s*I have made some minor changes.*$",
        r"\n?\s*analysis\s+.*$",
        r"\n?\s*assistantfinal\s+.*$",
        r"\n?\s*assistant final\s+.*$",
    ]

    for pattern in note_patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.DOTALL).strip()

    text = re.sub(
        r"^(Rewritten paragraph|Rewritten|Answer|Output)\s*[:：-]\s*",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()

    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")

    return remove_repeated_sentences(text).strip()


def should_stop_generation(decoded_text: str) -> bool:
    if not decoded_text:
        return False

    lowered = decoded_text.lower().lstrip()

    dangerous_markers = [
        "\nnote:",
        "\nexplanation:",
        "\noriginal:",
        "\nrewritten:",
    ]

    return any(marker in lowered for marker in dangerous_markers)


class StopOnMarkersCriteria(StoppingCriteria):
    def __init__(self, tokenizer, prompt_len: int):
        self.tokenizer = tokenizer
        self.prompt_len = prompt_len

    def __call__(self, input_ids, scores, **kwargs):
        generated_ids = input_ids[0, self.prompt_len:]
        decoded = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return should_stop_generation(decoded)


# ---------------------------------------------------------------------------
# File I/O helpers
# ---------------------------------------------------------------------------

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Model loading（120b / gpt-oss 專用，沿用自 120b 收集腳本）
# ---------------------------------------------------------------------------

def load_model_tokenizer_and_cfg(
    model_name: str,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
    torch_dtype: str = "bfloat16",
    max_memory: str | None = None,
):
    print(f"Loading model: {model_name}")

    if torch_dtype == "float16":
        dtype = torch.float16
    elif torch_dtype == "float32":
        dtype = torch.float32
    else:
        dtype = torch.bfloat16

    model_kwargs = dict(
        device_map="auto",
        dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    if max_memory:
        # 範例："0:78GiB,cpu:200GiB"
        memory_map = {}
        for item in max_memory.split(","):
            k, v = item.split(":", 1)
            if k.strip().lower() == "cpu":
                memory_map["cpu"] = v.strip()
            else:
                memory_map[int(k.strip())] = v.strip()
        model_kwargs["max_memory"] = memory_map

    # openai/gpt-oss 系列已經內建 MXFP4 量化，不能再疊 BitsAndBytesConfig
    if "gpt-oss" in model_name.lower():
        if load_in_4bit:
            print("[warning] gpt-oss 已使用 MXFP4，忽略 --load_in_4bit")
            load_in_4bit = False
        if load_in_8bit:
            print("[warning] gpt-oss 已使用 MXFP4，忽略 --load_in_8bit")
            load_in_8bit = False

    if load_in_4bit and load_in_8bit:
        raise ValueError("load_in_4bit 和 load_in_8bit 只能擇一。")

    if load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )

    if load_in_8bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    if model.get_output_embeddings() is not None:
        real_vocab_size = model.get_output_embeddings().out_features
    else:
        real_vocab_size = model.config.vocab_size

    # avoid/favor 改寫需要抽樣多樣性（NUM_RETRIES 次取最好的候選），
    # 所以這裡用 do_sample=True，跟 120b 收集腳本（do_sample=False）不同。
    cfg = TransformersConfig(
        model=model,
        tokenizer=tokenizer,
        vocab_size=real_vocab_size,
        device=DEVICE,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.08,
        no_repeat_ngram_size=4,
    )

    return model, tokenizer, cfg


def reset_synthid_state(wm):
    """
    安全版 reset：不用 fill_() 這種就地操作，改成建立新的零張量，
    避免 'Inplace update to inference tensor outside InferenceMode' 錯誤。
    """
    if not hasattr(wm, "logits_processor"):
        return

    lp = wm.logits_processor

    if not hasattr(lp, "state") or lp.state is None:
        return

    state = lp.state

    if "num_calls" in state:
        state["num_calls"] = 0

    for key in ("context", "context_history"):
        if key in state and torch.is_tensor(state[key]):
            old = state[key]
            state[key] = torch.zeros(old.shape, dtype=old.dtype, device=old.device)


# ---------------------------------------------------------------------------
# Path helpers（依照你 120b 產出的檔名規則）
# ---------------------------------------------------------------------------

def get_input_text_json(domain, algorithm, model_name, sample_mode, sample_seed, max_samples):
    safe = sanitize_model_name(model_name)
    return os.path.join(
        INPUT_DIR,
        f"rewritten_{domain}_{algorithm}_{safe}_{sample_mode}_seed{sample_seed}_n{max_samples}_wm_tokens.json",
    )


def get_token_set_json(domain, algorithm, model_name, sample_mode, sample_seed, max_samples):
    safe = sanitize_model_name(model_name)
    return os.path.join(
        TOKEN_DIR,
        f"rewritten_{domain}_{algorithm}_{safe}_{sample_mode}_seed{sample_seed}_n{max_samples}_wm_token_freq.json",
    )


def get_output_json(domain, algorithm, model_name, sample_mode, sample_seed, max_samples, tag=None):
    safe = sanitize_model_name(model_name)
    suffix = f"_{tag}" if tag else ""
    return os.path.join(
        OUTPUT_DIR,
        f"rewrite_avoid_favor_{domain}_{algorithm}_{safe}_{sample_mode}_seed{sample_seed}_n{max_samples}_top{TOP_K_TOKENS}{suffix}.json",
    )


def load_top_token_ids(path, top_k=None):
    data = load_json(path)
    if top_k is not None:
        data = data[:top_k]
    return [x["token_id"] for x in data]


# ---------------------------------------------------------------------------
# Token / term helpers（沿用自 0517 avoid/favor 腳本）
# ---------------------------------------------------------------------------

def normalize_term(s: str) -> str:
    s = s.replace("\n", " ").strip()
    s = s.replace("Ġ", " ").replace("▁", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def token_ids_to_terms(tokenizer, token_ids, min_len=2):
    terms = []

    for tid in token_ids:
        txt = tokenizer.decode([tid], skip_special_tokens=True)
        txt = normalize_term(txt)

        if not txt:
            continue
        if len(txt) < min_len:
            continue
        if re.fullmatch(r"[^\wA-Za-z]+", txt):
            continue

        terms.append(txt)

    seen = set()
    uniq = []
    for t in terms:
        if t not in seen:
            seen.add(t)
            uniq.append(t)

    return uniq


# ---------------------------------------------------------------------------
# Prompt builder（沿用 avoid/favor 邏輯，加上 gpt-oss 的 "Reasoning: low" 提示
# 以減少 harmony analysis channel 雜訊）
# ---------------------------------------------------------------------------

def build_messages(text, terms, mode="avoid"):
    terms_preview = ", ".join(terms[:80])

    reasoning_hint = "Reasoning: low\n"

    if mode == "avoid":
        system_msg = (
            reasoning_hint +
            "You are a rewriting engine. "
            "The very first token of your response MUST be the first word of the rewritten text. "
            "Output ONLY the rewritten text — nothing before it and nothing after it. "
            "Do not reveal reasoning, analysis, hidden thoughts, channel names, labels, or notes. "
            "NEVER include preambles such as 'Here is', 'The rewritten text', 'I rewrote', "
            "'Note:', 'Below is', or any sentence describing what you did. "
            "NEVER append notes, comments, explanations, bullet points, or lists of changes. "
            "NEVER use double newlines (\\n\\n) — output a single continuous block of text. "
            "Preserve the original meaning, facts, tone, and approximate length. "
            "Stop immediately after the last sentence of the rewritten text."
        )
        instruction = "Avoid these favored words/phrases as much as possible"

    elif mode == "favor":
        system_msg = (
            reasoning_hint +
            "You are a rewriting engine. "
            "The very first token of your response MUST be the first word of the rewritten text. "
            "Output ONLY the rewritten text — nothing before it and nothing after it. "
            "Do not reveal reasoning, analysis, hidden thoughts, channel names, labels, or notes. "
            "NEVER include preambles such as 'Here is', 'The rewritten text', 'I rewrote', "
            "'Note:', 'Below is', 'The original text has been rewritten', "
            "or any sentence describing what you did. "
            "NEVER append notes, comments, explanations, bullet points, or lists of changes. "
            "NEVER use double newlines (\\n\\n) — output a single continuous block of text. "
            "Do NOT summarize, shorten, omit, compress, or remove details. "
            "The rewritten text must be approximately the same length as the input. "
            "Use the listed favored words or phrases whenever they fit naturally. "
            "Stop immediately after the last sentence of the rewritten text."
        )
        instruction = "Use these favored words/phrases as much as possible when natural"

    else:
        raise ValueError(f"Unknown rewrite mode: {mode}")

    user_msg = (
        "Rewrite the following text.\n\n"
        "Rules:\n"
        "1. Preserve all meaning and factual content.\n"
        "2. Do not summarize, shorten, omit, or compress the text.\n"
        "3. Keep the output fluent and natural.\n"
        "4. The rewritten text must be between 90%–110% of the input length.\n"
        f"5. {instruction}:\n"
        f"   {terms_preview}\n"
        "6. YOUR RESPONSE = REWRITTEN TEXT ONLY.\n"
        "   - Do NOT write 'Note:', 'Note that', 'I rewrote', 'Here is', "
        "'The original text has been rewritten', 'Below is', or anything\n"
        "     that is not part of the rewritten text itself.\n"
        "   - Do NOT use double newlines (\\n\\n) anywhere in your response.\n"
        "   - Do NOT repeat sentences.\n"
        "   - Do NOT add any text after the final sentence.\n"
        "   - Begin your response with the first word of the rewritten text immediately.\n\n"
        f"Original text:\n{text}"
    )

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]


# ---------------------------------------------------------------------------
# Generation helpers（用 120b 版的 stopping criteria + gpt-oss 清洗）
# ---------------------------------------------------------------------------

def generate_with_exp_watermark(wm, prompt_text: str, max_new_tokens: int = MAX_NEW_TOKENS):
    tokenizer = wm.config.generation_tokenizer
    model = wm.config.generation_model
    device = wm.config.device
    temperature = wm.config.temperature

    encoded = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=True).to(device)
    prefix_ids = encoded["input_ids"]
    attention_mask = encoded.get("attention_mask", None)
    prompt_len = prefix_ids.shape[1]

    for _ in range(max_new_tokens):
        with torch.inference_mode():
            if attention_mask is not None:
                logits = model(prefix_ids, attention_mask=attention_mask).logits[:, -1, :]
            else:
                logits = model(prefix_ids).logits[:, -1, :]

        vocab_size = logits.shape[-1]
        probs = torch.softmax(logits[:, :vocab_size] / temperature, dim=-1).cpu()

        wm.utils.seed_rng(prefix_ids[0])
        u = torch.rand(vocab_size, generator=wm.utils.rng).unsqueeze(0)
        next_token = wm.utils.exp_sampling(probs, u).to(device)

        next_id = next_token.view(1, 1)
        prefix_ids = torch.cat([prefix_ids, next_id], dim=1)

        if attention_mask is not None:
            attention_mask = torch.cat([
                attention_mask,
                torch.ones((1, 1), device=device, dtype=attention_mask.dtype),
            ], dim=1)

        if tokenizer.eos_token_id is not None and next_id.item() == tokenizer.eos_token_id:
            break

        partial_text = tokenizer.decode(prefix_ids[0, prompt_len:], skip_special_tokens=True)
        if should_stop_generation(partial_text):
            break

    completion_ids = prefix_ids[0, prompt_len:]
    raw = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
    return clean_rewritten_text(raw)


@torch.no_grad()
def generate_completion(
    model,
    tokenizer,
    encoded_prompt,
    prompt_len,
    logits_processor=None,
    gen_kwargs=None,
    max_new_tokens=None,
):
    safe_gen_kwargs = dict(gen_kwargs or {})
    if max_new_tokens is not None:
        safe_gen_kwargs["max_new_tokens"] = max_new_tokens
    safe_gen_kwargs.setdefault("max_new_tokens", MAX_NEW_TOKENS)
    safe_gen_kwargs.setdefault("do_sample", True)

    if tokenizer.eos_token_id is not None:
        safe_gen_kwargs.setdefault("eos_token_id", tokenizer.eos_token_id)
    if tokenizer.pad_token_id is not None:
        safe_gen_kwargs.setdefault("pad_token_id", tokenizer.pad_token_id)

    stopping_criteria = StoppingCriteriaList([
        StopOnMarkersCriteria(tokenizer, prompt_len)
    ])

    with torch.inference_mode():
        output_ids = model.generate(
            **encoded_prompt,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            **safe_gen_kwargs,
        )[0]

    completion_ids = output_ids[prompt_len:]
    raw = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
    return clean_rewritten_text(raw)


# ---------------------------------------------------------------------------
# Rewrite functions（沿用 0517 avoid/favor 邏輯）
# ---------------------------------------------------------------------------

def rewrite_once_watermarked_avoid(model, tokenizer, wm, cfg, algorithm, text, avoid_terms):
    """Version A：保留 watermark，盡量避開 token set terms。"""
    messages = build_messages(text, avoid_terms, mode="avoid")
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )

    if algorithm == "EXP":
        return generate_with_exp_watermark(wm, prompt, max_new_tokens=MAX_NEW_TOKENS)

    gen_model = wm.config.generation_model
    gen_tokenizer = wm.config.generation_tokenizer

    encoded = gen_tokenizer(
        prompt, return_tensors="pt", add_special_tokens=True,
    ).to(wm.config.device)

    prompt_len = encoded["input_ids"].shape[1]

    if algorithm == "SynthID":
        reset_synthid_state(wm)

    return generate_completion(
        model=gen_model,
        tokenizer=gen_tokenizer,
        encoded_prompt=encoded,
        prompt_len=prompt_len,
        logits_processor=LogitsProcessorList([wm.logits_processor]),
        gen_kwargs=cfg.gen_kwargs,
        max_new_tokens=MAX_NEW_TOKENS,
    )


def rewrite_once_unwatermarked_favor(model, tokenizer, cfg, text, favor_terms):
    """Version B：不加 watermark logits processor，盡量使用 token set terms。"""
    messages = build_messages(text, favor_terms, mode="favor")
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )

    encoded = tokenizer(
        prompt, return_tensors="pt", add_special_tokens=True,
    ).to(DEVICE)

    prompt_len = encoded["input_ids"].shape[1]

    return generate_completion(
        model=model,
        tokenizer=tokenizer,
        encoded_prompt=encoded,
        prompt_len=prompt_len,
        logits_processor=None,
        gen_kwargs=cfg.gen_kwargs,
        max_new_tokens=MAX_NEW_TOKENS,
    )


# ---------------------------------------------------------------------------
# Candidate scoring（沿用 0517 avoid/favor 腳本，原封不動）
# ---------------------------------------------------------------------------

def count_term_hits(text, terms):
    low = text.lower()
    total_hits = 0
    hit_terms = []

    for t in terms:
        tt = t.lower().strip()
        if not tt:
            continue
        count = low.count(tt)
        if count > 0:
            total_hits += count
            hit_terms.append([t, count])

    return total_hits, hit_terms


def choose_best_avoid_candidate(candidates, terms, src_len):
    scored = []
    for c in candidates:
        hits, hit_terms = count_term_hits(c, terms)
        scored.append({
            "text": c,
            "set_hits": hits,
            "set_hit_terms": hit_terms,
            "length_gap": abs(len(c) - src_len),
        })
    scored.sort(key=lambda x: (x["set_hits"], x["length_gap"]))
    return scored[0], scored


def choose_best_favor_candidate(candidates, terms, src_len):
    scored = []
    for c in candidates:
        hits, hit_terms = count_term_hits(c, terms)
        scored.append({
            "text": c,
            "set_hits": hits,
            "set_hit_terms": hit_terms,
            "length_gap": abs(len(c) - src_len),
        })
    scored.sort(key=lambda x: (-x["set_hits"], x["length_gap"]))
    return scored[0], scored


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_one_setting(
    model, tokenizer, cfg, model_name, domain, algorithm,
    sample_mode, sample_seed, max_samples,
    test_limit=None, num_retries=NUM_RETRIES, output_tag=None,
):
    input_path = get_input_text_json(domain, algorithm, model_name, sample_mode, sample_seed, max_samples)
    token_path = get_token_set_json(domain, algorithm, model_name, sample_mode, sample_seed, max_samples)
    save_path = get_output_json(domain, algorithm, model_name, sample_mode, sample_seed, max_samples, tag=output_tag)

    if not os.path.exists(input_path):
        print(f"[SKIP] missing input file: {input_path}")
        return

    if not os.path.exists(token_path):
        print(f"[SKIP] missing token file: {token_path}")
        return

    if os.path.exists(save_path):
        print(f"[SKIP] already exists: {save_path}")
        return

    wm = AutoWatermark.load(algorithm, f"config/{algorithm}.json", cfg)

    print(f"\n=== Running domain={domain}, algorithm={algorithm} ===")
    print(f"Input : {input_path}")
    print(f"Token : {token_path}")
    print(f"Output: {save_path}")

    data = load_json(input_path)

    if test_limit is not None:
        data = data[:test_limit]
        print(f"[TEST MODE] Only running first {len(data)} samples.")

    token_ids = load_top_token_ids(token_path, TOP_K_TOKENS)
    set_terms = token_ids_to_terms(tokenizer, token_ids)

    output = []

    for item in tqdm(data, desc=f"{domain}/{algorithm}"):
        src_text = item.get("rewritten") or item.get("plain") or item.get("text") or item.get("original") or ""

        if not src_text:
            output.append({
                **item,
                "domain": domain,
                "algorithm": algorithm,
                "rewrite_model": model_name,
                "rewrite_watermarked_avoid_set": "",
                "watermarked_avoid_set_hits": None,
                "watermarked_avoid_set_hit_terms": [],
                "watermarked_avoid_candidates": [],
                "rewrite_unwatermarked_favor_set": "",
                "unwatermarked_favor_set_hits": None,
                "unwatermarked_favor_set_hit_terms": [],
                "unwatermarked_favor_candidates": [],
            })
            continue

        watermarked_avoid_candidates = []
        unwatermarked_favor_candidates = []

        for _ in range(num_retries):
            try:
                rw_avoid = rewrite_once_watermarked_avoid(
                    model=model, tokenizer=tokenizer, wm=wm, cfg=cfg,
                    algorithm=algorithm, text=src_text, avoid_terms=set_terms,
                )
            except Exception as e:
                print(f"[WARN] {domain}/{algorithm} watermarked avoid failed: {e}")
                rw_avoid = ""

            watermarked_avoid_candidates.append(rw_avoid)

            try:
                rw_favor = rewrite_once_unwatermarked_favor(
                    model=model, tokenizer=tokenizer, cfg=cfg,
                    text=src_text, favor_terms=set_terms,
                )
            except Exception as e:
                print(f"[WARN] {domain}/{algorithm} unwatermarked favor failed: {e}")
                rw_favor = ""

            unwatermarked_favor_candidates.append(rw_favor)

        best_avoid, scored_avoid = choose_best_avoid_candidate(
            watermarked_avoid_candidates, set_terms, len(src_text),
        )
        best_favor, scored_favor = choose_best_favor_candidate(
            unwatermarked_favor_candidates, set_terms, len(src_text),
        )

        output.append({
            **item,
            "domain": domain,
            "algorithm": algorithm,
            "rewrite_model": model_name,

            "rewrite_watermarked_avoid_set": best_avoid["text"],
            "watermarked_avoid_set_hits": best_avoid["set_hits"],
            "watermarked_avoid_set_hit_terms": best_avoid["set_hit_terms"],
            "watermarked_avoid_candidates": scored_avoid,

            "rewrite_unwatermarked_favor_set": best_favor["text"],
            "unwatermarked_favor_set_hits": best_favor["set_hits"],
            "unwatermarked_favor_set_hit_terms": best_favor["set_hit_terms"],
            "unwatermarked_favor_candidates": scored_favor,
        })

    save_json(output, save_path)
    print(f"[DONE] Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--domains", type=str, nargs="+", default=DEFAULT_DOMAINS)
    parser.add_argument("--algorithms", type=str, nargs="+", default=DEFAULT_ALGORITHMS)
    parser.add_argument("--sample_mode", type=str, default=DEFAULT_SAMPLE_MODE,
                         choices=["sequential", "random"])
    parser.add_argument("--sample_seed", type=int, default=DEFAULT_SAMPLE_SEED)
    parser.add_argument("--max_samples", type=int, default=DEFAULT_MAX_SAMPLES)
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16",
                         choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--max_memory", type=str, default=None)
    parser.add_argument("--test", action="store_true",
                         help="小測試模式：每個 domain/algorithm 只跑前 --test_limit 筆，"
                              "輸出檔名會加上 _test 後綴，不會覆蓋正式輸出。")
    parser.add_argument("--test_limit", type=int, default=TEST_LIMIT,
                         help="測試模式下每個 domain/algorithm 跑幾筆（預設 3）")
    parser.add_argument("--num_retries", type=int, default=NUM_RETRIES,
                         help="每筆資料 avoid/favor 各取樣幾次挑最佳候選（測試時可調小，例如 2）")
    args = parser.parse_args()

    set_seed(args.sample_seed)

    model, tokenizer, cfg = load_model_tokenizer_and_cfg(
        model_name=args.model_name,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        torch_dtype=args.torch_dtype,
        max_memory=args.max_memory,
    )

    test_limit = args.test_limit if args.test else None
    output_tag = "test" if args.test else None

    if args.test:
        print(f"[TEST MODE] test_limit={args.test_limit}, num_retries={args.num_retries}")

    for domain in args.domains:
        for algorithm in args.algorithms:
            run_one_setting(
                model=model,
                tokenizer=tokenizer,
                cfg=cfg,
                model_name=args.model_name,
                domain=domain,
                algorithm=algorithm,
                sample_mode=args.sample_mode,
                sample_seed=args.sample_seed,
                max_samples=args.max_samples,
                test_limit=test_limit,
                num_retries=args.num_retries,
                output_tag=output_tag,
            )

    del model
    del tokenizer
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()