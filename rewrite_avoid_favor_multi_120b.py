# ============================================================
# rewrite_avoid_favor_multi_120b.py
#
# 120B avoid/favor rewrite.
# Fix:
#   src_text ALWAYS comes from item["original"] first.
#   Never use item["rewritten"] as source because 120B rewritten may be empty
#   or contaminated by gpt-oss analysis/meta text.
# ============================================================

import argparse
import json
import os
import random
import re
import math

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

DEFAULT_MODEL_NAME = "openai/gpt-oss-120b"
DEFAULT_ALGORITHMS = ["KGW", "SWEET", "Unigram", "EXP", "SynthID"]
DEFAULT_DOMAINS = ["ai", "bio", "med", "mis", "security"]

DEFAULT_SAMPLE_MODE = "sequential"
DEFAULT_SAMPLE_SEED = 30
DEFAULT_MAX_SAMPLES = 200

TOP_K_TOKENS = 200
NUM_RETRIES = 5
MAX_NEW_TOKENS = 260

TEST_LIMIT = 3

INPUT_DIR = "outputs/wm_tokens_120b_0_200"
TOKEN_DIR = "outputs/wm_tokens_120b_0_200"
OUTPUT_DIR = "outputs/rewrite_avoid_favor_multi_120b"


BAD_META_PATTERNS = [
    r"\bwe need to\b",
    r"\blet'?s craft\b",
    r"\blet'?s rewrite\b",
    r"\bthe user wants\b",
    r"\buser\s*\n",
    r"\bassistant\s*\n",
    r"\bsystem\s*\n",
    r"\banalysis\b",
    r"\bfinal answer\b",
    r"\bmust rewrite\b",
    r"\boutput only\b",
    r"\boriginal paragraph\b",
    r"\bgiven paragraph\b",
    r"\bpreserving meaning\b",
    r"\bpreserve all numbers\b",
]

STOP_MARKERS = [
    "\nNote that",
    "\nNote:",
    "\n(Note:",
    "\nNotes:",
    "\nExplanation:",
    "\nOriginal:",
    "\nRewritten:",
    "\nRewritten paragraph:",
    "\nUser",
    "\nAssistant",
    "\nSystem",
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
        sent = sent.strip()
        normalized = re.sub(r"\s+", " ", sent.lower())
        if not normalized:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        cleaned.append(sent)

    return " ".join(cleaned).strip()


def strip_special_tokens(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"<\|[^>]*?\|>", "", text)
    return text.strip()


def extract_after_last_final_marker(text: str) -> str:
    if text is None:
        return ""

    text = text.replace("\\n", "\n").strip()
    if not text:
        return ""

    markers = [
        r"assistantfinal",
        r"assistant\s+final",
        r"<\|channel\|>\s*final\s*<\|message\|>",
        r"<\|channel\|>\s*final",
        r"<\|final\|>",
        r"final\s*<\|message\|>",
    ]

    lowered = text.lower()
    best_end = None

    for pat in markers:
        matches = list(re.finditer(pat, lowered, flags=re.I))
        if matches:
            m = matches[-1]
            if best_end is None or m.end() > best_end:
                best_end = m.end()

    if best_end is not None:
        return text[best_end:].strip()

    return text.strip()


def has_meta_contamination(text: str) -> bool:
    if not text:
        return True

    low = text.lower()
    for pat in BAD_META_PATTERNS:
        if re.search(pat, low, flags=re.I):
            return True

    return False


def remove_leading_meta_block(text: str) -> str:
    if not text:
        return ""

    text = text.strip()
    text = re.sub(r"^(analysis|assistantfinal|assistant final|final)\s*", "", text, flags=re.I).strip()

    quote_match = re.search(r'["“](.{80,})["”]', text, flags=re.S)
    if quote_match:
        candidate = quote_match.group(1).strip()
        if not has_meta_contamination(candidate):
            text = candidate

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    kept = []
    skipping = True

    for ln in lines:
        low = ln.lower()
        looks_meta = (
            low in {"user", "assistant", "system"}
            or low.startswith("we need")
            or low.startswith("let's")
            or low.startswith("the user")
            or low.startswith("need to")
            or "output only" in low
            or "preserve all" in low
            or "rewrite the following" in low
            or "given paragraph" in low
        )

        if skipping and looks_meta:
            continue

        skipping = False
        kept.append(ln)

    return " ".join(kept).strip() if kept else text.strip()


def is_bad_rewrite(text: str, original: str | None = None) -> bool:
    if not text or not text.strip():
        return True

    stripped = text.strip()

    if has_meta_contamination(stripped):
        return True

    if len(stripped.split()) < 20:
        return True

    if original:
        original_words = max(1, len(original.split()))
        rewrite_words = len(stripped.split())
        if rewrite_words < max(20, int(original_words * 0.35)):
            return True

    if re.match(r"^[\.\,\'\"\)\]\}]+", stripped):
        return True

    return False


def clean_rewritten_text(text: str, original: str | None = None) -> str:
    if text is None:
        return ""

    text = text.replace("\\n", "\n").strip()
    text = extract_after_last_final_marker(text)
    text = strip_special_tokens(text)
    text = remove_leading_meta_block(text)
    text = text.strip()

    for marker in STOP_MARKERS:
        idx = text.find(marker)
        if idx != -1:
            text = text[:idx].strip()

    note_patterns = [
        r"\n?\(?\s*Note\s*[:：-].*$",
        r"\n?\(?\s*Explanation\s*[:：-].*$",
        r"\n?\(?\s*Original\s*[:：-].*$",
        r"\n?\(?\s*Rewritten\s*(paragraph|text)?\s*[:：-].*$",
        r"\n?\(?\s*I made some minor changes.*$",
        r"\n?\(?\s*I have made some minor changes.*$",
        r"\n?\s*analysis\s+.*$",
        r"\n?\s*assistantfinal\s+.*$",
        r"\n?\s*assistant final\s+.*$",
    ]

    for pattern in note_patterns:
        text = re.sub(pattern, "", text, flags=re.I | re.S).strip()

    text = re.sub(
        r"^(Rewritten paragraph|Rewritten text|Rewritten|Answer|Output)\s*[:：-]\s*",
        "",
        text,
        flags=re.I,
    ).strip()

    text = re.sub(r"\s+", " ", text).strip()
    text = remove_repeated_sentences(text)

    if is_bad_rewrite(text, original=original):
        return ""

    return text.strip()


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


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


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
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    if max_memory:
        memory_map = {}
        for item in max_memory.split(","):
            k, v = item.split(":", 1)
            if k.strip().lower() == "cpu":
                memory_map["cpu"] = v.strip()
            else:
                memory_map[int(k.strip())] = v.strip()
        model_kwargs["max_memory"] = memory_map

    if "gpt-oss" in model_name.lower():
        if load_in_4bit:
            print("[warning] gpt-oss already uses MXFP4; ignoring --load_in_4bit")
            load_in_4bit = False
        if load_in_8bit:
            print("[warning] gpt-oss already uses MXFP4; ignoring --load_in_8bit")
            load_in_8bit = False

    if load_in_4bit and load_in_8bit:
        raise ValueError("load_in_4bit and load_in_8bit cannot both be enabled.")

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


def build_messages(text, terms, mode="avoid", strict=False):
    terms_preview = ", ".join(terms[:80])

    if mode == "avoid":
        system_msg = (
            "You are a rewriting engine. "
            "Output only the rewritten text. "
            "Do not reveal reasoning, analysis, hidden thoughts, channel names, labels, or notes. "
            "Preserve meaning, facts, tone, and approximate length."
        )
        instruction = "Avoid these words/phrases as much as possible"
    elif mode == "favor":
        system_msg = (
            "You are a rewriting engine. "
            "Output only the rewritten text. "
            "Do not reveal reasoning, analysis, hidden thoughts, channel names, labels, or notes. "
            "Do not summarize, shorten, omit, compress, or remove details. "
            "Preserve meaning, facts, tone, and approximate length."
        )
        instruction = "Use these words/phrases as much as possible when natural"
    else:
        raise ValueError(f"Unknown rewrite mode: {mode}")

    if strict:
        user_msg = (
            "Paraphrase the following text only.\n"
            "No preface. No notes. No labels. No analysis.\n"
            f"{instruction}: {terms_preview}\n\n"
            f"{text}"
        )
    else:
        user_msg = (
            "Rewrite the following text.\n\n"
            "Rules:\n"
            "1. Preserve all meaning and factual content.\n"
            "2. Do not summarize, shorten, omit, or compress the text.\n"
            "3. Keep the output fluent and natural.\n"
            "4. The rewritten text must be between 90%-110% of the input length.\n"
            f"5. {instruction}:\n"
            f"   {terms_preview}\n"
            "6. Output the rewritten text only.\n"
            "7. Do not write Note, User, Assistant, Analysis, Original, Rewritten, or Explanation.\n\n"
            f"Text:\n{text}"
        )

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]


def apply_chat_template_or_plain(tokenizer, messages):
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception as e:
            print(f"[warning] chat template failed; using plain prompt. {e}")

    return "\n\n".join([m["content"] for m in messages]) + "\n\n"


def generate_with_exp_watermark(wm, prompt_text: str, original_text: str, max_new_tokens: int = MAX_NEW_TOKENS):
    tokenizer = wm.config.generation_tokenizer
    model = wm.config.generation_model
    device = wm.config.device
    temperature = wm.config.temperature

    encoded = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=True).to(device)
    generated_ids = encoded["input_ids"]
    attention_mask = encoded.get("attention_mask", None)
    prompt_len = generated_ids.shape[1]

    cur_input_ids = generated_ids
    past_key_values = None

    for _ in range(max_new_tokens):
        with torch.inference_mode():
            outputs = model(
                input_ids=cur_input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )

        logits = outputs.logits[:, -1, :]
        past_key_values = outputs.past_key_values

        vocab_size = logits.shape[-1]
        probs = torch.softmax(logits[:, :vocab_size] / temperature, dim=-1).cpu()

        wm.utils.seed_rng(generated_ids[0])
        u = torch.rand(vocab_size, generator=wm.utils.rng).unsqueeze(0)
        next_token = wm.utils.exp_sampling(probs, u).to(device)

        next_id = next_token.view(1, 1)
        generated_ids = torch.cat([generated_ids, next_id], dim=1)
        cur_input_ids = next_id

        if attention_mask is not None:
            attention_mask = torch.cat([
                attention_mask,
                torch.ones((1, 1), device=device, dtype=attention_mask.dtype),
            ], dim=1)

        if tokenizer.eos_token_id is not None and next_id.item() == tokenizer.eos_token_id:
            break

        partial_text = tokenizer.decode(generated_ids[0, prompt_len:], skip_special_tokens=True)
        if should_stop_generation(partial_text):
            break

    completion_ids = generated_ids[0, prompt_len:]
    raw = tokenizer.decode(completion_ids, skip_special_tokens=False)
    return clean_rewritten_text(raw, original=original_text)


@torch.no_grad()
def generate_completion(
    model,
    tokenizer,
    encoded_prompt,
    prompt_len,
    original_text,
    logits_processor=None,
    gen_kwargs=None,
    max_new_tokens=None,
):
    safe_gen_kwargs = dict(gen_kwargs or {})
    safe_gen_kwargs["max_new_tokens"] = max_new_tokens or MAX_NEW_TOKENS
    safe_gen_kwargs.setdefault("do_sample", True)
    safe_gen_kwargs.setdefault("temperature", 0.7)
    safe_gen_kwargs.setdefault("top_p", 0.9)
    safe_gen_kwargs.setdefault("repetition_penalty", 1.08)
    safe_gen_kwargs.setdefault("no_repeat_ngram_size", 4)

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
    raw = tokenizer.decode(completion_ids, skip_special_tokens=False)
    cleaned = clean_rewritten_text(raw, original=original_text)

    if not cleaned:
        raw2 = tokenizer.decode(completion_ids, skip_special_tokens=True)
        cleaned = clean_rewritten_text(raw2, original=original_text)

    return cleaned


def rewrite_once_watermarked_avoid(model, tokenizer, wm, cfg, algorithm, text, avoid_terms, strict=False):
    messages = build_messages(text, avoid_terms, mode="avoid", strict=strict)
    prompt = apply_chat_template_or_plain(tokenizer, messages)

    if algorithm == "EXP":
        return generate_with_exp_watermark(
            wm,
            prompt,
            original_text=text,
            max_new_tokens=MAX_NEW_TOKENS,
        )

    gen_model = wm.config.generation_model
    gen_tokenizer = wm.config.generation_tokenizer

    encoded = gen_tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=True,
    ).to(wm.config.device)

    prompt_len = encoded["input_ids"].shape[1]

    if algorithm == "SynthID":
        reset_synthid_state(wm)

    return generate_completion(
        model=gen_model,
        tokenizer=gen_tokenizer,
        encoded_prompt=encoded,
        prompt_len=prompt_len,
        original_text=text,
        logits_processor=LogitsProcessorList([wm.logits_processor]),
        gen_kwargs=cfg.gen_kwargs,
        max_new_tokens=MAX_NEW_TOKENS,
    )


def rewrite_once_unwatermarked_favor(model, tokenizer, cfg, text, favor_terms, strict=False):
    messages = build_messages(text, favor_terms, mode="favor", strict=strict)
    prompt = apply_chat_template_or_plain(tokenizer, messages)

    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=True,
    ).to(DEVICE)

    prompt_len = encoded["input_ids"].shape[1]

    return generate_completion(
        model=model,
        tokenizer=tokenizer,
        encoded_prompt=encoded,
        prompt_len=prompt_len,
        original_text=text,
        logits_processor=None,
        gen_kwargs=cfg.gen_kwargs,
        max_new_tokens=MAX_NEW_TOKENS,
    )


def count_term_hits(text, terms):
    low = (text or "").lower()
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
        c = c or ""
        hits, hit_terms = count_term_hits(c, terms)
        scored.append({
            "text": c,
            "set_hits": hits,
            "set_hit_terms": hit_terms,
            "length_gap": abs(len(c) - src_len),
            "is_empty": not bool(c.strip()),
        })

    scored.sort(key=lambda x: (x["is_empty"], x["set_hits"], x["length_gap"]))
    return scored[0], scored


def choose_best_favor_candidate(candidates, terms, src_len):
    scored = []
    for c in candidates:
        c = c or ""
        hits, hit_terms = count_term_hits(c, terms)
        scored.append({
            "text": c,
            "set_hits": hits,
            "set_hit_terms": hit_terms,
            "length_gap": abs(len(c) - src_len),
            "is_empty": not bool(c.strip()),
        })

    scored.sort(key=lambda x: (x["is_empty"], -x["set_hits"], x["length_gap"]))
    return scored[0], scored


def get_source_text(item):
    """
    Important fix:
    Always use original first.
    Do NOT use item["rewritten"] as source because rewrite_and_collect 120B
    may produce empty/contaminated rewritten strings.
    """
    src = item.get("original") or item.get("text") or item.get("prompt") or ""
    if isinstance(src, str):
        return src.strip()
    return str(src).strip()


def run_one_setting(
    model,
    tokenizer,
    cfg,
    model_name,
    domain,
    algorithm,
    sample_mode,
    sample_seed,
    max_samples,
    test_limit=None,
    num_retries=NUM_RETRIES,
    output_tag=None,
    overwrite=False,
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

    if os.path.exists(save_path) and not overwrite:
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
        src_text = get_source_text(item)

        if not src_text:
            output.append({
                **item,
                "domain": domain,
                "algorithm": algorithm,
                "rewrite_model": model_name,
                "src_text_key": None,
                "src_text": "",
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

        avoid_retries = 1 if algorithm == "EXP" else num_retries

        for retry_idx in range(avoid_retries):
            try:
                rw_avoid = rewrite_once_watermarked_avoid(
                    model=model,
                    tokenizer=tokenizer,
                    wm=wm,
                    cfg=cfg,
                    algorithm=algorithm,
                    text=src_text,
                    avoid_terms=set_terms,
                    strict=(retry_idx > 0),
                )
            except Exception as e:
                print(f"[WARN] {domain}/{algorithm} watermarked avoid failed: {e}")
                rw_avoid = ""

            rw_avoid = clean_rewritten_text(rw_avoid, original=src_text)
            watermarked_avoid_candidates.append(rw_avoid)

        for retry_idx in range(num_retries):
            try:
                rw_favor = rewrite_once_unwatermarked_favor(
                    model=model,
                    tokenizer=tokenizer,
                    cfg=cfg,
                    text=src_text,
                    favor_terms=set_terms,
                    strict=(retry_idx > 0),
                )
            except Exception as e:
                print(f"[WARN] {domain}/{algorithm} unwatermarked favor failed: {e}")
                rw_favor = ""

            rw_favor = clean_rewritten_text(rw_favor, original=src_text)
            unwatermarked_favor_candidates.append(rw_favor)

        best_avoid, scored_avoid = choose_best_avoid_candidate(
            watermarked_avoid_candidates,
            set_terms,
            len(src_text),
        )

        best_favor, scored_favor = choose_best_favor_candidate(
            unwatermarked_favor_candidates,
            set_terms,
            len(src_text),
        )

        output.append({
            **item,
            "domain": domain,
            "algorithm": algorithm,
            "rewrite_model": model_name,
            "src_text_key": "original",
            "src_text": src_text,

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
    parser.add_argument(
        "--sample_mode",
        type=str,
        default=DEFAULT_SAMPLE_MODE,
        choices=["sequential", "random"],
    )
    parser.add_argument("--sample_seed", type=int, default=DEFAULT_SAMPLE_SEED)
    parser.add_argument("--max_samples", type=int, default=DEFAULT_MAX_SAMPLES)
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
    )
    parser.add_argument("--max_memory", type=str, default=None)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--test_limit", type=int, default=TEST_LIMIT)
    parser.add_argument("--num_retries", type=int, default=NUM_RETRIES)
    parser.add_argument("--overwrite", action="store_true")

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
                overwrite=args.overwrite,
            )

    del model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
