# ============================================================
# rewrite_and_collect_watermark_tokens_120b.py
#
# 120B version.
# Fixes:
#   1. Decode gpt-oss output with special tokens first, then extract final answer.
#   2. Remove analysis/User/prompt/meta contamination from rewritten.
#   3. Validate rewritten quality; retry with safer prompt when output is bad.
#   4. If rewritten/plain is still bad, keep it empty instead of saving prompt garbage.
#   5. Keep original in every record so downstream scripts can always use original.
# ============================================================

import os
import re
import json
import random
import argparse
import torch
import numpy as np
from collections import Counter

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    LogitsProcessorList,
    StoppingCriteria,
    StoppingCriteriaList,
)

from evaluation.dataset import C4Dataset, ZHTWC4Dataset
from watermark.auto_watermark import AutoWatermark
from utils.transformers_config import TransformersConfig


DEFAULT_SEED = 30
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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


def get_dynamic_max_new_tokens(text: str, max_cap: int = 300) -> int:
    if not text:
        return min(160, max_cap)
    word_count = len(text.split())
    return min(max_cap, max(180, int(word_count * 1.45)))


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

    best_end = None
    lowered = text.lower()

    for pat in markers:
        matches = list(re.finditer(pat, lowered, flags=re.IGNORECASE))
        if matches:
            m = matches[-1]
            if best_end is None or m.end() > best_end:
                best_end = m.end()

    if best_end is not None:
        return text[best_end:].strip()

    return text.strip()


def remove_leading_meta_block(text: str) -> str:
    if not text:
        return ""

    text = text.strip()

    # Remove common gpt-oss visible channel/meta prefixes.
    text = re.sub(r"^(analysis|assistantfinal|assistant final|final)\s*", "", text, flags=re.I).strip()

    # If a meta paragraph appears before a quoted rewrite, keep the quoted paragraph.
    quote_match = re.search(r'["“](.{80,})["”]', text, flags=re.S)
    if quote_match:
        candidate = quote_match.group(1).strip()
        if not has_meta_contamination(candidate):
            text = candidate

    # Remove obvious leading reasoning paragraphs until a sentence that looks like content.
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


def has_meta_contamination(text: str) -> bool:
    if not text:
        return True

    low = text.lower()
    for pat in BAD_META_PATTERNS:
        if re.search(pat, low, flags=re.I):
            return True

    return False


def is_bad_rewrite(text: str, original: str | None = None) -> bool:
    if not text or not text.strip():
        return True

    stripped = text.strip()
    low = stripped.lower()

    if has_meta_contamination(stripped):
        return True

    if len(stripped.split()) < 20:
        return True

    if original:
        original_words = max(1, len(original.split()))
        rewrite_words = len(stripped.split())
        if rewrite_words < max(20, int(original_words * 0.35)):
            return True

    # Bad output often starts with punctuation/meta fragments.
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
        text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.DOTALL).strip()

    text = re.sub(
        r"^(Rewritten paragraph|Rewritten text|Rewritten|Answer|Output)\s*[:：-]\s*",
        "",
        text,
        flags=re.IGNORECASE,
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


def get_transformers_config(
    model_name: str,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
    torch_dtype: str = "bfloat16",
    max_memory: str | None = None,
):
    print(f"Using model: {model_name}")

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

    # gpt-oss already uses MXFP4. Do not wrap again with BitsAndBytesConfig.
    if "gpt-oss" in model_name.lower():
        if load_in_4bit:
            print("[warning] gpt-oss already uses MXFP4; ignoring --load_in_4bit")
            load_in_4bit = False
        if load_in_8bit:
            print("[warning] gpt-oss already uses MXFP4; ignoring --load_in_8bit")
            load_in_8bit = False

    if load_in_4bit and load_in_8bit:
        raise ValueError("Choose only one: load_in_4bit or load_in_8bit.")

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

    print("tokenizer.vocab_size =", tokenizer.vocab_size)
    print("len(tokenizer) =", len(tokenizer))
    print("model.config.vocab_size =", model.config.vocab_size)
    print("real_vocab_size =", real_vocab_size)

    return TransformersConfig(
        model=model,
        tokenizer=tokenizer,
        vocab_size=real_vocab_size,
        device=DEVICE,
        max_new_tokens=300,
        do_sample=False,
        repetition_penalty=1.08,
        no_repeat_ngram_size=4,
    )


def build_rewrite_prompt(tokenizer, text: str, use_chat_template: bool = True, strict: bool = False) -> str:
    system_instruction = (
        "You are a text rewriting engine. "
        "Return only the rewritten paragraph. "
        "Do not show reasoning, analysis, prompt text, labels, notes, or explanations."
    )

    user_instruction = (
        "Rewrite the paragraph below in your own words while preserving the meaning.\n"
        "Keep all numbers, percentages, dataset names, and technical terms exactly.\n"
        "Do not add new information.\n"
        "Do not summarize.\n"
        "Do not include labels such as User, Assistant, Analysis, Original, Rewritten, or Note.\n"
        "Output only one rewritten paragraph.\n\n"
        f"Paragraph:\n{text}"
    )

    if strict:
        user_instruction = (
            "Paraphrase this paragraph only.\n"
            "Return the paraphrased paragraph directly.\n"
            "No preface. No notes. No analysis. No labels.\n\n"
            f"{text}"
        )

    if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_instruction},
        ]
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception as e:
            print(f"[warning] chat template failed; using plain prompt. {e}")

    return system_instruction + "\n\n" + user_instruction + "\n\n"


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


def generate_completion(
    model,
    tokenizer,
    encoded_prompt,
    prompt_len,
    original_text=None,
    logits_processor=None,
    gen_kwargs=None,
    max_new_tokens=None,
):
    safe_gen_kwargs = dict(gen_kwargs or {})
    safe_gen_kwargs["max_new_tokens"] = max_new_tokens or 300
    safe_gen_kwargs.setdefault("do_sample", False)
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

    # Keep special tokens first so final/channel markers remain visible for cleaning.
    decoded_raw = tokenizer.decode(completion_ids, skip_special_tokens=False)
    cleaned = clean_rewritten_text(decoded_raw, original=original_text)

    if not cleaned:
        decoded_clean = tokenizer.decode(completion_ids, skip_special_tokens=True)
        cleaned = clean_rewritten_text(decoded_clean, original=original_text)

    if not cleaned:
        print("\n[DEBUG bad/empty generation]")
        print("completion_token_count =", len(completion_ids))
        print("decoded_raw[:1000] =", repr(decoded_raw[:1000]))

    return cleaned


def generate_with_exp_watermark(wm, prompt_text: str, original_text: str | None = None, max_new_tokens: int = 300):
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
    decoded_raw = tokenizer.decode(completion_ids, skip_special_tokens=False)
    cleaned = clean_rewritten_text(decoded_raw, original=original_text)

    if not cleaned:
        decoded_clean = tokenizer.decode(completion_ids, skip_special_tokens=True)
        cleaned = clean_rewritten_text(decoded_clean, original=original_text)

    if not cleaned:
        print("\n[DEBUG bad/empty EXP generation]")
        print("completion_token_count =", len(completion_ids))
        print("decoded_raw[:1000] =", repr(decoded_raw[:1000]))

    return cleaned


def collect_watermark_injected_tokens(wm, prompt_text, max_steps=300):
    tokenizer = wm.config.generation_tokenizer
    model = wm.config.generation_model
    device = wm.config.device
    algo = wm.config.algorithm_name

    encoded = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=True).to(device)
    prefix_ids = encoded["input_ids"]
    attention_mask = encoded.get("attention_mask", None)
    injected = []
    prompt_len = prefix_ids.shape[1]

    if algo == "SynthID":
        reset_synthid_state(wm)

    generated_len = 0

    for step in range(max_steps):
        generated_len = step + 1

        with torch.inference_mode():
            if attention_mask is not None:
                base_logits = model(prefix_ids, attention_mask=attention_mask).logits[:, -1, :]
            else:
                base_logits = model(prefix_ids).logits[:, -1, :]

        base_next = torch.argmax(base_logits, dim=-1).item()

        if algo == "EXP":
            temperature = wm.config.temperature
            vocab_size = base_logits.shape[-1]
            probs = torch.softmax(base_logits[:, :vocab_size] / temperature, dim=-1).cpu()
            wm.utils.seed_rng(prefix_ids[0])
            u = torch.rand(vocab_size, generator=wm.utils.rng).unsqueeze(0)
            token = wm.utils.exp_sampling(probs, u).to(device)
            wm_next = int(token.item())
        else:
            with torch.inference_mode():
                wm_logits = wm.logits_processor(prefix_ids, base_logits.clone())
            wm_next = torch.argmax(wm_logits, dim=-1).item()

        if wm_next != base_next:
            info = {
                "step": step,
                "base_next": base_next,
                "base_next_str": tokenizer.decode([base_next]),
                "wm_next": wm_next,
                "wm_next_str": tokenizer.decode([wm_next]),
                "algorithm": algo,
            }

            if algo == "SWEET":
                probs = torch.softmax(base_logits, dim=-1)
                entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=-1).item()
                info["entropy"] = entropy
                if entropy > wm.config.entropy_threshold:
                    injected.append(info)
            else:
                injected.append(info)

        next_id = torch.tensor([[wm_next]], device=device)
        prefix_ids = torch.cat([prefix_ids, next_id], dim=1)

        if attention_mask is not None:
            attention_mask = torch.cat([
                attention_mask,
                torch.ones((1, 1), device=device, dtype=attention_mask.dtype),
            ], dim=1)

        if tokenizer.eos_token_id is not None and wm_next == tokenizer.eos_token_id:
            break

        partial_text = tokenizer.decode(prefix_ids[0, prompt_len:], skip_special_tokens=True)
        if should_stop_generation(partial_text):
            break

    return injected, {"generated_len": generated_len}


def load_prompt_jsonl(dataset_path, max_samples=None):
    data = []

    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            item = json.loads(line)

            if isinstance(item, dict) and "prompt" in item:
                data.append({"prompt": item["prompt"]})

            if max_samples is not None and len(data) >= max_samples:
                break

    return data


def load_dataset_all(dataset_path, tokenizer=None):
    if dataset_path.endswith(".jsonl"):
        return load_prompt_jsonl(dataset_path, max_samples=None)

    if "zhtw" in dataset_path.lower():
        return ZHTWC4Dataset(dataset_path, tokenizer=tokenizer, max_samples=None)

    return C4Dataset(dataset_path, max_samples=None)


def choose_indices(
    total: int,
    max_samples: int,
    sample_seed: int,
    start_index: int = 0,
    random_sample: bool = True,
):
    if max_samples is None or max_samples >= total:
        return list(range(total))

    if random_sample:
        rng = random.Random(sample_seed)
        return sorted(rng.sample(range(total), max_samples))

    start = min(start_index, total)
    end = min(start_index + max_samples, total)
    return list(range(start, end))


def get_plain_cache_path(output_dir, domain, model_name, sample_seed, max_samples, random_sample):
    safe_model_name = sanitize_model_name(model_name)
    cache_dir = os.path.join(output_dir, "plain_cache")
    os.makedirs(cache_dir, exist_ok=True)
    mode = "random" if random_sample else "sequential"

    return os.path.join(
        cache_dir,
        f"plain_cache_{domain}_{safe_model_name}_{mode}_seed{sample_seed}_n{max_samples}.json",
    )


def load_plain_cache(cache_path):
    if not os.path.exists(cache_path):
        return {}

    with open(cache_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return {int(k): v for k, v in data.items()}


def save_plain_cache(cache, cache_path):
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in cache.items()}, f, ensure_ascii=False, indent=2)


def generate_rewrite_with_retry(
    algorithm,
    wm,
    cfg,
    original_text,
    max_new_tokens,
    use_chat_template=True,
    watermarked=True,
    max_attempts=2,
):
    tokenizer = cfg.tokenizer
    model = cfg.model

    for attempt in range(max_attempts):
        prompt = build_rewrite_prompt(
            tokenizer,
            original_text,
            use_chat_template=use_chat_template,
            strict=(attempt > 0),
        )

        encoded = tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=True,
        ).to(DEVICE)

        prompt_len = encoded["input_ids"].shape[1]

        if watermarked:
            if algorithm == "EXP":
                rewritten = generate_with_exp_watermark(
                    wm,
                    prompt,
                    original_text=original_text,
                    max_new_tokens=max_new_tokens,
                )
            else:
                if algorithm == "SynthID":
                    reset_synthid_state(wm)

                rewritten = generate_completion(
                    model=model,
                    tokenizer=tokenizer,
                    encoded_prompt=encoded,
                    prompt_len=prompt_len,
                    original_text=original_text,
                    logits_processor=LogitsProcessorList([wm.logits_processor]),
                    gen_kwargs=cfg.gen_kwargs,
                    max_new_tokens=max_new_tokens,
                )
        else:
            rewritten = generate_completion(
                model=model,
                tokenizer=tokenizer,
                encoded_prompt=encoded,
                prompt_len=prompt_len,
                original_text=original_text,
                logits_processor=None,
                gen_kwargs=cfg.gen_kwargs,
                max_new_tokens=max_new_tokens,
            )

        rewritten = clean_rewritten_text(rewritten, original=original_text)

        if rewritten:
            return rewritten

        print(f"[warning] bad rewrite attempt={attempt + 1}, retrying...")

    return ""


def rewrite_and_collect_120b(
    algorithm,
    dataset_path,
    max_samples,
    domain,
    model_name,
    output_dir="outputs/wm_tokens_120b",
    start_index=0,
    sample_seed=30,
    random_sample=True,
    cfg=None,
    wm=None,
    use_plain_cache=True,
    skip_plain=False,
    max_new_tokens_cap=300,
    use_chat_template=True,
    load_in_4bit=False,
    load_in_8bit=False,
    torch_dtype="bfloat16",
    max_memory=None,
):
    set_seed(sample_seed)

    if cfg is None:
        cfg = get_transformers_config(
            model_name=model_name,
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            torch_dtype=torch_dtype,
            max_memory=max_memory,
        )

    if wm is None:
        wm = AutoWatermark.load(algorithm, f"config/{algorithm}.json", cfg)

    dataset = load_dataset_all(dataset_path, tokenizer=cfg.tokenizer)
    total = len(dataset)

    selected_indices = choose_indices(
        total=total,
        max_samples=max_samples,
        sample_seed=sample_seed,
        start_index=start_index,
        random_sample=random_sample,
    )

    os.makedirs(output_dir, exist_ok=True)
    results = []
    token_counter = Counter()
    safe_model_name = sanitize_model_name(model_name)

    plain_cache_path = get_plain_cache_path(
        output_dir=output_dir,
        domain=domain,
        model_name=model_name,
        sample_seed=sample_seed,
        max_samples=max_samples,
        random_sample=random_sample,
    )

    plain_cache = load_plain_cache(plain_cache_path) if use_plain_cache and not skip_plain else {}

    print(f"Dataset total: {total}")
    print(f"Selected samples: {len(selected_indices)}")
    print(f"Sample seed: {sample_seed}")
    print(f"Random sample: {random_sample}")
    print(f"Selected indices preview: {selected_indices[:20]}")

    if use_plain_cache and not skip_plain:
        print(f"Plain cache path: {plain_cache_path}")
        print(f"Loaded plain cache entries: {len(plain_cache)}")

    for local_idx, i in enumerate(selected_indices, start=1):
        s = dataset[i]
        original_text = s.get("prompt") if isinstance(s, dict) else str(s)

        max_new_tokens = get_dynamic_max_new_tokens(
            original_text,
            max_cap=max_new_tokens_cap,
        )

        rewritten = generate_rewrite_with_retry(
            algorithm=algorithm,
            wm=wm,
            cfg=cfg,
            original_text=original_text,
            max_new_tokens=max_new_tokens,
            use_chat_template=use_chat_template,
            watermarked=True,
            max_attempts=2,
        )

        if skip_plain:
            plain = None
        elif use_plain_cache and i in plain_cache:
            plain = clean_rewritten_text(plain_cache[i], original=original_text)
            print(f"[plain cache hit] dataset_index={i}")
        else:
            plain = generate_rewrite_with_retry(
                algorithm=algorithm,
                wm=wm,
                cfg=cfg,
                original_text=original_text,
                max_new_tokens=max_new_tokens,
                use_chat_template=use_chat_template,
                watermarked=False,
                max_attempts=2,
            )

            if use_plain_cache:
                plain_cache[i] = plain
                save_plain_cache(plain_cache, plain_cache_path)
                print(f"[plain cache saved] dataset_index={i}")

        if not rewritten:
            print(f"[warning] empty or contaminated rewritten output at dataset_index={i}")

        if plain is not None and not plain:
            print(f"[warning] empty or contaminated plain output at dataset_index={i}")

        prompt_for_token_collection = build_rewrite_prompt(
            cfg.tokenizer,
            original_text,
            use_chat_template=use_chat_template,
            strict=False,
        )

        injected, stats = collect_watermark_injected_tokens(
            wm,
            prompt_for_token_collection,
            max_steps=max_new_tokens,
        )

        for x in injected:
            token_counter[x["wm_next"]] += 1

        results.append({
            "sample_index": i,
            "sample_seed": sample_seed,
            "random_sample": random_sample,
            "model_name": model_name,
            "algorithm": algorithm,
            "domain": domain,
            "original": original_text,
            "rewritten": rewritten,
            "plain": plain,
            "watermark_injected_tokens": injected,
            "stats": stats,
        })

        print(
            f"[{local_idx}/{len(selected_indices)}] done "
            f"(dataset_index={i}, max_new_tokens={max_new_tokens}, "
            f"rewritten_words={len(rewritten.split()) if rewritten else 0})"
        )

    mode = "random" if random_sample else "sequential"

    out_base = (
        f"{output_dir}/rewritten_{domain}_{algorithm}_{safe_model_name}_"
        f"{mode}_seed{sample_seed}_n{max_samples}"
    )

    with open(f"{out_base}_wm_tokens.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    freq = [
        {
            "token_id": k,
            "token": cfg.tokenizer.decode([k]),
            "count": v,
        }
        for k, v in token_counter.most_common()
    ]

    with open(f"{out_base}_wm_token_freq.json", "w", encoding="utf-8") as f:
        json.dump(freq, f, ensure_ascii=False, indent=2)

    with open(f"{out_base}_selected_indices.json", "w", encoding="utf-8") as f:
        json.dump({
            "model_name": model_name,
            "algorithm": algorithm,
            "domain": domain,
            "dataset_path": dataset_path,
            "sample_seed": sample_seed,
            "max_samples": max_samples,
            "random_sample": random_sample,
            "selected_indices": selected_indices,
        }, f, ensure_ascii=False, indent=2)

    print(f"Finished: {algorithm} - {domain} - {model_name}")
    print(f"Output base: {out_base}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--algorithm", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=30)
    parser.add_argument("--domain", type=str, default="ai")
    parser.add_argument("--output_dir", type=str, default="outputs/wm_tokens_120b")
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--sample_seed", type=int, default=30)
    parser.add_argument("--random_sample", action="store_true")
    parser.add_argument("--sequential_sample", action="store_true")
    parser.add_argument("--skip_plain", action="store_true")
    parser.add_argument("--no_plain_cache", action="store_true")
    parser.add_argument("--max_new_tokens_cap", type=int, default=300)
    parser.add_argument("--no_chat_template", action="store_true")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
    )
    parser.add_argument("--max_memory", type=str, default=None)

    args = parser.parse_args()

    if args.sequential_sample:
        random_sample = False
    elif args.random_sample:
        random_sample = True
    else:
        random_sample = True

    rewrite_and_collect_120b(
        algorithm=args.algorithm,
        dataset_path=args.dataset,
        max_samples=args.max_samples,
        domain=args.domain,
        model_name=args.model_name,
        output_dir=args.output_dir,
        start_index=args.start_index,
        sample_seed=args.sample_seed,
        random_sample=random_sample,
        use_plain_cache=not args.no_plain_cache,
        skip_plain=args.skip_plain,
        max_new_tokens_cap=args.max_new_tokens_cap,
        use_chat_template=not args.no_chat_template,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        torch_dtype=args.torch_dtype,
        max_memory=args.max_memory,
    )
