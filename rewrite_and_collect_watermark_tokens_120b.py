# ============================================================
# rewrite_and_collect_watermark_tokens_120b.py
# 120B subset version
# Fixed version: remove gpt-oss analysis/final meta text
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


# ================== basic setup ==================
DEFAULT_SEED = 30

device = "cuda" if torch.cuda.is_available() else "cpu"


# ================== stop / clean config ==================
# NOTE:
# gpt-oss models can emit harmony/channel-looking text such as
# "analysis" / "assistantfinal" early in the generation. Do NOT use those
# as hard stopping markers, otherwise generation may stop after 1-2 tokens
# and the cleaned rewritten text becomes empty.
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


def get_dynamic_max_new_tokens(text: str, max_cap: int = 260) -> int:
    if not text:
        return min(120, max_cap)

    word_count = len(text.split())

    return min(max_cap, max(180, int(word_count * 1.45)))


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
    gpt-oss may output harmony-style visible text like:
    analysisWe need to rewrite...
    assistantfinalActual answer...

    Keep only the content after assistantfinal / assistant final / final marker.
    Do NOT use this as a stopping condition during generation.
    Only clean after generation is complete.
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

    # Keep content after the last final marker.
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

    # If there is still a leading "analysis..." block and no final marker was found,
    # remove only short obvious leading meta text, not the whole generation.
    text = re.sub(
        r"^analysis\s*(we need to|let'?s|we should|i need to).*?(?=[A-Z][a-z])",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    ).strip()

    # Remove any remaining special tokens.
    text = re.sub(r"<\|.*?\|>", "", text).strip()

    return text

def clean_rewritten_text(text: str) -> str:
    if text is None:
        return ""

    text = strip_gpt_oss_meta_prefix(text)
    text = text.replace("\\n", "\n")
    text = text.strip()

    # Cut off note/explanation/meta parts.
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

    # Remove remaining one-line labels at the beginning.
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

    # Only stop on real post-answer add-ons.
    # Do NOT stop on "analysis", "assistantfinal", or harmony channel markers here;
    # gpt-oss may produce them at the beginning, and stopping immediately causes
    # generated_len ~= 1-2 with rewritten == "".
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


# ================== model config ==================
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
        dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    if max_memory:
        # Example: --max_memory "0:78GiB,cpu:200GiB"
        memory_map = {}
        for item in max_memory.split(","):
            k, v = item.split(":", 1)
            if k.strip().lower() == "cpu":
                memory_map["cpu"] = v.strip()
            else:
                memory_map[int(k.strip())] = v.strip()
        model_kwargs["max_memory"] = memory_map

    # openai/gpt-oss models already carry an MXFP4 quantization config.
    # Passing BitsAndBytesConfig via --load_in_4bit / --load_in_8bit conflicts with it.
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
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=True,
        )

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
        device=device,
        max_new_tokens=300,
        do_sample=False,
    )


def build_rewrite_prompt(tokenizer, text: str, use_chat_template: bool = True) -> str:
    system_instruction = (
        "Reasoning: low\n"
        "You are a rewriting engine. "
        "Return only the rewritten paragraph. "
        "Do not reveal reasoning, analysis, hidden thoughts, channel names, labels, or notes."
    )

    user_instruction = (
        "Rewrite the following paragraph in your own words while preserving the meaning.\n"
        "Preserve all numbers, percentages, dataset names, and technical terms exactly.\n"
        "Output only the rewritten paragraph itself.\n"
        "Do not add explanations, notes, labels, comments, or meta text.\n"
        "Do not repeat sentences.\n"
        "Stop immediately after the rewritten paragraph.\n\n"
        f"Text:\n{text}"
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

    return system_instruction + "\n\n" + user_instruction + "\n\nRewritten paragraph:"


def reset_synthid_state(wm):
    """
    Reset SynthID logits_processor state safely.

    Some SynthID state tensors may be created inside torch.inference_mode().
    In-place ops like fill_() / zero_() on those tensors outside inference mode
    can trigger:

    RuntimeError: Inplace update to inference tensor outside InferenceMode is not allowed.
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
            state[key] = torch.zeros(
                old.shape,
                dtype=old.dtype,
                device=old.device,
            )
            
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
    decoded_raw = tokenizer.decode(completion_ids, skip_special_tokens=False)
    decoded = tokenizer.decode(completion_ids, skip_special_tokens=True)
    cleaned = clean_rewritten_text(decoded)

    if not cleaned:
        print("\n[DEBUG empty generation]")
        print("completion_token_count =", len(completion_ids))
        print("decoded_raw[:1000] =", repr(decoded_raw[:1000]))
        print("decoded_clean[:1000] =", repr(decoded[:1000]))

    return cleaned


def generate_with_exp_watermark(wm, prompt_text: str, max_new_tokens: int = 300):
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

        V = logits.shape[-1]
        probs = torch.softmax(logits[:, :V] / temperature, dim=-1).cpu()

        wm.utils.seed_rng(prefix_ids[0])
        u = torch.rand(V, generator=wm.utils.rng).unsqueeze(0)
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
    decoded = tokenizer.decode(completion_ids, skip_special_tokens=True)
    cleaned = clean_rewritten_text(decoded)

    if not cleaned:
        print("\n[DEBUG empty EXP generation]")
        print("completion_token_count =", len(completion_ids))
        print("decoded_raw[:1000] =", repr(decoded_raw[:1000]))
        print("decoded_clean[:1000] =", repr(decoded[:1000]))

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
            V = base_logits.shape[-1]
            probs = torch.softmax(base_logits[:, :V] / temperature, dim=-1).cpu()
            wm.utils.seed_rng(prefix_ids[0])
            u = torch.rand(V, generator=wm.utils.rng).unsqueeze(0)
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
    # For random sampling, do not pre-truncate to start_index + max_samples.
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
        text = s.get("prompt") if isinstance(s, dict) else str(s)

        max_new_tokens = get_dynamic_max_new_tokens(
            text,
            max_cap=max_new_tokens_cap,
        )

        tokenizer = cfg.tokenizer
        model = cfg.model

        prompt = build_rewrite_prompt(
            tokenizer,
            text,
            use_chat_template=use_chat_template,
        )

        encoded = tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=True,
        ).to(device)

        prompt_len = encoded["input_ids"].shape[1]

        if algorithm == "EXP":
            rewritten = generate_with_exp_watermark(
                wm,
                prompt,
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
                logits_processor=LogitsProcessorList([wm.logits_processor]),
                gen_kwargs=cfg.gen_kwargs,
                max_new_tokens=max_new_tokens,
            )

        if skip_plain:
            plain = None
        elif use_plain_cache and i in plain_cache:
            plain = plain_cache[i]
            print(f"[plain cache hit] dataset_index={i}")
        else:
            plain = generate_completion(
                model=model,
                tokenizer=tokenizer,
                encoded_prompt=encoded,
                prompt_len=prompt_len,
                logits_processor=None,
                gen_kwargs=cfg.gen_kwargs,
                max_new_tokens=max_new_tokens,
            )

            plain = clean_rewritten_text(plain)

            if use_plain_cache:
                plain_cache[i] = plain
                save_plain_cache(plain_cache, plain_cache_path)
                print(f"[plain cache saved] dataset_index={i}")

        rewritten = clean_rewritten_text(rewritten)

        if not rewritten:
            print(f"[warning] empty rewritten output at dataset_index={i}")

        if plain is not None:
            plain = clean_rewritten_text(plain)

        injected, stats = collect_watermark_injected_tokens(
            wm,
            prompt,
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
            "original": text,
            "rewritten": rewritten,
            "plain": plain,
            "watermark_injected_tokens": injected,
            "stats": stats,
        })

        print(
            f"[{local_idx}/{len(selected_indices)}] done "
            f"(dataset_index={i}, max_new_tokens={max_new_tokens})"
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
        # 120B script defaults to random sampling.
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
