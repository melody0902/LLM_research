# ============================================================
# rewrite_and_collect_watermark_tokens.py
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
seed = 30
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

device = "cuda" if torch.cuda.is_available() else "cpu"


# ================== stop / clean config ==================
STOP_MARKERS = [
    "\nNote that",
    "\nNote:",
    "\n(Note:",
    "\nNotes:",
    "\nExplanation:",
    "\nOriginal:",
    # "\nText:",
    # "\nRewritten:",
    # "\nRewritten paragraph:",
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


# ================== utils ==================
def sanitize_model_name(model_name: str) -> str:
    return model_name.replace("/", "__").replace(" ", "_")


def get_dynamic_max_new_tokens(text: str) -> int:
    """
    Dynamically control generation length.

    The goal is to avoid long repeated tails while still allowing
    longer input paragraphs to receive enough output budget.
    """
    if not text:
        return 100

    word_count = len(text.split())

    # Usually rewrite length should be close to original.
    # Keep a reasonable lower and upper bound.
    return min(200, max(70, int(word_count * 1.15)))


def remove_repeated_sentences(text: str) -> str:
    """
    Light post-processing to remove repeated sentence tails.

    This is useful because some outputs keep repeating the same
    rewritten sentence until max_new_tokens is reached.
    """
    if not text:
        return ""

    sentences = re.split(r"(?<=[.!?。！？])\s+", text.strip())

    cleaned = []
    seen = set()

    for sent in sentences:
        normalized = re.sub(r"\s+", " ", sent.strip().lower())

        if not normalized:
            continue

        # If the exact same sentence appears again, drop later copies.
        if normalized in seen:
            continue

        seen.add(normalized)
        cleaned.append(sent.strip())

    return " ".join(cleaned).strip()


def clean_rewritten_text(text: str) -> str:
    """
    Clean common trailing artifacts from rewritten output.

    Removed examples:
    - excessive blank lines
    - Note that...
    - Note:
    - (Note: I made some minor changes...)
    - Explanation:
    - Original:
    - Text:
    - Rewritten:
    - repeated sentence tails
    """
    if text is None:
        return ""

    # Some outputs may literally contain "\\n".
    text = text.replace("\\n", "\n")
    text = text.strip()

    # Cut from the first occurrence of any meta marker.
    for marker in STOP_MARKERS:
        idx = text.find(marker)
        if idx != -1:
            text = text[:idx].strip()

    # Extra regex-based cleanup for variations like:
    # "(Note: ...)", "Note - ...", "Note：..."
    note_patterns = [
        r"\n?\(?\s*Note\s*[:：-].*$",
        r"\n?\(?\s*Explanation\s*[:：-].*$",
        r"\n?\(?\s*Original\s*[:：-].*$",
        r"\n?\(?\s*Rewritten\s*(paragraph)?\s*[:：-].*$",
        r"\n?\(?\s*I made some minor changes.*$",
        r"\n?\(?\s*I have made some minor changes.*$",
    ]

    for pattern in note_patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.DOTALL).strip()

    # Compress excessive blank lines.
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")

    # Remove repeated sentence tails.
    text = remove_repeated_sentences(text)

    return text.strip()


def should_stop_generation(decoded_text: str) -> bool:
    """
    Stop early when the model starts producing notes / explanations / labels.
    Used during manual EXP generation and HF stopping criteria.
    """
    if not decoded_text:
        return False

    return any(marker in decoded_text for marker in STOP_MARKERS)


class StopOnMarkersCriteria(StoppingCriteria):
    """
    Hugging Face generation stopping criteria.

    This lets plain generation stop immediately when the model starts
    producing unwanted meta text such as:
    - Note:
    - Explanation:
    - (Note: I made some minor changes...)
    """
    def __init__(self, tokenizer, prompt_len: int):
        self.tokenizer = tokenizer
        self.prompt_len = prompt_len

    def __call__(self, input_ids, scores, **kwargs):
        generated_ids = input_ids[0, self.prompt_len:]
        decoded = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        )

        return should_stop_generation(decoded)


# ================== model config ==================
def get_transformers_config(model_name):
    print(f"Using model: {model_name}")

    nf4 = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # model = AutoModelForCausalLM.from_pretrained(
    #     model_name,
    #     device_map={"": 0},
    #     torch_dtype=torch.bfloat16,
    #     quantization_config=nf4,
    #     low_cpu_mem_usage=True,
    # )

    # tokenizer = AutoTokenizer.from_pretrained(model_name)


    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        dtype=torch.bfloat16,
        quantization_config=nf4,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    # Use the model's real output vocab size instead of len(tokenizer).
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
        max_new_tokens=180,
        do_sample=False,
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
    """
    Normal generation path.

    This version:
    - supports watermark logits_processor
    - stops on unwanted Note / Explanation / labels
    - reduces repeated output
    - cleans final text
    """
    safe_gen_kwargs = dict(gen_kwargs or {})

    if max_new_tokens is None:
        max_new_tokens = 120

    safe_gen_kwargs["max_new_tokens"] = max_new_tokens
    safe_gen_kwargs.setdefault("do_sample", False)

    # Reduce repetition.
    safe_gen_kwargs.setdefault("repetition_penalty", 1.1)
    safe_gen_kwargs.setdefault("no_repeat_ngram_size", 4)

    if tokenizer.eos_token_id is not None:
        safe_gen_kwargs.setdefault("eos_token_id", tokenizer.eos_token_id)

    if tokenizer.pad_token_id is not None:
        safe_gen_kwargs.setdefault("pad_token_id", tokenizer.pad_token_id)

    stopping_criteria = StoppingCriteriaList([
        StopOnMarkersCriteria(tokenizer, prompt_len)
    ])

    with torch.no_grad():
        output_ids = model.generate(
            **encoded_prompt,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            **safe_gen_kwargs,
        )[0]

    completion_ids = output_ids[prompt_len:]
    decoded = tokenizer.decode(completion_ids, skip_special_tokens=True)

    return clean_rewritten_text(decoded)


def generate_with_exp_watermark(
    wm,
    prompt_text: str,
    max_new_tokens: int = 120,
):
    """
    Generate rewritten text using EXP watermark sampling.

    This version:
    - keeps the true EXP watermark path
    - stops on EOS
    - stops if the model starts generating Note / Explanation / labels
    - cleans the final rewritten text
    """
    tokenizer = wm.config.generation_tokenizer
    model = wm.config.generation_model
    device = wm.config.device

    temperature = wm.config.temperature

    encoded = tokenizer(
        prompt_text,
        return_tensors="pt",
        add_special_tokens=True,
    ).to(device)

    prefix_ids = encoded["input_ids"]
    attention_mask = encoded.get("attention_mask", None)

    prompt_len = prefix_ids.shape[1]

    for step in range(max_new_tokens):
        with torch.no_grad():
            if attention_mask is not None:
                logits = model(
                    prefix_ids,
                    attention_mask=attention_mask,
                ).logits[:, -1, :]
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
            next_attention = torch.ones(
                (1, 1),
                device=device,
                dtype=attention_mask.dtype,
            )
            attention_mask = torch.cat([attention_mask, next_attention], dim=1)

        if tokenizer.eos_token_id is not None and next_id.item() == tokenizer.eos_token_id:
            break

        partial_text = tokenizer.decode(
            prefix_ids[0, prompt_len:],
            skip_special_tokens=True,
        )

        if should_stop_generation(partial_text):
            break

    completion_ids = prefix_ids[0, prompt_len:]
    decoded = tokenizer.decode(completion_ids, skip_special_tokens=True)

    return clean_rewritten_text(decoded)


def collect_watermark_injected_tokens(
    wm,
    prompt_text,
    max_steps=120,
):
    tokenizer = wm.config.generation_tokenizer
    model = wm.config.generation_model
    device = wm.config.device
    algo = wm.config.algorithm_name

    prefix_ids = tokenizer(prompt_text, return_tensors="pt").to(device)["input_ids"]
    injected = []

    # SynthID state initialization.
    if algo == "SynthID":
        if hasattr(wm.logits_processor, "state") and wm.logits_processor.state is not None:
            wm.logits_processor.state["num_calls"] = 0
            wm.logits_processor.state["context"].fill_(0)
            wm.logits_processor.state["context_history"].fill_(0)
        else:
            wm.logits_processor.state = None

    generated_len = 0
    
    prompt_len = prefix_ids.shape[1]

    for step in range(max_steps):
        generated_len = step + 1

        with torch.no_grad():
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

        if tokenizer.eos_token_id is not None and wm_next == tokenizer.eos_token_id:
            break

        partial_text = tokenizer.decode(
            prefix_ids[0, prompt_len:],
            skip_special_tokens=True,
        )

        if should_stop_generation(partial_text):
            break

    return injected, {
        "generated_len": generated_len,
    }


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

def get_plain_cache_path(output_dir, domain, model_name, start_index, max_samples):
    safe_model_name = sanitize_model_name(model_name)

    cache_dir = os.path.join(output_dir, "plain_cache")
    os.makedirs(cache_dir, exist_ok=True)

    return os.path.join(
        cache_dir,
        f"plain_cache_{domain}_{safe_model_name}_start{start_index}_n{max_samples}.json"
    )


def load_plain_cache(cache_path):
    if not os.path.exists(cache_path):
        return {}

    with open(cache_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return {
        int(k): v
        for k, v in data.items()
    }


def save_plain_cache(cache, cache_path):
    data = {
        str(k): v
        for k, v in cache.items()
    }

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# ============================================================
# Main pipeline
# ============================================================
def rewrite_and_collect(
    algorithm,
    dataset_path,
    max_samples,
    domain,
    model_name,
    output_dir="outputs/wm_tokens",
    start_index=0,
    cfg=None,
    wm=None,
    use_plain_cache=True,
):
    if cfg is None:
        cfg = get_transformers_config(model_name)

    if wm is None:
        wm = AutoWatermark.load(algorithm, f"config/{algorithm}.json", cfg)

    if dataset_path.endswith(".jsonl"):
        dataset = load_prompt_jsonl(
            dataset_path,
            max_samples=start_index + max_samples,
        )
    elif "zhtw" in dataset_path.lower():
        dataset = ZHTWC4Dataset(
            dataset_path,
            tokenizer=cfg.tokenizer,
            max_samples=start_index + max_samples,
        )
    else:
        dataset = C4Dataset(
            dataset_path,
            max_samples=start_index + max_samples,
        )

    os.makedirs(output_dir, exist_ok=True)

    results = []
    token_counter = Counter()
    safe_model_name = sanitize_model_name(model_name)

    plain_cache_path = get_plain_cache_path(
        output_dir=output_dir,
        domain=domain,
        model_name=model_name,
        start_index=start_index,
        max_samples=max_samples,
    )

    plain_cache = load_plain_cache(plain_cache_path) if use_plain_cache else {}

    if use_plain_cache:
        print(f"Plain cache path: {plain_cache_path}")
        print(f"Loaded plain cache entries: {len(plain_cache)}")

    total = len(dataset)
    start = min(start_index, total)
    end = min(start_index + max_samples, total)

    for i in range(start, end):
        s = dataset[i]
        text = s.get("prompt") if isinstance(s, dict) else str(s)

        max_new_tokens = get_dynamic_max_new_tokens(text)

        prompt = (
            "Rewrite the following paragraph in your own words while preserving the meaning.\n"
            "Output only one rewritten paragraph.\n"
            "Do not add notes, explanations, labels, comments, parentheses, or meta text.\n"
            "Do not mention what changes you made.\n"
            "Do not repeat sentences.\n"
            "Stop immediately after the rewritten paragraph.\n\n"
            f"Text:\n{text}\n\n"
            "Rewritten paragraph:"
        )

        tokenizer = cfg.tokenizer
        model = cfg.model

        encoded = tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=True,
        ).to(device)

        prompt_len = encoded["input_ids"].shape[1]

        # 1. Generate watermarked rewrite.
        if algorithm == "EXP":
            rewritten = generate_with_exp_watermark(
                wm,
                prompt,
                max_new_tokens=max_new_tokens,
            )
        else:
            rewritten = generate_completion(
                model=model,
                tokenizer=tokenizer,
                encoded_prompt=encoded,
                prompt_len=prompt_len,
                logits_processor=LogitsProcessorList([wm.logits_processor]),
                gen_kwargs=cfg.gen_kwargs,
                max_new_tokens=max_new_tokens,
            )

        # 2. Generate or reuse non-watermarked rewrite.
        if use_plain_cache and i in plain_cache:
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

        # 3. Final cleanup.
        rewritten = clean_rewritten_text(rewritten)
        plain = clean_rewritten_text(plain)

        # 4. Collect injected watermark tokens.
        injected, stats = collect_watermark_injected_tokens(
            wm,
            prompt,
            max_steps=max_new_tokens,
        )

        for x in injected:
            token_id = x["wm_next"]
            token_counter[token_id] += 1

        results.append({
            "sample_index": i,
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
            f"[{i - start + 1}/{end - start}] done "
            f"(dataset_index={i}, max_new_tokens={max_new_tokens})"
        )

    out_base = f"{output_dir}/rewritten_{domain}_{algorithm}_{safe_model_name}"

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

    print(f"Finished: {algorithm} - {domain} - {model_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=10)
    parser.add_argument("--domain", type=str, default="ai")
    parser.add_argument("--output_dir", type=str, default="outputs/wm_tokens")
    parser.add_argument("--start_index", type=int, default=0)

    args = parser.parse_args()

    rewrite_and_collect(
        algorithm=args.algorithm,
        dataset_path=args.dataset,
        max_samples=args.max_samples,
        domain=args.domain,
        model_name=args.model_name,
        output_dir=args.output_dir,
        start_index=args.start_index,
    )
