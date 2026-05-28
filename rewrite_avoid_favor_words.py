# MODELS = [
#     "Qwen/Qwen2.5-7B-Instruct",
#     "01-ai/Yi-1.5-9B-Chat",
#     "meta-llama/Llama-3.1-8B-Instruct",
# ]

import json
import os
import re
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import LogitsProcessorList
from detect_avoidfavor import REWRITE_MODEL_TAG
from watermark.auto_watermark import AutoWatermark
from utils.transformers_config import TransformersConfig

# ALGORITHMS = ["KGW", "SWEET", "Unigram", "EXP", "SynthID"]
# DOMAINS = ["ai", "bio", "med", "mis", "security"]
ALGORITHMS = ["KGW", "SWEET"]
DOMAINS = ["ai", "bio"]

TOP_K_TOKENS = 200
# NUM_RETRIES = 3
# MAX_NEW_TOKENS = 256
NUM_RETRIES = 5
MAX_NEW_TOKENS = 400


MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

OUTPUT_DIR = "outputs/rewrite_avoid_favor_multi_0511"
INPUT_DIR = "outputs/0123_200green"
TOKEN_DIR = "outputs/0305_200test"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def normalize_model_name(model_name):
    return model_name.replace("/", "__")


def load_model_tokenizer_and_cfg(model_name):
    print(f"Loading model: {model_name}")

    nf4 = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
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
        do_sample=False,
    )

    return model, tokenizer, cfg


def get_input_text_json(domain, algorithm):
    return os.path.join(
        INPUT_DIR,
        f"rewritten_{domain}_{algorithm}_{REWRITE_MODEL_TAG}_wm_tokens.json"
    )


def get_token_set_json(domain, algorithm):
    return os.path.join(
        TOKEN_DIR,
        f"rewritten_{domain}_{algorithm}_wm_token_freq.json"
    )


def get_output_json(domain, algorithm, model_name):
    return os.path.join(
        OUTPUT_DIR,
        f"rewritten_{domain}_{algorithm}_{normalize_model_name(model_name)}_avoid_top{TOP_K_TOKENS}.json"
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


def build_messages(text, avoid_terms):
    avoid_preview = ", ".join(avoid_terms[:80])

    system_msg = (
        "You are a precise rewriting assistant. "
        "Rewrite the user's text while preserving its meaning, facts, tone, and length. "
        "Do not summarize, shorten, omit, compress, or remove details. "
        "The rewritten text must be approximately the same length as the input text. "
        "Avoid using the listed favored words or phrases whenever possible. "
        "If total avoidance would hurt fluency or meaning, preserve meaning first but still minimize their usage. "
        "Return only the rewritten text."
    )

    user_msg = f"""Please rewrite the following text.

Requirements:
1. Preserve all meaning and factual content.
2. Do not summarize, shorten, omit, or compress the text.
3. Keep the output fluent and natural.
4. The rewritten text should be between 90% and 110% of the input length.
5. Avoid these favored words/phrases as much as possible:
{avoid_preview}

Original text:
{text}
"""
    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]


def generate_with_exp_watermark(
    wm,
    prompt_text: str,
    max_new_tokens: int = MAX_NEW_TOKENS,
):
    tokenizer = wm.config.generation_tokenizer
    model = wm.config.generation_model
    device = wm.config.device
    temperature = wm.config.temperature

    prefix_ids = tokenizer(
        prompt_text,
        return_tensors="pt",
        add_special_tokens=True
    ).to(device)["input_ids"]

    prompt_len = prefix_ids.shape[1]

    for step in range(max_new_tokens):
        with torch.no_grad():
            logits = model(prefix_ids).logits[:, -1, :]

        V = logits.shape[-1]
        probs = torch.softmax(logits[:, :V] / temperature, dim=-1).cpu()

        wm.utils.seed_rng(prefix_ids[0])
        u = torch.rand(V, generator=wm.utils.rng).unsqueeze(0)
        next_token = wm.utils.exp_sampling(probs, u).to(device)

        next_id = next_token.view(1, 1)
        prefix_ids = torch.cat([prefix_ids, next_id], dim=1)

        if next_id.item() == tokenizer.eos_token_id:
            break

    completion_ids = prefix_ids[0, prompt_len:]
    return tokenizer.decode(completion_ids, skip_special_tokens=True).strip()

def reset_synthid_state(wm):
    if getattr(wm.config, "algorithm_name", None) != "SynthID":
        return

    lp = wm.logits_processor

    if not hasattr(lp, "state") or lp.state is None:
        lp.state = None
        return

    if "num_calls" in lp.state:
        lp.state["num_calls"] = 0
    if "context" in lp.state and lp.state["context"] is not None:
        lp.state["context"].fill_(0)
    if "context_history" in lp.state and lp.state["context_history"] is not None:
        lp.state["context_history"].fill_(0)

@torch.no_grad()
def rewrite_once(model, tokenizer, wm, cfg, algorithm, text, avoid_terms):
    messages = build_messages(text, avoid_terms)
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    gen_model = wm.config.generation_model
    gen_tokenizer = wm.config.generation_tokenizer

    encoded = gen_tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=True
    ).to(wm.config.device)

    prompt_len = encoded["input_ids"].shape[1]

    if algorithm == "EXP":
        return generate_with_exp_watermark(
            wm, prompt, max_new_tokens=MAX_NEW_TOKENS
        )

    reset_synthid_state(wm)

    return generate_completion(
        model=gen_model,
        tokenizer=gen_tokenizer,
        encoded_prompt=encoded,
        prompt_len=prompt_len,
        logits_processor=LogitsProcessorList([wm.logits_processor]),
        gen_kwargs=cfg.gen_kwargs,
    )
@torch.no_grad()
def generate_completion(
    model,
    tokenizer,
    encoded_prompt,
    prompt_len,
    logits_processor=None,
    gen_kwargs=None,
):
    output_ids = model.generate(
        **encoded_prompt,
        logits_processor=logits_processor,
        **(gen_kwargs or {})
    )[0]

    completion_ids = output_ids[prompt_len:]
    return tokenizer.decode(completion_ids, skip_special_tokens=True).strip()

def count_term_hits(text, terms):
    low = text.lower()
    total_hits = 0
    hit_terms = []

    for t in terms:
        tt = t.lower().strip()
        if not tt:
            continue
        c = low.count(tt)
        if c > 0:
            total_hits += c
            hit_terms.append([t, c])

    return total_hits, hit_terms


def choose_best_candidate(candidates, avoid_terms, src_len):
    scored = []
    for c in candidates:
        hits, hit_terms = count_term_hits(c, avoid_terms)
        scored.append({
            "text": c,
            "hits": hits,
            "hit_terms": hit_terms,
            "length_gap": abs(len(c) - src_len),
        })

    scored.sort(key=lambda x: (x["hits"], x["length_gap"]))
    return scored[0], scored


def run_one_setting(model, tokenizer, cfg, model_name, domain, algorithm):
    input_path = get_input_text_json(domain, algorithm)
    token_path = get_token_set_json(domain, algorithm)
    save_path = get_output_json(domain, algorithm, model_name)

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
    token_ids = load_top_token_ids(token_path, TOP_K_TOKENS)
    avoid_terms = token_ids_to_terms(tokenizer, token_ids)

    output = []

    for item in tqdm(data, desc=f"{domain}/{algorithm}"):
        src_text = item.get("rewritten") or item.get("plain") or item.get("text") or ""

        if not src_text:
            output.append({
                **item,
                "domain": domain,
                "algorithm": algorithm,
                "rewrite_model": model_name,
                "rewrite_avoid_favor": "",
                "avoid_hits": None,
                "avoid_hit_terms": [],
                "all_candidates": [],
            })
            continue

        candidates = []
        for _ in range(NUM_RETRIES):
            try:
                rw = rewrite_once(
                    model=model,
                    tokenizer=tokenizer,
                    wm=wm,
                    cfg=cfg,
                    algorithm=algorithm,
                    text=src_text,
                    avoid_terms=avoid_terms,
                )
            except Exception as e:
                print(f"[WARN] {domain}/{algorithm} failed: {e}")
                rw = ""
            candidates.append(rw)

        best, scored = choose_best_candidate(candidates, avoid_terms, len(src_text))

        output.append({
            **item,
            "domain": domain,
            "algorithm": algorithm,
            "rewrite_model": model_name,
            "rewrite_avoid_favor": best["text"],
            "avoid_hits": best["hits"],
            "avoid_hit_terms": best["hit_terms"],
            "all_candidates": scored,
        })

    save_json(output, save_path)
    print(f"[DONE] Saved: {save_path}")

def main():
    model, tokenizer, cfg = load_model_tokenizer_and_cfg(MODEL_NAME)

    for domain in DOMAINS:
        for algorithm in ALGORITHMS:
            run_one_setting(model, tokenizer, cfg, MODEL_NAME, domain, algorithm)

if __name__ == "__main__":
    main()