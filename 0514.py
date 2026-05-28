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
# ALGORITHMS = ["KGW"]

ALGORITHMS = ["KGW", "SWEET", "Unigram", "EXP", "SynthID"]
DOMAINS = ["ai"]

# TOP_K_TOKENS = 200
# NUM_RETRIES = 3
# MAX_NEW_TOKENS = 200

TEST_LIMIT = 3
# TEST_MODE = True

TOP_K_TOKENS = 200
NUM_RETRIES = 5
MAX_NEW_TOKENS = 200

TEST_MODE = False


MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

OUTPUT_DIR = "outputs/rewrite_avoid_favor_multi_0517"
INPUT_DIR = "outputs/0517_200green"
# INPUT_DIR = "outputs/0517_Test"
TOKEN_DIR = "outputs/0305_200test"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Noise stripping
# ---------------------------------------------------------------------------

# 前綴雜訊：正文前的說明句
_PREFIX_RE = re.compile(
    r"^("
    r"The original text has been rewritten[^.]*\.\s*"
    r"|The following (is|text) (a |the )?(rewritten|revised|paraphrased)[^.]*\.\s*"
    r"|Here is (the |a |my )?(rewritten|revised|paraphrased)[^.]*[:.]\s*"
    r"|Here's (the |a |my )?(rewritten|revised|paraphrased)[^.]*[:.]\s*"
    r"|Below is (the |a )?(rewritten|revised|paraphrased)[^.]*[:.]\s*"
    r"|I (have |'ve )?(rewritten|paraphrased|revised|rephras)[^.]*\.\s*"
    r"|Rewritten (text|version|paragraph)\s*[:\-]\s*"
    r"|Revised (text|version|paragraph)\s*[:\-]\s*"
    r"|Paraphrased (text|version|paragraph)\s*[:\-]\s*"
    r"|Output\s*[:\-]\s*"
    r"|Note\s*[:\-][^\n]*\n+"
    r")+",
    re.IGNORECASE,
)


def strip_model_noise(text: str) -> str:
    """
    三道清洗，順序如下：
    1. 前綴：去掉開頭的說明句（反覆直到穩定）。
    2. 雙換行截斷：\n\n 之後一律視為模型附加說明，直接截斷。
       正常改寫文字不需要段落分隔符，出現 \n\n 就是雜訊開始。
    3. 末尾空白清理。
    """
    text = text.strip()

    # --- 第一道：前綴清洗 ---
    prev = None
    while prev != text:
        prev = text
        text = _PREFIX_RE.sub("", text, count=1).strip()

    # --- 第二道：雙換行截斷（最關鍵） ---
    double_newline_pos = text.find("\n\n")
    if double_newline_pos != -1:
        text = text[:double_newline_pos]

    return text.strip()


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


def normalize_model_name(model_name):
    return model_name.replace("/", "__")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

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
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )

    return model, tokenizer, cfg


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Token / term helpers
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
# Prompt builder
# ---------------------------------------------------------------------------

def build_messages(text, terms, mode="avoid"):
    terms_preview = ", ".join(terms[:80])

    if mode == "avoid":
        system_msg = (
            "You are a rewriting engine. "
            "The very first token of your response MUST be the first word of the rewritten text. "
            "Output ONLY the rewritten text — nothing before it and nothing after it. "
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
            "You are a rewriting engine. "
            "The very first token of your response MUST be the first word of the rewritten text. "
            "Output ONLY the rewritten text — nothing before it and nothing after it. "
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
# Generation helpers
# ---------------------------------------------------------------------------

def generate_with_exp_watermark(wm, prompt_text: str, max_new_tokens: int = MAX_NEW_TOKENS):
    tokenizer = wm.config.generation_tokenizer
    model = wm.config.generation_model
    device = wm.config.device
    temperature = wm.config.temperature

    prefix_ids = tokenizer(
        prompt_text,
        return_tensors="pt",
        add_special_tokens=True,
    ).to(device)["input_ids"]

    prompt_len = prefix_ids.shape[1]

    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model(prefix_ids).logits[:, -1, :]

        vocab_size = logits.shape[-1]
        probs = torch.softmax(logits[:, :vocab_size] / temperature, dim=-1).cpu()

        wm.utils.seed_rng(prefix_ids[0])
        u = torch.rand(vocab_size, generator=wm.utils.rng).unsqueeze(0)
        next_token = wm.utils.exp_sampling(probs, u).to(device)

        next_id = next_token.view(1, 1)
        prefix_ids = torch.cat([prefix_ids, next_id], dim=1)

        if next_id.item() == tokenizer.eos_token_id:
            break

    completion_ids = prefix_ids[0, prompt_len:]
    raw = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
    return strip_model_noise(raw)


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
        **(gen_kwargs or {}),
    )[0]

    completion_ids = output_ids[prompt_len:]
    raw = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
    return strip_model_noise(raw)


# ---------------------------------------------------------------------------
# Rewrite functions
# ---------------------------------------------------------------------------

@torch.no_grad()
def rewrite_once_watermarked_avoid(model, tokenizer, wm, cfg, algorithm, text, avoid_terms):
    """
    Version A: Keep watermark generation. Try to avoid using the token set terms.
    """
    messages = build_messages(text, avoid_terms, mode="avoid")
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    if algorithm == "EXP":
        return generate_with_exp_watermark(wm, prompt, max_new_tokens=MAX_NEW_TOKENS)

    gen_model = wm.config.generation_model
    gen_tokenizer = wm.config.generation_tokenizer

    encoded = gen_tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=True,
    ).to(wm.config.device)

    prompt_len = encoded["input_ids"].shape[1]

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
def rewrite_once_unwatermarked_favor(model, tokenizer, cfg, text, favor_terms):
    """
    Version B: No watermark logits processor. Try to use the token set terms.
    """
    messages = build_messages(text, favor_terms, mode="favor")
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

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
        logits_processor=None,
        gen_kwargs=cfg.gen_kwargs,
    )


# ---------------------------------------------------------------------------
# Candidate scoring
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
    """Prefer fewer set-token hits. Tie-breaker: length closer to source."""
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
    """Prefer more set-token hits. Tie-breaker: length closer to source."""
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

    if TEST_MODE:
        data = data[:TEST_LIMIT]
        print(f"[TEST MODE] Only running first {len(data)} samples.")

    token_ids = load_top_token_ids(token_path, TOP_K_TOKENS)
    set_terms = token_ids_to_terms(tokenizer, token_ids)

    output = []

    for item in tqdm(data, desc=f"{domain}/{algorithm}"):
        src_text = item.get("rewritten") or item.get("plain") or item.get("text") or ""

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

        for _ in range(NUM_RETRIES):
            try:
                rw_avoid = rewrite_once_watermarked_avoid(
                    model=model,
                    tokenizer=tokenizer,
                    wm=wm,
                    cfg=cfg,
                    algorithm=algorithm,
                    text=src_text,
                    avoid_terms=set_terms,
                )
            except Exception as e:
                print(f"[WARN] {domain}/{algorithm} watermarked avoid failed: {e}")
                rw_avoid = ""

            watermarked_avoid_candidates.append(rw_avoid)

            try:
                rw_favor = rewrite_once_unwatermarked_favor(
                    model=model,
                    tokenizer=tokenizer,
                    cfg=cfg,
                    text=src_text,
                    favor_terms=set_terms,
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

            # Version A: watermark preserved, set terms avoided.
            "rewrite_watermarked_avoid_set": best_avoid["text"],
            "watermarked_avoid_set_hits": best_avoid["set_hits"],
            "watermarked_avoid_set_hit_terms": best_avoid["set_hit_terms"],
            "watermarked_avoid_candidates": scored_avoid,

            # Version B: no watermark logits processor, set terms favored.
            "rewrite_unwatermarked_favor_set": best_favor["text"],
            "unwatermarked_favor_set_hits": best_favor["set_hits"],
            "unwatermarked_favor_set_hit_terms": best_favor["set_hit_terms"],
            "unwatermarked_favor_candidates": scored_favor,
        })

    save_json(output, save_path)
    print(f"[DONE] Saved: {save_path}")


def main():
    model, tokenizer, cfg = load_model_tokenizer_and_cfg(MODEL_NAME)

    for domain in DOMAINS:
        for algorithm in ALGORITHMS:
            run_one_setting(
                model=model,
                tokenizer=tokenizer,
                cfg=cfg,
                model_name=MODEL_NAME,
                domain=domain,
                algorithm=algorithm,
            )


if __name__ == "__main__":
    main()