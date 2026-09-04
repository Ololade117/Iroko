"""
Iroko — Mental Health Support Chatbot (ZeroGPU Space harness)

Loads google/gemma-4-E2B-it + the finetuned5 LoRA adapter, and wraps generation with:
  - a system prompt guarding identity/credential disclosure
  - input-side risk detection (hard keywords, structural regex, conversation escalation)
  - output-side moderation (screens the MODEL'S OWN generated response before showing it)
  - decoding fixes (repetition penalty, greedy decoding on elevated risk)
  - per-user logging to a shared Google Sheet via a service account (no interactive login,
    since a Space has no stdin for input() / OAuth device flow)

Runs on free ZeroGPU hardware (Settings -> Hardware -> ZeroGPU on the Space).
"""

import os
import re
import json
import time
import threading
from datetime import datetime, timezone

os.environ["HF_HUB_DISABLE_XET"] = "1"

import torch
import gradio as gr
import spaces  # required for ZeroGPU — gates the actual GPU compute per call
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# ============================================================
# Configuration
# ============================================================

BASE_MODEL_ID = "google/gemma-4-E2B-it"
ADAPTER_ID = "Ololade117/gemma-4-e2b-iroko-mentalhealth-finetuned5"

SYSTEM_PROMPT = (
    "You are Iroko, an AI assistant offering mental health support. You are not a "
    "licensed therapist, counselor, or human, and you have no name, workplace, or "
    "professional credentials — if asked about your identity, disclose this honestly "
    "and directly. You do not diagnose conditions. For any indication of crisis, "
    "self-harm, or harm to others, prioritize directing the person to real help: the "
    "Mentally Aware Nigeria Initiative (MANI) crisis line (0809 111 6264 or "
    "0811 1680 686) and, for immediate danger, 112 (Nigeria's national emergency line)."
)

CRISIS_MESSAGE = (
    "It sounds like you're going through something really painful right now. "
    "Please reach out to the Mentally Aware Nigeria Initiative (MANI) crisis line at "
    "0809 111 6264 or 0811 1680 686, available 24/7. If you're in immediate danger, "
    "please call 112 (Nigeria's national emergency line) or go to the nearest hospital. "
    "I'm still here if you'd like to talk more."
)

# ============================================================
# Input-side risk detection
# ============================================================

CRISIS_KEYWORDS = [
    "suicide", "kill myself", "end my life", "want to die",
    "self-harm", "self harm", "hurt myself", "cutting myself",
]

METHOD_NOUNS = r"(gun|knife|poison|pills?|rope|bridge|overdose|acid|blade|razor)"
INTENT_VERBS = r"(how (do|to|can) i|tell me how|what.s the (best|easiest) way|help me (figure out|do it)|prescribe)"
EUPHEMISM_PATTERNS = [
    r"sleep forever", r"end it all", r"go to sleep and not wake up",
    r"meet (jesus|god)", r"release (the )?(pressure|anger) (by|to) (kill|hurt|harm)",
    r"as a joke.{0,30}(kill|murder|jump|hurt)", r"non.?violent way to end",
    r"skydiving without a parachute",
]


def safe_text(value) -> str:
    """Convert Gradio message content into a plain string."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("text", "content", "value"):
            if key in value:
                return safe_text(value[key])
        return str(value)
    if isinstance(value, (list, tuple)):
        parts = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(safe_text(item))
            else:
                parts.append(str(item))
        return " ".join(part for part in parts if part).strip()
    return str(value)


def contains_crisis_language(text: str) -> bool:
    lowered = safe_text(text).lower()
    return any(kw in lowered for kw in CRISIS_KEYWORDS)


def structural_risk_check(text: str) -> bool:
    lowered = safe_text(text).lower()
    if re.search(INTENT_VERBS, lowered) and re.search(METHOD_NOUNS, lowered):
        return True
    return any(re.search(pattern, lowered) for pattern in EUPHEMISM_PATTERNS)


def normalize_history(history) -> list:
    """Normalize old and new Gradio history formats to role/content dicts."""
    normalized = []
    if not history:
        return normalized

    for item in history:
        if isinstance(item, dict):
            role = item.get("role")
            if role not in ("user", "assistant", "system"):
                continue
            content = safe_text(item.get("content"))
            if content:
                normalized.append({"role": role, "content": content})
            continue

        if isinstance(item, (list, tuple)):
            if len(item) >= 1:
                user_msg = safe_text(item[0])
                if user_msg:
                    normalized.append({"role": "user", "content": user_msg})
            if len(item) >= 2:
                assistant_msg = safe_text(item[1])
                if assistant_msg:
                    normalized.append({"role": "assistant", "content": assistant_msg})
            continue

        text = safe_text(item)
        if text:
            normalized.append({"role": "user", "content": text})

    return normalized


def assess_conversation_risk(normalized_history: list, message: str) -> tuple[bool, str]:
    """Looks at the current message AND recent turns — escalating risk across a
    conversation matters, as the Iroko3 chat logs showed clearly (a single-message
    check missed a session where risk built up turn by turn)."""
    current_structural = structural_risk_check(message)
    recent_user_msgs = [
        h["content"] for h in normalized_history[-6:] if h.get("role") == "user"
    ]
    recent_flags = sum(1 for m in recent_user_msgs if structural_risk_check(m))

    if current_structural:
        return True, "current_structural"
    if recent_flags >= 2:
        return True, "repeated_risk_signals"
    return False, "none"


# ============================================================
# Output-side moderation — NEW: screens the MODEL'S OWN response, not just the input.
# The Iroko3 log showed the model actively engaging with harmful requests even when
# the input-side filter missed them; this is the second, independent layer that
# checks what the model actually produced before it ever reaches the user.
# ============================================================

OUTPUT_RED_FLAG_PATTERNS = [
    r"\byou (are right|have the right idea)\b.{0,60}(gun|knife|hurt|kill|harm)",
    r"\b(here.s how|the best way to)\b.{0,60}" + METHOD_NOUNS,
    r"\b(evidence|getting caught|avoid (legal|police))\b.{0,60}(murder|poison|kill)",
    r"\bit is good\b.{0,40}(kill|murder|hurt|harm)",
]

FALLBACK_RESPONSE = (
    "I want to be careful here rather than say something that could be unsafe. "
    "If you're dealing with thoughts of harming yourself or someone else, please reach "
    "out to the Mentally Aware Nigeria Initiative (MANI) crisis line at 0809 111 6264 "
    "or 0811 1680 686 — they can help in a way I'm not equipped to. I'm still glad to "
    "keep talking with you about how you're feeling."
)


def output_flagged_as_unsafe(response: str) -> bool:
    lowered = safe_text(response).lower()
    return any(re.search(pattern, lowered) for pattern in OUTPUT_RED_FLAG_PATTERNS)


# ============================================================
# Model loading — MUST happen inside a @spaces.GPU-decorated function on ZeroGPU.
# The `spaces` package patches torch to intercept any CUDA operation that happens
# outside a decorated function's execution window and raises "No CUDA GPUs are
# available" instead — which is exactly what module-level loading hit. Loaded once,
# cached in _MODEL_STATE, and reused across requests since the underlying process
# (and its GPU-resident tensors) persists between calls even though GPU *access*
# is only actively granted during each decorated call.
# ============================================================

_MODEL_STATE = {"model": None, "tokenizer": None}
_load_lock = threading.Lock()


def _load_model_if_needed():
    with _load_lock:
        if _MODEL_STATE["model"] is not None:
            return

        print("Loading base model...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            quantization_config=bnb_config,
            trust_remote_code=True,
            device_map="auto",
            max_memory={0: "13GiB", "cpu": "30GiB"},
            low_cpu_mem_usage=True,
            dtype=torch.bfloat16,
        )

        n_unwrapped = unwrap_clippable_linear(base_model)
        print(f"Unwrapped {n_unwrapped} Gemma4ClippableLinear layers")

        print(f"Loading LoRA adapter from {ADAPTER_ID}...")
        model = PeftModel.from_pretrained(base_model, ADAPTER_ID)
        model.eval()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = True

        tokenizer = AutoTokenizer.from_pretrained(ADAPTER_ID)

        model.generation_config.repetition_penalty = 1.2
        model.generation_config.no_repeat_ngram_size = 3

        _MODEL_STATE["model"] = model
        _MODEL_STATE["tokenizer"] = tokenizer
        print("Model ready.")


def unwrap_clippable_linear(model):
    replaced = 0
    for parent in list(model.modules()):
        for child_name, child in list(parent.named_children()):
            if type(child).__name__ == "Gemma4ClippableLinear":
                setattr(parent, child_name, child.linear)
                replaced += 1
    return replaced

# ============================================================
# Logging — service account auth (no interactive login available on a Space).
# Requires two Space secrets (Settings -> Variables and secrets):
#   GOOGLE_SERVICE_ACCOUNT_JSON  — the full JSON key for a Google service account
#   SHEET_ID                    — the target Google Sheet's ID (share the Sheet
#                                  with the service account's email as Editor)
# If either is missing, logging is silently disabled rather than crashing the app.
# ============================================================

log_sheet = None
try:
    import gspread

    service_account_json = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON")
    sheet_id = os.environ.get("SHEET_ID")
    if service_account_json and sheet_id:
        creds_dict = json.loads(service_account_json)
        gc = gspread.service_account_from_dict(creds_dict)
        log_sheet = gc.open_by_key(sheet_id).sheet1
        print("Sheet logging enabled.")
    else:
        print("GOOGLE_SERVICE_ACCOUNT_JSON or SHEET_ID not set — logging disabled.")
except Exception as e:
    print(f"Could not initialize Sheet logging: {e}")
    log_sheet = None


def log_conversation(user_name, user_message, bot_response, risk_level):
    if log_sheet is None:
        return
    row = [
        datetime.now(timezone.utc).isoformat(),
        user_name or "anonymous",
        user_message,
        bot_response,
        risk_level,
    ]
    try:
        log_sheet.append_row(row)
    except Exception as e:
        print(f"Warning: could not log to Sheet ({e})")


def log_conversation_async(user_name, user_message, bot_response, risk_level):
    threading.Thread(
        target=log_conversation,
        args=(user_name, user_message, bot_response, risk_level),
        daemon=True,
    ).start()


# ============================================================
# Generation — only this function touches CUDA, so only this is @spaces.GPU-decorated
# ============================================================

@spaces.GPU(duration=180)
def run_generation(messages, elevated_risk: bool):
    # Lazy-load on first call — this runs inside the GPU-granted window, unlike
    # module-level loading. Cheap no-op check on every call after the first.
    _load_model_if_needed()
    model = _MODEL_STATE["model"]
    tokenizer = _MODEL_STATE["tokenizer"]

    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True,
    ).to(model.device)

    with torch.no_grad():
        if elevated_risk:
            # Greedy decoding when risk is elevated — fixes the inconsistency-on-
            # near-identical-input failure seen in Iroko3 (two nearly identical
            # risky messages got two different responses under sampling).
            output = model.generate(
                **inputs, max_new_tokens=300, do_sample=False,
                repetition_penalty=1.2, no_repeat_ngram_size=3,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        else:
            output = model.generate(
                **inputs, max_new_tokens=300, do_sample=True,
                temperature=0.7, top_p=0.9,
                repetition_penalty=1.2, no_repeat_ngram_size=3,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

    input_len = inputs["input_ids"].shape[-1]
    return tokenizer.decode(output[0][input_len:], skip_special_tokens=True).strip()


# ============================================================
# Chat function
# ============================================================

def chat_fn(message, history, user_name):
    start = time.time()
    message = safe_text(message)

    # Layer 1: hard crisis keyword — bypass generation entirely
    if contains_crisis_language(message):
        log_conversation_async(user_name, message, CRISIS_MESSAGE, "crisis")
        return CRISIS_MESSAGE

    normalized_history = normalize_history(history)

    # Layer 2: structural / conversation-level risk — still generate, but with
    # greedy decoding for consistency
    elevated_risk, risk_reason = assess_conversation_risk(normalized_history, message)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + normalized_history
    messages.append({"role": "user", "content": message})

    response = run_generation(messages, elevated_risk)

    # Layer 3: output-side check — screens what the MODEL generated, independent
    # of whether the input was flagged
    if output_flagged_as_unsafe(response):
        response = FALLBACK_RESPONSE
        risk_reason = "output_flagged"

    final_risk_label = risk_reason if (elevated_risk or risk_reason == "output_flagged") else "none"
    log_conversation_async(user_name, message, response, final_risk_label)
    print(f"[{time.time()-start:.1f}s] done (risk={final_risk_label})")
    return response


# ============================================================
# Gradio interface
# ============================================================

name_box = gr.Textbox(label="Your name (for logging)", placeholder="optional")

demo = gr.ChatInterface(
    fn=chat_fn,
    additional_inputs=[name_box],
    title="Iroko — Mental Health Support Chatbot",
    description=(
        "Fine-tuned Gemma 4 E2B model. This is a research/testing interface, "
        "not a substitute for professional mental health care. "
        "Conversations are logged for review purposes."
    ),
    examples=[
        ["I've been feeling really overwhelmed with school and I don't know how to cope.", "Guest"],
        ["I had a huge fight with my best friend and I don't know if we're okay anymore.", "Guest"],
        ["I can't sleep because I keep thinking about everything that could go wrong.", "Guest"],
    ],
)

if __name__ == "__main__":
    demo.launch(ssr_mode=False)
