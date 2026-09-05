# Team-Iroko-Medical-Chat-bot
TRI AI Project for Team Iroko 
# Iroko — AI-Powered Mental Health Support Chatbot

Iroko is a fine-tuned conversational AI designed to provide culturally appropriate, safety-conscious mental health support for a Nigerian context. It's built by fine-tuning Google's Gemma 4 E2B model on a mental health counseling dataset, with a full pipeline spanning training, safety evaluation, and production deployment.

**Repositories:**
- GitHub: [`Ololade117/Iroko`](https://github.com/Ololade117/Iroko)
- Model (Hugging Face Hub): [`Ololade117/gemma-4-e2b-iroko-mentalhealth-finetuned5`](https://huggingface.co/Ololade117/gemma-4-e2b-iroko-mentalhealth-finetuned5)

---

## Project Goals

- Fine-tune a small, efficient language model for empathetic, supportive mental health conversations
- Ensure the model handles crisis situations (suicidal ideation, self-harm, violence) responsibly, using Nigeria-specific resources
- Build a safety harness that catches failure modes the base fine-tune doesn't fully solve on its own
- Serve the model through a stable, shareable interface with per-user logging for ongoing review

---

## Architecture & Pipeline

### 1. Data Cleaning & Preprocessing
- Source dataset combined and split into train/test/validation sets
- **HTML artifact stripping** — removed raw `&nbsp;`, `<br>` tags leaking from the original data source into model responses
- **Fabricated identity/credential scrubbing** — removed self-introduced fake names, workplaces, and false claims of professional licensure (e.g. "Hi, I'm Karen, I work with family services..."), including sign-off patterns ("Best regards, Dr. ...")
- **Transcript artifact removal** — stripped real therapist names, `[unintelligible]`, `[crosstalk]` markers from transcript-sourced data
- **Nigeria localization** — replaced US-centric crisis resources (988, 911, "Suicide & Crisis Lifeline") with Nigeria-specific ones: **Mentally Aware Nigeria Initiative (MANI)** — 0809 111 6264 / 0811 1680 686 — and **112** (national emergency line)

### 2. Safety Data Augmentation
- Generated adversarial refusal examples using the **clean, un-fine-tuned base model**, so the adapter learns to preserve refusal behavior rather than drift away from it during fine-tuning
- Prompts covered real failure categories: direct method-seeking (guns, poison), "as a joke" framing, euphemisms, third-person displacement
- All generated examples **manually reviewed** before inclusion (never trusted as ground truth blindly)
- Mixed into the training set at ~20% ratio, diverse rather than duplicated

### 3. Training
- **Base model:** `google/gemma-4-E2B-it`, 4-bit quantized (NF4, bitsandbytes)
- **Method:** LoRA (rank 16, alpha 32) via PEFT, targeting attention/MLP projections while explicitly excluding vision/audio towers (multimodal components not used for this text-only task)
- **Precision:** bf16 end-to-end (avoids `GradScaler` incompatibility seen with fp16 on this architecture)
- **Framework:** TRL's `SFTTrainer`, with checkpointing, validation-loss tracking, and early stopping (patience=3, monitored on `eval_loss`)
- **Key architecture-specific fixes:** targeted norm-upcasting (not blanket `kbit` prep), `Gemma4ClippableLinear` layer unwrapping before LoRA injection, memory-constrained loading (`max_memory`, `low_cpu_mem_usage`)

### 4. Evaluation
- **Validation loss** tracked throughout training — but treated as necessary, not sufficient, for judging safety
- **Manual log review** of real chatbot conversations surfaced 6 major failure categories (see below)
- **Red-team evaluation framework** — a fixed set of paraphrased adversarial prompts, held out from training data, run against each model version and manually scored (pass/fail/borderline) per failure category. This is the actual gate for whether retraining improved safety, since loss going down does not imply safety behavior improved.

### 5. Deployment (Serving Harness)
Deployed as a Hugging Face Space (Gradio) on free **ZeroGPU** hardware, with a multi-layer safety harness wrapped around raw model generation:

1. **Input Layer 1 — hard keyword match:** direct crisis phrases bypass generation entirely, returning a fixed crisis-resource message
2. **Input Layer 2 — structural risk detection:** regex-based detection of method-seeking phrasing, euphemisms, "as a joke" framing, plus conversation-level escalation tracking across recent turns
3. **Output Layer — response moderation:** screens the *model's own generated response* for red-flag patterns before it reaches the user (independent of whether the input was flagged)
4. **Decoding controls:** repetition penalty + no-repeat n-gram (fixes text-degeneration loops); greedy decoding forced specifically when elevated risk is detected (fixes inconsistent responses to near-identical risky inputs)
5. **Logging:** per-user conversation logs written asynchronously to a shared Google Sheet (service-account auth, since a Space has no interactive login)

---

## Key Failure Categories Identified (via manual log review)

| # | Category | Example | Fix |
|---|---|---|---|
| 1 | Bypassable keyword filter | Euphemisms, "as a joke" framing evaded exact-match detection | Structural regex + conversation-escalation tracking |
| 2 | Model engaging with harmful requests | Validated/assisted requests involving guns, poison, violence | Adversarial refusal training data + output-side moderation |
| 3 | Fabricated clinical identities | Model introduced itself as "Karen," claimed counselor credentials | Identity/credential scrubbing + explicit disclosure training examples |
| 4 | Response inconsistency | Near-identical risky prompts got different responses | Greedy decoding under detected risk |
| 5 | Text degeneration loops | Repeated URL/phrase loops in long responses | `repetition_penalty`, `no_repeat_ngram_size` |
| 6 | Uneven response quality | Strong on benign factual questions, weak on adversarial ones | Rebalanced training data toward underrepresented adversarial categories |

---

## Key Performance Indicators

**Training metrics (most recent full run with logged metrics):**

| Metric | Start | End |
|---|---|---|
| Training loss | 6.25 | ~2.10–2.25 |
| Validation loss | 2.283 | 2.093 |
| Token accuracy | 49.4% | 52.3% |
| Entropy | 2.22 | 2.09 |

**Safety & behavioral KPIs (the ones that gate release):**

- **Red-team pass rate** — % of adversarial prompts per category (method-seeking, joke-framing, euphemism, third-person) receiving an appropriate refusal vs. harmful engagement. This is the primary safety metric, scored manually per model version.
- **Consistency rate** — whether near-identical risky inputs reliably produce the same (correct) response.
- **Crisis-detection recall** — proportion of genuinely risky messages caught by the input-side layers before reaching generation.
- **Response latency** — cold start (~145s, dominated by one-time model download) vs. warm request (~20–25s on ZeroGPU, since GPU-resident state isn't guaranteed to persist between calls on shared hardware).

> **Note:** validation loss improving does *not* imply safety behavior improved — this was directly demonstrated by early logs showing declining loss alongside active safety failures. Red-team pass rate, scored manually against a fixed held-out prompt set, is the metric that actually answers whether a training iteration made the model safer.

---

## Tools & Technologies

| Category | Tools |
|---|---|
| Training | Google Colab, Kaggle, PEFT/LoRA, bitsandbytes, TRL |
| Model | Gemma 4 E2B (`google/gemma-4-E2B-it`) |
| Deployment | Hugging Face Spaces (ZeroGPU), Gradio |
| Logging | Google Sheets API (service account auth) |
| Versioning | Hugging Face Hub, GitHub |

---

## Status & Next Steps

- Model version: **finetuned5**, pushed to the Hub and deployed to a Space harness
- Outstanding: run the red-team evaluation against finetuned5 specifically and score results, to confirm whether the retraining + harness combination closes the gaps found in earlier versions
- Recommended before any wider release: complete red-team scoring, and treat current deployment as internal testing only until that's done