# HYPERPARAMETERS — what we run, why, and when to change it

**Single source of truth: `configs/`.** `base.yaml` holds every training/eval
knob; `session{A,B,C}_*.yaml` override only what differs. Python code contains
only safe fallbacks (`dict.get` defaults) — if a value matters, it lives in
YAML. To change anything below, edit the YAML, never the code.

## Session A — Bodhan Gemma-4 (8B) QLoRA (primary)

Provenance: the combination is mirrored from a recipe already proven on **this
exact model + dependency stack** (transformers 5.13.1 / trl 1.6.0
/ peft 0.20.0 on a Kaggle T4; 8/10 runs converged, COMET ≈ 74.5 reference). We
change only what the assignment forces (Marathi target, Samanantar en→mr data —
the preferred education-domain set, `coild-aikosh/Education_v2`, was
manual-gated and access was denied).

| Knob (config path) | Value | Why this value | When/how to shift |
| --- | --- | --- | --- |
| `model.load_in_4bit` + `bnb.quant_type: nf4` + `double_quant: true` | on | 8B on a 16GB T4 is impossible in fp16 (16GB weights alone). NF4+double-quant is the standard QLoRA recipe (Dettmers et al., 2023) — ~5GB weights leaves room for activations+optimizer. | No alternative on T4. |
| `bnb.compute_dtype: bfloat16` | bf16 | bf16 has fp32-range exponent → no overflow NaNs (unlike fp16). | T4 (sm_75) has no native bf16: on NaN/bnb dtype errors flip the documented fp16 path (`training.bf16:false`, `fp16:true`, `compute_dtype: float16`) — see SETUP.md §4. |
| `lora.r: 32`, `lora.alpha: 64` | 32/64 | alpha/r = 2 keeps the proven effective adapter LR; r=32 is the sweet spot for domain-adapting an 8B on ~8k pairs (r=16 underfits slightly, r=64 gains nothing at this data size and costs VRAM). Proven values. | Session C ablates r=16/alpha=32. Raise r only if dev loss plateaus far above train loss AND more data is available. |
| `lora.target_modules: all-linear` + `exclude_modules` (vision/audio) | all text linears | Attention-only LoRA underperforms all-linear for MT style transfer; vision/audio towers are excluded (text-only task) — saves memory, protects encoders. | Fixed. |
| `lora.modules_to_save: [lm_head]` | lm_head | Devanagari target distribution shift; training the head lets Marathi token logits adapt. (`embed_tokens` is listed in base.yaml but deliberately stripped in code — Gemma ties input/output embeddings.) | Fixed. |
| `lora.dropout: 0.05` | 0.05 | Light regularization for 8k rows. | Raise to 0.1 if dev loss diverges from train. |
| `training.learning_rate: 1e-4` | 1e-4 | Proven on this stack; QLoRA typical band is 1e-4–2e-4. | NaN/spikes → 5e-5. Plateaued high loss → 2e-4. |
| `lr_scheduler_type: cosine_with_min_lr` + `min_lr_rate: 0.1` | cosine, 10% floor | Smooth decay for convergence without killing tail-end learning; proven. | Fixed. |
| `warmup_ratio: 0.03` | 3% (30 steps) | Standard short-run warmup; stabilizes 8-bit Adam without burning budget. transformers 5.x folded this into `warmup_steps` (float in `[0, 1)` = ratio) and deprecates `warmup_ratio`, so Session A forwards the ratio through `warmup_steps` (`_warmup_value`). | Fixed. |
| `max_seq_length: 1024` | 1024 | Rows are ≤100 words (prepare filters), so 1024 covers p99 + template overhead; attention cost grows steeply with length on T4. | OOM → 768 (near-zero truncation at the 100-word cap). |
| `per_device_train_batch_size: 1` × `gradient_accumulation_steps: 16` | eff. 16 | VRAM-bound at 8B/4-bit/seq-1024; eff 16 gives stable gradients. | VRAM headroom >2GB → bs=4/accum=4 (same eff 16), ~10-20% faster steps. Only after smoke test. |
| `max_steps: 1000` (= 2 epochs of 8k) | 1000 | 1000 × 16 = 16k samples = 2 epochs — proven; a 3rd epoch overfits 8k rows. | Deadline pressure → 600-800 (checkpoints every 100, stop anywhere). |
| `save_steps/eval_steps: 100`, `logging_steps: 10` | 100/100/10 | A 12h Kaggle kill must cost ≤ 1 save interval; eval-at-save gives the selection curve free. | Do NOT raise save_steps (TRAINING.md). |
| `optim: paged_adamw_8bit` | paged 8-bit | QLoRA-standard: pages states to CPU on VRAM spikes, 8-bit halves optimizer memory. | Fixed. |
| `gradient_checkpointing: true`, `use_cache: false` | on | Required to fit; ~30% step slowdown is the price. | Fixed on T4. |
| `packing: false` | off | TRL 1.6 packing is incompatible with `assistant_only_loss`; losing assistant masking hurts more than packing helps at seq 1024 with short rows. | Fixed. |
| `assistant_only_loss: true` | on | Loss on Marathi answer tokens only — the prompt is an instruction; training on it wastes capacity and teaches prompt-echoing. Needs `{% generation %}` template markers (fallback documented in `_load_chat_dataset`). | Fixed. |
| `weight_decay: 0.01`, `max_grad_norm: 1.0`, `seed: 42` | defaults | HF-standard, proven; shared seed for cross-session comparability. | Fixed. |

## Session B — IndicTrans2 en-indic-dist-200M LoRA

| Knob | Value | Why |
| --- | --- | --- |
| `model.name` | indictrans2-en-indic-dist-200M | En-Indic distilled checkpoint (200M): native English source → Marathi, matching the `eng_Latn->mar_Deva` Samanantar train direction. Gated `auto` — one-time accept. |
| `lora.r:16/alpha:32/dropout:0.1`, targets `q_proj,k_proj` | small | 200M seq2seq needs far less adapter capacity than 8B; q/k-only mirrors known-good IndicTrans2 LoRA fine-tunes; heavier dropout suits the small model. |
| `learning_rate: 2e-4`, `inverse_sqrt`, `warmup_steps: 1000` | — | inverse_sqrt+warmup is the classic Noam-style NMT schedule; 2e-4 suits LoRA on a 200M. Session B config sets `warmup_steps: 1000` and `warmup_ratio: 0` (explicit step warmup, not the base 0.03 ratio). |
| `max_seq_length: 256` | 256 | Sentence-level MT; IndicTrans2 convention. |
| `per_device 8 × accum 16` (eff 128) | 128 | Standard NMT token-batch equivalent for stable seq2seq gradients. |
| `max_steps: 3000`, save/eval 500 | — | 3000 × 128 ≈ 48 passes over 8k — small model, needs many epochs; small checkpoints, so 500-step cadence is cheap. |
| `fp16: true`, eager attention, `dataloader_num_workers: 0` | — | Documented stability guards for the IndicTrans2 random-segfault issue (upstream #117); do not "optimize" these away. |
| `predict_with_generate: false` | off | Beam-5 generation every 500 steps over ~1000 dev rows wastes 20-50 min/run; loss-only eval suffices for model selection. Real BLEU/chrF++ come from `mr_mt.evaluate`. |

## Eval (both families) — `eval:` in base.yaml

| Knob | Value | Why |
| --- | --- | --- |
| `num_beams: 5` | 5 | MT-benchmark standard; comparable to published IN22/FLORES numbers. |
| `max_new_tokens: 256` | 256 | Matches the 100-word row cap with margin. |
| `batch_size: 8` | 8 | T4 4-bit decode sweet spot; sources are length-sorted before batching (less padding, ~15-30% faster). |
| metrics | sacrebleu BLEU + chrF++ (`word_order=2`) | chrF++ is robust for morphologically rich Marathi; BLEU for comparability; signatures printed for reproducibility. |

## Quick shift table (if the smoke test / run says X)

| Symptom | Change (YAML only) |
| --- | --- |
| OOM at step 1 | `max_seq_length: 768` (bs stays 1) |
| NaN/inf loss on T4 | fp16 path per SETUP.md §4; if persists, `learning_rate: 5e-5` |
| Train loss flat, dev flat-high | `learning_rate: 2e-4` OR `max_steps: 1500` — pick one |
| Dev loss rising while train falls | `lora.dropout: 0.1`; pick an earlier checkpoint (saved every 100) |
| Not enough time | `max_steps: 600`, evaluate checkpoint-600 — do not touch LR |
| VRAM headroom >2GB | `per_device_train_batch_size: 4`, `gradient_accumulation_steps: 4` (same eff. 16) for ~10-20% faster steps |

