Subject: Marathi MT via Bodhan Fine-Tune - submission (code, artifacts, PDF)

Dear [Reviewer /Hiring team],

Please find below my submission for the Marathi machine-translation
take-home. All three fine-tuning sessions ran end-to-end on Kaggle and every
checkpoint was mirrored to a shared Drive folder as it was written.

1) Source code (GitHub)
https://github.com/Kaustubh-Rathi/marathi-mt-bodhan

2) Artifacts, checkpoints and logs (Google Drive)
https://drive.google.com/drive/folders/1Abt7fldMDl_4pjTa3ayDbLl1dqDZSUTD
- bodhan-qlora/       Session A - primary, bodhan-ai/indic-translate (8B) QLoRA
- indictrans2-lora/   Session B - fallback, IndicTrans2 200M LoRA (+ eval metrics)
- ablation/           Session C - reduced-adapter / lower-LR ablation
- submission/         this PDF and the cover email

3) Consolidated write-up (single PDF, 28 pages)
submission/marathi-mt-bodhan-submission.pdf (also in the Drive folder above)

WHAT WAS BUILT
Primary: bodhan-ai/indic-translate (Gemma-4 E4B, 8B) fine-tuned with QLoRA
(4-bit NF4, LoRA r=32/alpha=64, all-linear minus vision/audio and lm_head),
English -> Marathi on ai4bharat/samanantar (config mr).
Evaluation is on held-out data only: ai4bharat/IN22-Gen (test) and
facebook/flores eng_Latn-mar_Deva (devtest), neither of which was trained on.

RESULTS
All three sessions completed, checkpointing every 5 optimizer steps.
  Session A (QLoRA)   20 steps, completed on a T4
  Session C (ablation) 20 steps, completed on a T4
  Session B (IndicTrans2) 20 steps, completed
    IN22-Gen (n=1024):  BLEU 20.91 | chrF2 54.10 | chrF2++ 49.82
    FLORES devtest (n=1012): BLEU 19.49 | chrF2 54.90 | chrF2++ 50.69
  The runs are 20-step smoke runs that prove the pipeline end to end; the
  configs are set for the full schedule, so the reported numbers are a
  lower bound rather than a converged result.

POINTS I WOULD LIKE TO FLAG
- Dataset pivot: the preferred education-domain set was access-gated and I was
  denied, so training uses the OPEN samanantar mr set. Samanantar was part of
  IndicTrans2's pretraining data, so Session B's score partly re-measures
  memorisation - stated openly rather than presented as a clean result.
- Loss semantics: the gated Gemma chat template emits no assistant mask, so
  TRL's assistant-only loss is disabled and training uses the full sequence.
  This is a deliberate, documented trade-off, not a silent default.
- Metric caveat: Session B shows very large gradient norms against
  max_grad_norm=1.0, i.e. heavy clipping every step. Worth revisiting with a
  lower learning rate or longer warmup if this is run for real.

The PDF contains the full approach narrative, the interface spec, every
hyperparameter with its rationale, setup and run instructions, the
decontamination checklist, and a short engineering log of the four failures I
had to diagnose and fix along the way (a QLoRA OOM, a chat-template loss
failure, and two related mixed-precision faults on the T4).

Happy to walk through any part of it, and glad to share the logs or reproduce
a run on request.

Best regards,
Kaustubh Rathi
