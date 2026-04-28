# SOBP Session 1 Talk — Project Summary

**Speaker:** Logan Grosenick (Weill Cornell Psychiatry)
**Venue:** SOBP, Session 1
**Duration:** 45 min
**Official description:** *"Session 1 introduces LLM fundamentals—tokenization, context windows, retrieval augmented generation, schema constrained outputs—and maps them to common psychiatric use cases while emphasizing failure modes (hallucinations, leakage) and mitigations."*
**Next session in series:** foundation models + transformers (different speaker — this talk sets up for it).

---

## Format

- Talk is the primary deliverable. Colab notebooks are the live demo mechanism (run during talk).
- Aesthetic reference: [Cosyne 2025 Transformers in Neuroscience Tutorial](https://cosyne-tutorial-2025.github.io/) — clean, domain-grounded, narratively structured, finished-feeling. Match the overall vibe, not the depth (4-hr hands-on ≠ 45-min talk).
- **Provider-agnostic:** **Phi-3.5 mini Instruct** (Microsoft, 3.8B params, 128K context) is default — runs locally on Colab's free T4 GPU at 4-bit quantization, no API key, no auth, no setup beyond a 1–2 min model download on first run. Wrapped in `call_llm()` abstraction with commented-out Gemini / Claude / OpenAI alternatives so audience sees this is about concepts, not vendor promotion. Small open-source model is a *better* default for the failure-mode demos because the mechanisms (lost-in-middle, hallucination, schema slop) show up reliably at small scale; frontier models hide them intermittently.

---

## Structure: two notebooks

**Notebook 1 — "How LLMs see clinical text"** (data representation, ~18 min of talk)

- §0 Title + framing
- §1 Tokenization: model sees sub-word fragments, not words. Live-tokenize clinical sentence.
- §1b Brief mention of tokenization-free research: Meta BLT (2412.09871), H-Net (2507.07955). H-Net gives a one-line SSM-as-alternative-to-transformers teaser without a detour.
- §2 Unification move (temporaldata-inspired): same clinical fact in 4 formats (prose, structured med list, JSON/RxNorm, markdown table) all tokenize to the same kind of object.
- §3 Positional encoding: verbal only, "timestamp on each token" analogy, no math, no demo code.
- §4 Context windows + long-chart needle demo.
- §5 Teaser cell pointing at next talk (attention).

**Notebook 2 — "Using LLMs for psychiatric research"** (applied, ~20 min of talk)

- RAG (grounded vs. ungrounded)
- Schema-constrained output (structured extraction)
- Failure modes + mitigations synthesis

*Currently being outlined — see next deliverable.*

---

## Key decisions made

| Decision | Resolution |
|---|---|
| Talk length | 45 min |
| Colab use | Run live during talk |
| Model | Phi-3.5 mini Instruct (Microsoft, 3.8B, 128K context, runs locally on Colab T4 — no API key, demonstrates failure modes reliably) |
| Wrapper | Provider-agnostic `call_llm()` |
| Clinical data | Fully synthetic — no real patient data, even de-identified |
| Transformer demo in NB2 | Cut (defer to next talk's speaker) |
| SSMs | Name-drop only, via H-Net mention |
| Concepts prioritized | Tokenization, positional encoding, context — these three are the "minimum viable mental model" |
| Long-chart needle | SSRI-induced SIADH/seizure in 2022, current admission Plan re-proposes sertraline |
| Safety catch in Doc 25 (HD1 progress note) | Removed — chart contains no in-chart correction, maximum demo teeth |
| Chart length | ~12.5k words. Pad to 20–25k only if rehearsal shows frontier Flash blowing through it. |

---

## Deliverables produced

| File | Status | Purpose |
|---|---|---|
| `synthetic_clinical_content_DRAFT.md` | Complete | Original draft: tokenization sentence, 4-format unification example, original ~3.5k-word H&P. Source for reviewer red-penning. |
| `synthetic_clinical_content_ADDENDUM.md` | Complete | Drafts of 27 additional documents (outpatient notes, 2022 admission + DC summary, progress notes, consults, med rec, nursing notes). Source for reviewer red-penning. |
| `whitfield_chart_FULL.md` | Complete | **Canonical chart file.** 28 documents, chronological order, 12.5k words. Doc 25/26 safety catches removed. Ready to load into Colab. |
| Notebook 1 (`.ipynb`) | Not yet built | Awaiting NB2 outline + talk flow approval |
| Notebook 2 (`.ipynb`) | Not yet built | Awaiting NB2 outline + talk flow approval |
| Landing page (`.html` or similar) | Not yet built | Cosyne-style shareable page linking both Colabs |

---

## Audience

SOBP skews clinical/translational psychiatry — MDs, PhDs in biological psychiatry, neuroimaging, genetics, pharmacology. Most have used ChatGPT but haven't thought carefully about mechanics. Sweet spot: mental models sturdy enough to evaluate vendor claims and reason about when LLMs are (un)safe for a given use case.

---

## NB2 decisions (locked)

| Decision | Resolution |
|---|---|
| RAG corpus topic | Ketamine/esketamine for TRD |
| RAG corpus papers | Berman 2000, Zarate 2006, Murrough 2013, Daly 2018, McIntyre 2021 — abstracts/key paragraphs baked into notebook as plain strings (zero live-day fetch risk) |
| Schema extraction input | Whitfield chart Doc 24 (continuity with NB1) |
| Prompt injection demo | Build the cell, decide at rehearsal whether to run live |
| Hallucination prompt | Iterate at build-time; ship 3–4 candidate prompts in a "pre-talk rehearsal" cell, Logan locks the most reproducibly-failing one |

## Visual aesthetic (locked)

Inspiration: Cosyne 2025 tutorial header illustrations — looks hand-drawn but isn't, conveys "made with care."

| Element | Plan |
|---|---|
| Production tool | Excalidraw / rough.js (free, browser-based, wobbly-line aesthetic, exports SVG) |
| Landing page | Header illustration — sets the tone for the whole talk |
| Section dividers | 2–3 across the notebooks (likely 1 for NB1, 1–2 for NB2) |
| Conceptual diagrams | 1–2 — top candidates: tokenization fragmentation sketch, lost-in-the-middle / chart-with-needle schematic |
| Cell-level figures | Still none — keep cells clean and figure-light per earlier decision |

## Known risks for rehearsal

**Phi-3.5 mini and the needle demo (NB1 §4).** With a 3.8B-param model on a 17K-token chart, the failure-rate risk inverts from the original Gemini Flash plan: instead of "model might beat the needle," now the worry is "model fails everywhere uniformly, undermining the lost-in-the-middle pattern." For the demo to land we want roughly: succeeds at start/end, fails in middle. Pure failure across all three positions reads as "small model is just bad" rather than "look at how attention degrades positionally."

*Rehearsal escape hatches in order:* (a) shorten the chart by removing the long verbose nursing notes (Docs 27–28) — keep the chart well above the model's effective attention window but reduce overall noise; (b) make the needle slightly more salient by leaving the explicit ADR table in Doc 23; (c) pivot the framing live: even uniform failure is teachable as "this is what happens when context exceeds what the model can actually integrate."

**Schema-constrained extraction (NB2 §2).** Phi-3.5 mini via `outlines` should produce structurally valid JSON every time, but the *semantic* fidelity (correct ICD codes, correct dose values, correct ADR severity classification) will be wobbly. Rehearsal task: run the extraction 3–4 times, look for the kind of subtle wrongness (right structure, wrong content) that *makes* the cell's teaching point about "schema doesn't validate correctness." If the output is too obviously broken, dial up the prompt's "be conservative; only extract what is explicitly documented" framing.

**Hallucination demo prompt (NB2 §1).** Small models tend to fabricate citations more eagerly than frontier models — this is good for the demo. Per NB2 spec, ship 3–4 candidate prompts and lock the most reproducibly-failing one at rehearsal. Phi-3.5 may go further than Flash would: be ready for citations to entirely fictional papers (good demo material).

**Inference speed on stage.** A 17K-token input on Phi-3.5 mini at 4-bit on a Colab T4 takes ~30–60 sec per generation. The needle demo (3 calls) is 1.5–3 min of compute. Plan the talk pacing accordingly: kick the cell, then talk through the conceptual setup while it runs.

## Outstanding / deferred

- Build NB1 (skeleton + cells) — next up
- Build NB2 (skeleton + cells)
- Build landing page
- **Visual assets** — header illustration, 2–3 section dividers, 1–2 conceptual diagrams (Excalidraw/rough.js)
- Rehearsal testing — verify the chart task fails appropriately on Flash; verify the hallucination prompt fails reproducibly
