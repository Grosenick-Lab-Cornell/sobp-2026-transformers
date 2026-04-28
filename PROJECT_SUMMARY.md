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
- **Provider-agnostic:** Gemini 2.5 Flash is default (available via Google Cloud), but wrapped in `call_llm()` abstraction with commented-out Claude / OpenAI alternatives so audience sees this is about concepts, not vendor promotion.

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
| Model | Gemini 2.5 Flash (mid-tier, shows failure modes) |
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

**Gemini 2.5 Flash and lost-in-the-middle (NB1 §4 demo).** Recent literature (McKinnon 2025, arxiv 2511.05850) shows Gemini 2.5 Flash specifically does NOT exhibit the canonical lost-in-the-middle effect on simple needle-in-haystack retrieval, even at the input context limit. More broadly, the Chroma "Context Rot" report (2025) confirms frontier models pass simple NIAH but fail on harder long-context tasks: non-lexical matching (NoLiMa), multi-hop reasoning, conflict detection, distractor handling, absence detection.

*Why we're still OK:* our demo task is multi-hop + conflict-detection across multiple documents (find the SSRI ADR across docs 9–12, recognize the Plan in doc 24 contradicts it, recommend an alternative). This is the harder kind of task that frontier models still fail on. Simple retrieval would be the wrong demo; we don't have that problem.

*Rehearsal escape hatches if Flash blows through anyway, in order:* (a) make the needle subtler (e.g., remove the explicit ADR table from Doc 23 so the only signal is in the discharge summary narrative); (b) drop to Gemini 1.5 Flash for that one cell; (c) pivot live: "the model got this right — but you can't know it got it right without checking, which is the deeper point and what the rest of this talk is about."

**Hallucination demo prompt (NB2 §1).** Gemini Flash output varies. Per NB2 spec, ship 3–4 candidate prompts and lock the most reproducibly-failing one at rehearsal. Pedagogical fallback: if Flash answers correctly, the lesson becomes "you can't tell that without the corpus" rather than "look how it fabricates."

## Outstanding / deferred

- Build NB1 (skeleton + cells) — next up
- Build NB2 (skeleton + cells)
- Build landing page
- **Visual assets** — header illustration, 2–3 section dividers, 1–2 conceptual diagrams (Excalidraw/rough.js)
- Rehearsal testing — verify the chart task fails appropriately on Flash; verify the hallucination prompt fails reproducibly
