# SOBP Session 1 Talk — Project Repo

**Speaker:** Logan Grosenick (Weill Cornell Psychiatry)
**Venue:** SOBP, Session 1
**Length:** 45 minutes
**Topic:** *Generative AI and LLM Basics with Relevance to Mental Health*

---

## What this repo is

This is the working repo for a 45-minute talk that introduces LLM fundamentals (tokenization, context windows, RAG, schema-constrained output, failure modes) to a clinical/translational psychiatry audience, with two live Colab notebook demos.

Content design (specs, synthetic clinical data, visual assets) is **complete**. What's left is engineering: convert the specs to working `.ipynb` files, verify against a real Vertex AI project, host on GitHub Pages, rehearse.

---

## Where things live

```
sobp-talk/
├── README.md                          ← you are here
├── PROJECT_SUMMARY.md                 ← all decisions made, locked vs. open
├── specs/
│   ├── NB1_SPEC.md                    ← Notebook 1 spec, cell-by-cell
│   └── NB2_SPEC.md                    ← Notebook 2 spec, cell-by-cell
├── content/
│   └── whitfield_chart_FULL.md        ← the 12.5k-word synthetic chart used in NB1 §4
└── visuals/
    ├── header.{svg,png}               ← landing page header illustration
    ├── tokenization.{svg,png}         ← NB1 §1 conceptual diagram
    ├── lost_in_middle.{svg,png}       ← NB1 §4 conceptual diagram
    └── divider_0{1,2,3}_*.{svg,png}   ← three section divider illustrations
```

PNG + SVG of each visual; SVG is the source of truth, PNG is for quick mobile preview. SVGs have the Breip handwriting font embedded as base64 — fully portable, no font dependencies.

---

## What's locked (don't redesign without good reason)

- **Two notebooks**, not one. NB1 = "How LLMs see clinical text" (tokenization, positional encoding, context). NB2 = "Using LLMs for psychiatric research" (RAG, schema extraction, failure modes).
- **Provider-agnostic `call_llm()` wrapper** with Gemini 2.5 Flash as the default backend (via Vertex AI). Commented Claude/OpenAI alternatives. Audience sees this is concept-first, not vendor-promotion.
- **Whitfield chart** as the long-context demo content. SSRI-induced SIADH/seizure in 2022 is the documented contraindication; current admission's Plan re-proposes sertraline. ~12,500 words, 28 documents, no in-chart safety catch (max demo teeth).
- **RAG corpus** = 5 real open-access ketamine/esketamine TRD papers (Berman 2000, Zarate 2006, Murrough 2013, Daly 2018, McIntyre 2021), abstracts to be baked into NB2 as Python strings.
- **Schema extraction** input = Document 24 of the chart (current admission H&P).
- **Visual aesthetic**: zine-like, Breip handwriting + monospace tokens, electric teal `#06B6D4` accent on `#fdfbf4` paper, ink black `#1a1a1a`.

---

## What's still open

**Immediate (Claude Code work):**
- Convert NB1_SPEC.md → working `.ipynb` that runs end-to-end on Logan's Vertex AI project.
- Convert NB2_SPEC.md → working `.ipynb`.
- Resolve `google-genai` SDK API surface — code in specs is best-guess, will need adjustment to match installed SDK version. Especially the structured-output API (`response_schema` with Pydantic) which has been volatile.
- Replace the chart-loading placeholder URL (`https://[hosted-location]/whitfield_chart_FULL.md`) in NB1 with either a GitHub raw URL after pushing the repo, or paste the chart inline as a Python string.
- Build a simple GitHub Pages landing page (single HTML file) with `header.svg`, brief framing, and "Open in Colab" buttons for both notebooks.

**Rehearsal-time:**
- Verify Gemini 2.5 Flash actually fails (or partially fails) on the chart needle question. If it blows through cleanly, escape hatches in `PROJECT_SUMMARY.md` § "Known risks for rehearsal."
- Verify hallucination prompt fails reproducibly in NB2 §1. Three candidate prompts already drafted in NB2_SPEC.md; pick the most reliably-failing one.
- Decide whether to run NB2's prompt-injection demo live or just discuss in markdown. Cell built, decision deferred.

**Could add later:**
- Pad chart from 12.5k → 22k words if needle fails too easily on Flash. Source material exists in `synthetic_clinical_content_ADDENDUM.md` (not in this repo — left in the original drafting workspace).

---

## Quick context for Claude Code on first session

**Audience:** SOBP — clinical/translational psychiatrists. Most have used ChatGPT but not thought about mechanics. Sweet spot is mental models for evaluating vendor claims, not an ML crash course.

**Aesthetic reference:** Cosyne 2025 Transformers in Neuroscience tutorial (https://cosyne-tutorial-2025.github.io/) — clean, domain-grounded, narratively structured, finished-feeling. Match the vibe, not the depth (4-hr hands-on tutorial vs. 45-min talk + live demos).

**Pedagogical spine (NB1):** three concepts do most of the work — tokenization, positional encoding, context. Everything else (RAG, schema, failure modes in NB2) builds on these.

**Tone:** prose-first (not bullet-heavy), warm-formal, honest about what breaks.

---

## Suggested first task in Claude Code

**"Read PROJECT_SUMMARY.md and specs/NB1_SPEC.md. Then convert NB1_SPEC.md into a working `.ipynb` file at `notebooks/01_how_llms_see_clinical_text.ipynb`. For now, leave the chart-loading cell with the placeholder URL — we'll fix that once the repo is pushed to GitHub. After conversion, walk me through what you'd need to verify the SDK calls work against my Vertex AI project."**
