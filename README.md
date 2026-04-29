# SOBP Session 1 Talk, Project Repo

**Speaker:** Logan Grosenick (Weill Cornell Psychiatry)
**Venue:** SOBP, Session 1
**Length:** 45 minutes
**Topic:** *Transformer & LLM Basics with Relevance to Psychiatry*

---

## What this repo is

This is the working repo for a 45-minute talk that introduces LLM fundamentals (tokenization, context windows, RAG, schema-constrained output, failure modes) to a clinical/translational psychiatry audience, with two live Colab notebook demos.

Content design (specs, synthetic clinical data, visual assets) is **complete**. What's left is engineering: convert the specs to working `.ipynb` files, verify the local-Phi-3.5 setup runs end-to-end on Colab, host on GitHub Pages, rehearse.

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

PNG + SVG of each visual; SVG is the source of truth, PNG is for quick mobile preview. SVGs have the Breip handwriting font embedded as base64, fully portable, no font dependencies.

---

## What's locked (don't redesign without good reason)

- **Two notebooks**, not one. NB1 = "How LLMs see clinical text" (tokenization, positional encoding, context). NB2 = "Using LLMs for psychiatric research" (RAG, schema extraction, failure modes).
- **Provider-agnostic `call_llm()` wrapper** with **Phi-3.5 mini Instruct** (3.8B params, 128K context) as the default backend, running locally on Colab's GPU. No API key, no auth, no setup beyond a 1-2 min model download on first run. Commented Claude/OpenAI/Gemini alternatives. Audience sees this is concept-first, not vendor-promotion, and a small open-source model is *better* for these demos because the failure modes (lost-in-the-middle, hallucination, schema slop) are architecture-level mechanisms that show up more reliably at smaller scale.
- **Whitfield chart** as the long-context demo content. SSRI-induced SIADH/seizure in 2022 is the documented contraindication; current admission's Plan re-proposes sertraline. ~12,500 words, 28 documents, no in-chart safety catch (max demo teeth).
- **RAG corpus** = 5 real open-access ketamine/esketamine TRD papers (Berman 2000, Zarate 2006, Murrough 2013, Daly 2018, McIntyre 2021), abstracts to be baked into NB2 as Python strings.
- **Schema extraction** input = Document 40 of the chart (current admission H&P).
- **Visual aesthetic**: zine-like, Breip handwriting + monospace tokens, electric teal `#06B6D4` accent on `#fdfbf4` paper, ink black `#1a1a1a`.

---

## What's still open

**Immediate (Claude Code work):**
- ~~Convert NB1_SPEC.md → working `.ipynb`.~~ ✓ Shipped at `notebooks/01_how_llms_see_clinical_text.ipynb`.
- ~~Convert NB2_SPEC.md → working `.ipynb`.~~ ✓ Shipped at `notebooks/02_using_llms_for_psychiatric_research.ipynb`.
- ~~Replace the chart-loading placeholder URL.~~ ✓ Both notebooks point at `raw.githubusercontent.com/.../main/content/whitfield_chart_FULL.md`.
- ~~Build the GitHub Pages landing page.~~ ✓ Live at `docs/`, served from `https://grosenick-lab-cornell.github.io/sobp-2026-transformers/`.
- Verify Phi-3.5 mini load + generation work end-to-end on Colab's T4. Especially the `outlines` constrained-generation path in NB2 §2, `outlines.from_transformers(...)` API has shifted recently; may need a small adjustment. First test: open NB1 in Colab, set runtime to T4 GPU, run all cells.

**Rehearsal-time:**
- Verify Phi-3.5 mini fails the needle question in at least one position (start/middle/end), and that the failure pattern is interpretable as lost-in-the-middle rather than just "small model is confused." If it fails too uniformly, the demo loses its punch, rehearsal escape hatches in `PROJECT_SUMMARY.md` § "Known risks for rehearsal."
- Verify hallucination prompt fails reproducibly in NB2 §1. Three candidate prompts already drafted in NB2_SPEC.md; pick the most reliably-failing one.
- Decide whether to run NB2's prompt-injection demo live or just discuss in markdown. Cell built, decision deferred.

**Could add later:**
- Pad chart from 12.5k → 22k words if needle fails too easily on Flash. Source material exists in `synthetic_clinical_content_ADDENDUM.md` (not in this repo, left in the original drafting workspace).

---

## Quick context for Claude Code on first session

**Audience:** SOBP, clinical/translational psychiatrists. Most have used ChatGPT but not thought about mechanics. Sweet spot is mental models for evaluating vendor claims, not an ML crash course.

**Aesthetic reference:** Cosyne 2025 Transformers in Neuroscience tutorial (https://cosyne-tutorial-2025.github.io/), clean, domain-grounded, narratively structured, finished-feeling. Match the vibe, not the depth (4-hr hands-on tutorial vs. 45-min talk + live demos).

**Pedagogical spine (NB1):** three concepts do most of the work, tokenization, positional encoding, context. Everything else (RAG, schema, failure modes in NB2) builds on these.

**Tone:** prose-first (not bullet-heavy), warm-formal, honest about what breaks.

---

## Suggested first task in Claude Code

**"Read PROJECT_SUMMARY.md and `notebooks/utils.py`. The notebooks are shipped, both run end-to-end on Colab T4 with Phi-3.5 mini local (no API key, no GCP). Open NB1 in Colab, set runtime to T4 GPU, run all cells. Report back: do the visuals render against Colab's dark background, what does Phi-3.5 generate for the chart safety question, and do the three needle-position variants (start/middle/end) produce visibly different outputs. After that, do the same for NB2, flag if the `outlines.from_transformers(...)` schema-extraction call in §2 errors (likely fix is `outlines.models.transformers(...)` if the API has shifted)."**
