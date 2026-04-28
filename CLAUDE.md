# CLAUDE.md

Operational brief for Claude Code sessions on this repo. Read this **before** the README. The README explains *what* the project is; this file explains *how to work on it*.

---

## What this is

A 45-minute talk for **SOBP 2026 Session 1**: *Transformer & LLM Basics with Relevance to Psychiatry*. Speaker: Logan Grosenick, Weill Cornell Psychiatry. Two live Colab notebook demos. Audience: clinical and translational psychiatrists. The sweet spot is mental models for evaluating vendor claims, not an ML crash course.

The notebooks are demoware, to be run live during the talk and shared as a learning resource afterward. They are not a tutorial, not a library, not production code.

---

## How to work with me on this

- Use auto mode by default. Don't ask permission for routine, in-scope edits. Do ask before anything in the "Things never to do" list below, anything that touches a locked decision, or anything outside the immediate task.
- Write naturally and concisely. Short sentences are fine. Don't pad responses with restating the question, recapping what you just did, or summarizing what you're about to do.
- Don't be sycophantic. Skip "great question," "excellent point," "you're absolutely right." If something I propose is wrong or worse than the alternative, say so and explain why.
- Think critically. Push back when you disagree. Flag risks I haven't named. Notice when I'm asking for the second-best version of something. Don't just execute uncritically.
- Don't use em-dashes (—) or en-dashes (–). Use commas, parentheses, semicolons, colons, or new sentences instead. This applies to everything: code comments, markdown cells, chat replies, commit messages.
- When you're uncertain, say so. "I don't know if X works on this transformers version, want me to test?" beats guessing and breaking the demo.
- When something fails, report what actually happened, not what should have happened. Paste the error.

---

## Decisions that are locked (do not relitigate)

- **Two notebooks**, not one. NB1 = "How LLMs see clinical text" (tokenization, positional encoding, context). NB2 = "Using LLMs for psychiatric research" (RAG, schema extraction, failure modes).
- **Phi-3.5 mini Instruct** (3.8B, 4-bit on Colab T4) is the model. **No API providers.** Do not propose adding Gemini, OpenAI, Anthropic, or Vertex AI back into the working code. The provider-agnostic `call_llm()` wrapper has commented-out alternatives; leave them commented.
- **Whitfield chart** (~12.5k words, 28 documents, in `content/whitfield_chart_FULL.md`) is the long-context demo content. The clinical needle is sertraline-induced SIADH and seizure in 2022; the current admission's Plan re-proposes sertraline. Do not edit the chart content. Do not add an in-chart safety catch; the contradiction must stand for the demo to land.
- **RAG corpus** = 5 real open-access ketamine and esketamine TRD papers (Berman 2000, Zarate 2006, Murrough 2013, Daly 2018, McIntyre 2021). Abstracts baked in as Python strings. Do not swap for synthetic content or different papers.
- **Schema extraction** input = Document 24 of the chart (current admission H&P). Pydantic schema design is locked.
- **Visual aesthetic** is locked: Breip handwriting font (embedded as base64), electric teal `#06B6D4` accent, `#EDEDED` ink on `#1e1e1e` paper (Colab dark mode match). Do not redesign visuals. PNG is the embed format in notebooks; SVG is source of truth.
- **Talk title**: *Transformer & LLM Basics with Relevance to Psychiatry*. Not "Generative AI..."; the title was rebranded. Per-notebook titles ("How LLMs see clinical text" and "Using LLMs for psychiatric research") are notebook scopes, not the talk title; leave them alone.

---

## File conventions

- **Notebooks** live in `notebooks/`. Image embeds use raw GitHub URLs: `https://raw.githubusercontent.com/Grosenick-Lab-Cornell/sobp-2026-transformers/main/visuals/<filename>.png`. Use PNG, not SVG (Jupyter renderers are inconsistent on SVG).
- **Specs** in `specs/` are the chat-side design source of truth. If the implementation drifts from a spec, ask which one is right rather than silently aligning.
- **Visuals** in `visuals/` are dark-mode by design. If asked to make a light-mode version, generate alongside, don't replace.
- **Synthetic clinical content** (chart, drafts) lives in `content/`. Synthetic, but treated as clinically plausible. Corrections welcome, fabrications not.

---

## Things never to do

- Do not introduce new pip dependencies without asking. The current set in `requirements.txt` is what gets installed in Colab; adding more increases install time and breakage surface.
- Do not unpin `requirements.txt` versions. They are pinned for a reason (`outlines`, `bitsandbytes`, and `transformers` all have moving APIs that have already broken cells once).
- Do not "improve" the notebooks by adding cells, more explanation, or more examples. They are sized to a 45-minute talk budget. More is not better.
- Do not redesign visuals to be cleaner, more modern, or more professional. The hand-drawn aesthetic is intentional and matches the Cosyne 2025 reference (`https://cosyne-tutorial-2025.github.io/`).
- Do not promote any specific commercial LLM vendor in the framing. The talk is concept-first.
- Do not add tests, CI, pre-commit hooks, formatters, linters, or `pyproject.toml`. Demoware.

---

## When in doubt

Ask before changing anything in the locked list. "I'd like to do X, is it OK?" beats doing it and explaining afterward. The cost of asking is one round trip; the cost of fixing drift is a session and a half.

---

## Known risks (verify at rehearsal, do not paper over)

1. **Phi-3.5 might handle the chart too well.** If the needle demo in NB1 §4 succeeds in all three positions (start, middle, end), the demo's pedagogical force collapses. Escape hatches in `PROJECT_SUMMARY.md` § "Known risks for rehearsal." Do not just declare it working; actually compare the three outputs.
2. **Phi-3.5 hallucinates more crudely than frontier models.** This is fine for the RAG demo in NB2 §1 if the framing acknowledges it. The pedagogical point is the *mechanism*, not the specific way one model fabricates. If the RAG demo fails to land, the answer is reframing, not switching models.
3. **`outlines.from_transformers` API has shifted recently.** If the structured-output cell in NB2 §2 breaks on a new `outlines` release, pin to the working version rather than rewriting the demo.
4. **Transformers v5 (released Dec 2025) is a breaking change.** `requirements.txt` pins to v4. Do not upgrade casually.
