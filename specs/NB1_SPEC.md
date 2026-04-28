# Notebook 1 — "How LLMs see clinical text"
## Markdown spec for review before .ipynb assembly

**Convention used in this spec:**
- `[MD]` = markdown cell (the text quoted is what appears in the cell)
- `[CODE]` = code cell (the code shown is what's in the cell)
- `[OUTPUT]` = expected output, for sanity-checking — not part of the notebook
- `[REVIEWER NOTE]` = my notes to you, will not appear in the final notebook
- *italics in markdown cells* are real italics in the notebook

**Notebook target length:** ~15–18 minutes of live presentation, ~25 cells.

**Imports cell strategy:** one cell at the very top, hidden in the notebook UI as collapsed by default. Audience doesn't read import statements; you don't want them on screen.

---

## Cell 1 — Title and framing [MD]

> # How LLMs see clinical text
>
> *Notebook 1 of 2 — SOBP Session 1*
>
> The point of this notebook is not to teach you machine learning. It's to give you a sturdy enough mental model that, when a vendor shows you their psychiatric LLM product, you can reason about *what it can and can't do, and where it will break*.
>
> Three concepts do most of that work:
>
> 1. **Tokenization** — how the model represents the text you give it
> 2. **Positional encoding** — how it knows what order things came in
> 3. **Context** — what it actually "knows" at the moment it answers you
>
> We'll touch each one with a clinical example, then end with a demo of why these things matter when you ask an LLM to read a real chart.

[REVIEWER NOTE: I went with three concepts in the framing rather than the full description checklist (tokenization/context/RAG/schema/failure modes). NB1 is the foundations notebook and these three are the spine; NB2 is where RAG/schema/failure modes live. Talk-level framing in the opening 3 minutes will preview both notebooks. Flag if you want more here.]

---

## Cell 2 — Setup [CODE, collapsed by default]

```python
# Setup. Skip — pre-installs Google Cloud SDK, Pydantic, tiktoken, and configures
# Vertex AI auth from Colab. The model behind every LLM call below is Gemini 2.5 Flash,
# but the call is wrapped in `call_llm()` so you can swap providers in one line.

import os
from google import genai
from google.genai import types
import tiktoken

# Authenticate to Vertex AI from Colab.
# (For an attendee running this themselves: replace with your own auth.)
from google.colab import auth
auth.authenticate_user()

PROJECT_ID = "your-gcp-project"  # <-- replace at runtime
LOCATION = "us-central1"

client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)
MODEL = "gemini-2.5-flash"

def call_llm(prompt: str, system: str | None = None) -> str:
    """Provider-agnostic wrapper. Default backend: Gemini 2.5 Flash via Vertex AI.
    Swap the body of this function to use Anthropic, OpenAI, or a local model
    without touching anything else in the notebook.
    """
    config = types.GenerateContentConfig(system_instruction=system) if system else None
    response = client.models.generate_content(
        model=MODEL, contents=prompt, config=config,
    )
    return response.text
```

[REVIEWER NOTE: Two things to flag. (1) The exact `google-genai` API surface keeps shifting — the structured-output API in particular has changed names across versions. I'll re-verify against `https://googleapis.github.io/python-genai/` at .ipynb-build time. The above is my best-current-guess for the SDK API as of right now and may need a small adjustment. (2) PROJECT_ID is a placeholder — when you build the notebook for live use you'll set yours. I'll add a clear comment about this.]

---

## §1 — Tokenization

### Cell 3 — Section header [MD]

> ## §1. Tokenization: the model doesn't see words
>
> An LLM doesn't operate on words. It operates on **tokens** — integer IDs from a fixed vocabulary, where each ID corresponds to a sub-word fragment that the model's tokenizer learned during training.
>
> Two implications worth holding onto:
>
> - **Clinical shorthand fragments unpredictably.** Drug names, dosages, ICD codes, abbreviations, and typos all break in different ways.
> - **Numerical reasoning starts at a disadvantage.** "100mg" and "100 mg" are different token sequences. The model has to learn that they mean the same thing.

### Cell 4 — Live tokenization [CODE]

```python
# We'll use OpenAI's tiktoken to peek at GPT-style tokenization, since it's the
# easiest tokenizer to inspect cell-by-cell. Gemini's tokenizer is similar in
# spirit (BPE-family); the *specific* token boundaries differ, but the lessons
# transfer.

enc = tiktoken.get_encoding("cl100k_base")  # GPT-4 era tokenizer

clinical_sentence = (
    "Pt c/o ↑ anhedonia x 3wks, PHQ-9 = 19, h/o MDD (F33.1), "
    "currently on sertraline 100mg qAM + bupropion XL 300mg, "
    "denies SI/HI, sx refractory despite 8wk adequate trial."
)

token_ids = enc.encode(clinical_sentence)
tokens = [enc.decode([tid]) for tid in token_ids]

print(f"Sentence: {clinical_sentence}\n")
print(f"Word count:  {len(clinical_sentence.split())}")
print(f"Token count: {len(token_ids)}")
print(f"\nTokens (with sub-word boundaries shown as |):")
print("|".join(repr(t) for t in tokens))
```

[OUTPUT, expected — actual tokenization may shift]:

```
Sentence: Pt c/o ↑ anhedonia x 3wks, PHQ-9 = 19, h/o MDD (F33.1), currently on sertraline 100mg qAM + bupropion XL 300mg, denies SI/HI, sx refractory despite 8wk adequate trial.

Word count:  29
Token count: ~62

Tokens (with sub-word boundaries shown as |):
'Pt'|' c'|'/o'|' '|'↑'|' an'|'hed'|'onia'|' x'|' '|'3'|'w'|'ks'|','|' PH'|'Q'|'-'|'9'|' ='|' '|'19'|','|' h'|'/o'|' M'|'DD'|' ('|'F'|'33'|'.'|'1'|'),'|...
```

### Cell 5 — What just happened [MD]

> Take a look at what the tokenizer did:
>
> - **"sertraline"** got split — likely into something like `sert` + `raline`. The model never sees the word "sertraline" intact; it sees two arbitrary fragments and has to have learned during pretraining that this combination refers to a SSRI.
> - **"PHQ-9"** became three or four tokens (`PH`, `Q`, `-`, `9`). The "9" is a generic digit, not bound to the assessment.
> - **"F33.1"** fragmented similarly. The ICD-10 structure is invisible to the tokenizer.
> - **"100mg"** and **"100 mg"** would tokenize differently. Try it: change the spacing in the sentence and re-run.
>
> None of this is fatal. Modern models handle this routinely. But it explains a lot of small failures — miscounted dosages, mangled abbreviations, fragile performance on rare drug names — and it's the reason "garbage in, garbage out" is sharper for clinical text than for general prose.

### Cell 6 — Where this is going (research note) [MD]

> *A note for the research-curious.*
>
> Tokenization is a known weak spot, and there's active work to eliminate it. Two recent lines:
>
> - Meta's **Byte Latent Transformer** ([Pagnoni et al., 2024](https://arxiv.org/abs/2412.09871)) operates on raw bytes and learns to group them into variable-sized "patches" based on entropy of the next byte.
> - **H-Net** ([Hwang et al., 2025](https://arxiv.org/abs/2507.07955)) learns hierarchical chunking end-to-end from bytes; outperforms a strong BPE Transformer at matched compute, with the largest gains on Chinese, code, and DNA.
>
> Both eliminate the fixed vocabulary entirely. Notably, H-Net uses **state-space models** in its byte-level interface — an alternative to transformers that we won't get into today, but worth knowing the name (you'll hear "Mamba" and "SSM").
>
> None of the major commercial models you can use today (GPT, Claude, Gemini) take this approach yet. They're all BPE-tokenized. But the failure modes we're showing in this notebook are precisely what these lines of work are trying to remove.

[REVIEWER NOTE: ~20 sec of talk time on this cell. It's the SSM teaser plus tokenization-future hat-tip in one move.]

---

## §2 — The unification move (your temporaldata-inspired payoff)

### Cell 7 — Section header [MD]

> ## §2. Tokens are a universal adapter
>
> Clinical content arrives in wildly heterogeneous formats: free-text notes, structured medication lists, codes (ICD, RxNorm, LOINC), tables, lab panels, even pasted emails. To work with any of this, you'd normally need format-specific parsers.
>
> **Tokenization sidesteps that problem.** Whatever format you hand the model, it becomes the same kind of object — a sequence of token IDs with positions. The downstream machinery (attention, the next talk) operates on that unified representation indifferently.
>
> Same clinical fact, four ways:

### Cell 8 — The four formats [CODE]

```python
# Same clinical fact, expressed four ways. We'll tokenize all of them and ask
# the model the same question of each.

# Fact: A patient with recurrent MDD started sertraline 50 mg on 2024-03-15,
# titrated to 100 mg on 2024-04-10, with partial response (PHQ-9 18 → 12).

format_A_prose = """Ms. A is a 52-year-old woman with recurrent MDD who presented
on 3/15/24 with PHQ-9 of 18 and was started on sertraline 50 mg daily. At
follow-up on 4/10/24 she was titrated to 100 mg daily. By 5/1/24 her PHQ-9
had improved to 12, consistent with partial response."""

format_B_medlist = """MEDICATIONS — DEPRESSION
  Sertraline  50 mg PO daily   Start: 2024-03-15   Stop: 2024-04-10
  Sertraline 100 mg PO daily   Start: 2024-04-10   Stop: (active)

ASSESSMENTS
  PHQ-9   2024-03-15   Score 18
  PHQ-9   2024-05-01   Score 12"""

format_C_json = """{
  "diagnoses": [{"icd10": "F33.1"}],
  "medications": [
    {"rxnorm": "312940", "name": "sertraline", "dose_mg": 50,
     "start": "2024-03-15", "end": "2024-04-10"},
    {"rxnorm": "314277", "name": "sertraline", "dose_mg": 100,
     "start": "2024-04-10", "end": null}
  ],
  "assessments": [
    {"instrument": "PHQ-9", "date": "2024-03-15", "score": 18},
    {"instrument": "PHQ-9", "date": "2024-05-01", "score": 12}
  ]
}"""

format_D_table = """| Date       | Medication | Dose   | PHQ-9 |
|------------|------------|--------|-------|
| 2024-03-15 | Sertraline | 50 mg  | 18    |
| 2024-04-10 | Sertraline | 100 mg | —     |
| 2024-05-01 | Sertraline | 100 mg | 12    |"""

formats = {
    "A. Narrative prose": format_A_prose,
    "B. Structured med list": format_B_medlist,
    "C. JSON / coded": format_C_json,
    "D. Markdown table": format_D_table,
}

print(f"{'Format':<26} {'Words':>6} {'Tokens':>7}")
print("─" * 42)
for name, txt in formats.items():
    n_words = len(txt.split())
    n_tokens = len(enc.encode(txt))
    print(f"{name:<26} {n_words:>6} {n_tokens:>7}")
```

[OUTPUT, expected]:

```
Format                      Words  Tokens
──────────────────────────────────────────
A. Narrative prose             52      ~75
B. Structured med list         32      ~70
C. JSON / coded                52     ~120
D. Markdown table              28      ~80
```

[REVIEWER NOTE: The exact token counts will vary depending on tokenizer behavior; I've estimated. The teaching point is that they're all in the same order of magnitude and all fit easily in a context window. JSON is heaviest because of the punctuation overhead.]

### Cell 9 — Asking the same question of each [CODE]

```python
# Now: ask the model the same question of each format. Watch it answer correctly
# from any of them.

question = "What was this patient's PHQ-9 trajectory and what medication change accompanied it?"

for name, txt in formats.items():
    prompt = f"{txt}\n\n---\n\nQuestion: {question}\nAnswer in one sentence."
    answer = call_llm(prompt)
    print(f"\n=== {name} ===\n{answer}\n")
```

[OUTPUT, expected]:

```
=== A. Narrative prose ===
The patient's PHQ-9 improved from 18 to 12, consistent with partial response,
following sertraline titration from 50 mg to 100 mg daily.

=== B. Structured med list ===
PHQ-9 decreased from 18 (2024-03-15) to 12 (2024-05-01), corresponding to
the sertraline dose increase from 50 mg to 100 mg on 2024-04-10.

=== C. JSON / coded ===
PHQ-9 fell from 18 to 12 over the period when sertraline was titrated
from 50 mg to 100 mg.

=== D. Markdown table ===
The PHQ-9 score dropped from 18 to 12 after sertraline was increased
from 50 mg to 100 mg daily.
```

### Cell 10 — Why this matters [MD]

> Four totally different surface representations. Four different token sequences. The model answers the same question correctly from any of them.
>
> This is why you don't need a separate parser for "narrative notes vs. structured records vs. coded data vs. tables." Tokenization is the unifying move — once it's done, downstream machinery is format-agnostic.
>
> *(For the neuroscience-inclined audience: this is structurally the same move that libraries like `temporaldata` make for heterogeneous neural recordings — different sessions, rigs, sampling rates all get unified into a single time-indexed representation that downstream models operate on indifferently.)*

[REVIEWER NOTE: I parenthesized the temporaldata callout because it's a bonus for the neuroscience-trained subset of the audience and shouldn't be load-bearing for everyone. You can deliver it verbally or skip depending on the room.]

---

## §3 — Positional encoding (verbal-heavy, minimal code)

### Cell 11 — Section header [MD]

> ## §3. Positional encoding: how the model knows what came in what order
>
> Tokens alone are an unordered bag.
>
> If all the model saw were token IDs, "sertraline 100 mg" and "100 mg sertraline" would be identical input. Word order is *not* in the tokenization.
>
> Order is added separately, as **positional encoding**: each token's vector representation gets a position-specific signal added to it before the model does anything else. Today this is almost always *RoPE* (rotary positional embeddings); historically it was sinusoidal. The math doesn't matter for our purposes.

### Cell 12 — A short demonstration [CODE]

```python
# Tokens are a bag without position. Watch:

a = enc.encode("sertraline 100 mg")
b = enc.encode("100 mg sertraline")

print(f"sertraline 100 mg → tokens: {a}")
print(f"100 mg sertraline → tokens: {b}")
print(f"\nSame tokens, different order? {sorted(a) == sorted(b)}")
print(f"Same sequence?                {a == b}")
```

[OUTPUT, expected]:
```
sertraline 100 mg → tokens: [82740, 1739, 1041, 5494]
100 mg sertraline → tokens: [1041, 5494, 82740, 1739]

Same tokens, different order? True
Same sequence?                False
```

[REVIEWER NOTE: token IDs are illustrative, not actual. tiktoken will give different specific values but the equality results above will hold.]

### Cell 13 — The intuition that lands [MD]

> The useful intuition: **position is information that's added to each token, not something the model gets for free**.
>
> Three consequences worth knowing:
>
> 1. **Context windows have a hard ceiling.** A model trained on positions up to *N* doesn't smoothly extrapolate beyond *N*. It degrades, sometimes catastrophically.
>
> 2. **Order in the prompt matters.** Instructions placed at the *end* of a long prompt often outweigh instructions placed at the *start*. So do format-defining examples.
>
> 3. **The model doesn't attend uniformly across position.** It tends to over-weight the start and end of the input and under-weight the middle. The literature calls this "lost in the middle" ([Liu et al. 2023](https://arxiv.org/abs/2307.03172)).
>
> The next section is a clinical demonstration of #3.

[REVIEWER NOTE: I deliberately don't show RoPE math or visualize positional embeddings. Either would eat 5 minutes you don't have. The verbal intuition + the consequences is what an SOBP audience needs to evaluate vendor claims, not how the math works.]

---

## §4 — Context windows: the long-chart needle demo

### Cell 14 — Section header [MD]

> ## §4. Context: what the model actually "knows" right now
>
> Everything an LLM "knows" in a given call is one of two things:
>
> 1. **Pretraining weights.** Frozen at training time. Doesn't include this morning's news, your patient's chart, or anything that happened to the world after the model was trained.
> 2. **The context window.** The tokens currently in the prompt. This is the model's entire "working memory" for this call. Nothing else is accessible.
>
> No memory between calls. No filesystem. No database lookup unless you build one (we'll do this in Notebook 2). The window is everything.
>
> The window is finite, and — as we just said — the model doesn't attend uniformly across it. Let's see what that means for clinical work.

### Cell 15 — Loading the chart [CODE]

```python
# We'll use a synthetic 28-document psychiatric chart for a fictional patient,
# Eleanor Whitfield, ~12,500 words. The chart is realistic in structure and
# contains a clinically critical detail: a 2022 hospitalization complicated by
# sertraline-induced SIADH with a witnessed seizure, leading to documented
# class-level avoidance of SSRIs.
#
# The CURRENT admission's H&P (Document 24) ends with a Plan that proposes
# initiating sertraline 25 mg daily. Our question: will the model catch this?

import urllib.request

CHART_URL = "https://[hosted-location]/whitfield_chart_FULL.md"  # replace at build time
chart = urllib.request.urlopen(CHART_URL).read().decode("utf-8")

# Quick stats
print(f"Words:   {len(chart.split()):,}")
print(f"Tokens:  {len(enc.encode(chart)):,}")
print(f"Documents: {chart.count('## Document')}")
```

[OUTPUT, expected]:
```
Words:    12,461
Tokens:   ~17,000
Documents: 28
```

[REVIEWER NOTE: We need to host the chart somewhere fetchable from Colab. Three options: (1) a public GitHub repo, (2) a Google Drive shared file with a direct-link URL, (3) just paste the chart inline as a Python string in the cell. Option 3 is ugliest visually but bulletproof for live demo. I'd lean toward (1) for the polish, (3) as fallback. Flag preference.]

### Cell 16 — Asking the safety question [CODE]

```python
# The clinical question. This is the prompt we'd actually want a clinical-AI
# assistant to answer well.

question = """You are reviewing this patient's chart prior to attending rounds.
Are there any safety concerns with the current medication plan documented
in the most recent admission H&P? If so, describe them specifically and
explain what you would recommend."""

prompt = f"{chart}\n\n---\n\n{question}"
answer = call_llm(prompt)
print(answer)
```

[OUTPUT — three possible regimes, depending on model behavior]:

```
[Best case — model integrates correctly]
Yes — the Plan in the current admission H&P proposes initiating sertraline
25 mg daily. This is contraindicated for this patient. The chart documents
a severe sertraline-induced SIADH with hyponatremia (Na 122) complicated
by a witnessed generalized tonic-clonic seizure during the 2022 admission
(Documents 9–12). Both the 2022 Discharge Summary and the medication
reconciliation on the current admission flag this as a SEVERE adverse
drug reaction with class-level avoidance of all SSRIs. The current H&P
itself lists this in the Allergies section. I would recommend not initiating
sertraline; alternatives consistent with the documented contraindication
include continued bupropion optimization, mirtazapine titration (already
started), vortioxetine with sodium monitoring, or referral for ECT.

[Middle case — model finds the contradiction within Doc 24 but doesn't
integrate the deeper history]
The H&P lists sertraline as a severe adverse reaction in the Allergies
section but the Plan proposes initiating sertraline. This appears to
be a contradiction within the H&P that should be clarified before
proceeding.

[Failure case — model misses entirely]
The medication plan as documented appears appropriate for refractory
depression. Standard monitoring for SSRI side effects would be
warranted...
```

### Cell 17 — Discussion + the lost-in-the-middle move [MD]

> Whether the model caught it depends on (a) how the chart was packed into the context window and (b) how attention is distributed across position.
>
> Here's the more revealing experiment:

### Cell 18 — Moving the needle [CODE]

```python
# Same chart, same question, but we move the 2022 discharge summary
# (Document 12 — the most detailed account of the SSRI reaction) to three
# different positions: at the start, in the middle (where it natively lives),
# and at the end.

# We'll find the discharge summary by document header, extract it, and
# reinsert it at the chosen position.

import re

def extract_doc(chart_text: str, doc_num: int) -> tuple[str, str]:
    """Returns (the_doc, chart_with_doc_removed)."""
    pattern = rf"(## Document {doc_num} —.*?)(?=## Document {doc_num + 1} —|## Document 24 —)"
    # ^ simplified; actual notebook uses cleaner regex
    match = re.search(pattern, chart_text, re.DOTALL)
    doc = match.group(1).strip()
    chart_without = chart_text.replace(doc, "").strip()
    return doc, chart_without

dc_summary, chart_minus_dc = extract_doc(chart, doc_num=12)

prompts = {
    "needle at START": f"{dc_summary}\n\n{chart_minus_dc}",
    "needle in MIDDLE (native)": chart,  # already in middle
    "needle at END": f"{chart_minus_dc}\n\n{dc_summary}",
}

for label, prompt_text in prompts.items():
    full_prompt = f"{prompt_text}\n\n---\n\n{question}"
    answer = call_llm(full_prompt)
    print(f"\n{'='*60}\n{label}\n{'='*60}\n{answer[:600]}...\n")
```

[REVIEWER NOTE: This is the key demo. Talk through the three outputs as they print. Best case: needle-at-start and needle-at-end both succeed, needle-in-middle fails or partially fails — that's the clean lost-in-the-middle pattern. If needle-in-middle still succeeds (frontier model effect), you can speak to that honestly: "this newer model handles this better, but the principle still holds at longer contexts — and we've all seen this break in the wild." The phenomenon is robust; the threshold at which it shows up moves with model strength.

CRITICAL: this is the cell where you'll want to do the most rehearsal testing. If Gemini 2.5 Flash blows through the needle in all three positions, we have options: (a) pad the chart to ~25k words, (b) make the needle subtler (e.g., remove the explicit ADR table from Doc 23), (c) use a weaker/older model. I'd try in that order.]

### Cell 19 — Discussion [MD]

> A few takeaways from what just happened:
>
> 1. **The chart contained the answer.** In every position, the same words were in the same context window. The model "had access to" the SSRI contraindication in all three runs.
>
> 2. **Access is not retrieval.** What the model surfaces depends on where in the window the relevant tokens sit, what's near them, and how attention happens to be distributed for that prompt and that model.
>
> 3. **For clinical use this is the central limitation.** A long chart is the *normal* case, not the edge case. A real EHR pull for a complex patient runs 50,000–100,000+ words. Asking an LLM to "review the chart" without engineering around the context window is fragile.
>
> 4. **This is what motivates RAG** — retrieval-augmented generation, in Notebook 2. Instead of cramming the whole chart in and hoping, RAG retrieves the *relevant* chunks first and only puts those in front of the model.

---

## §5 — Teaser for the next talk

### Cell 20 — Closing [MD]

> ## What's next
>
> We've covered three things:
>
> - **Tokenization** is how text becomes the model's input — fragments of words as integers, with all the messiness that creates.
> - **Positional encoding** is how the model knows what came in what order, and why "lost in the middle" happens.
> - **Context** is the model's entire working memory for a given call — finite, non-uniform, and the source of most clinical failures.
>
> What we *haven't* covered is what the model actually does with the tokens once it has them. That's **attention** — the operation that lets each token in the window influence each other token. Attention is the heart of the transformer, and it's the subject of the next talk in this session.
>
> ---
>
> *Notebook 2 picks up the practical thread: how to use these systems for psychiatric research, given what we now know about how they work — and where they break.*

---

# End of NB1 spec

## Open questions for Logan

1. **Cell 2 (setup):** the `google-genai` SDK API I sketched is my best guess at the current surface. I'll re-verify against docs at .ipynb-build time. If you've already used the SDK and have a snippet you trust, paste it and I'll match it.

2. **Cell 6 (tokenization-future):** The arxiv links to BLT and H-Net are correct. ~20 sec of talk time is what I budgeted; can be shorter if you're tight.

3. **Cell 15 (chart hosting):** Do you have a preferred way to make the chart fetchable from Colab? GitHub raw, Google Drive, or paste-inline-as-Python-string? I'd go GitHub raw if you have a personal/lab repo, paste-inline if not.

4. **Cell 18 (needle-moving demo):** This is the cell most likely to need adjustment after rehearsal. If Gemini 2.5 Flash finds the needle in all three positions, our options in order of preference are: (a) pad chart to ~25k words, (b) make the needle subtler, (c) drop to a smaller/older model. Which would you want first?

5. **Cell 20 (closing):** I kept the closing very brief and explicitly handed off to the attention talk. If your co-presenter for talk 2 has specific framing they want, we can mirror it here.

6. **Tone calibration:** the markdown cells are written somewhat formally (matching SOBP audience expectations). If you want a looser, more conversational tone, easy to swap. The Cosyne tutorial reference suggests slightly drier-formal is fine.

7. **What's missing:** I didn't include any explicit visualization (e.g., a heatmap of attention patterns, a diagram of positional encoding). Cosyne-style would have these. I cut them deliberately for time and because the verbal-plus-code beats are already substantive. If you want one diagram, the highest-value one would be a "lost in the middle" figure showing accuracy as a function of needle position — but I'd cite Liu et al. 2023's figure and not regenerate it.

Once you sign off (with edits), I:
1. Apply your edits to this spec.
2. Convert spec → `.ipynb` file with proper Colab metadata.
3. Move to Notebook 2 spec.
