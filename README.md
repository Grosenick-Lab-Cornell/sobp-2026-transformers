# Transformer & LLM Basics with Relevance to Psychiatry

A 45-minute introduction to LLM basics for the **AI in Biomedical Sciences Workshop** at SOBP 2026.

**Speaker:** Logan Grosenick · Weill Cornell Psychiatry · [grosenicklab.org](https://grosenicklab.org)
**Landing page:** [grosenick-lab-cornell.github.io/sobp-2026-transformers](https://grosenick-lab-cornell.github.io/sobp-2026-transformers/)

---

## Run the notebooks

Both notebooks run on Colab's free T4 GPU with a small open-source model (Phi-3.5 mini Instruct) loaded locally. No API keys, no vendor accounts.

| Notebook | What it covers | Open |
|---|---|---|
| **01: How LLMs see clinical text** | Tokenization, positional encoding, context window | [Open in Colab](https://colab.research.google.com/github/Grosenick-Lab-Cornell/sobp-2026-transformers/blob/main/notebooks/01_how_llms_see_clinical_text.ipynb) |
| **02: Using LLMs for psychiatric research** | RAG, schema-constrained extraction, four failure modes | [Open in Colab](https://colab.research.google.com/github/Grosenick-Lab-Cornell/sobp-2026-transformers/blob/main/notebooks/02_using_llms_for_psychiatric_research.ipynb) |

Once a notebook is open in Colab:

1. **Runtime → Change runtime type → T4 GPU**
2. **Run all cells.** The first setup cell installs dependencies and downloads the model (about 1 to 2 minutes); subsequent cells are fast.

The §4 long-chart cells in Notebook 1 may replay cached outputs (see `content/cached_outputs.json`) so the demo stays on time during the talk. Delete the relevant cache labels to force live regeneration.

---

## Run locally

If you have an NVIDIA GPU and prefer a local Jupyter:

```bash
git clone https://github.com/Grosenick-Lab-Cornell/sobp-2026-transformers
cd sobp-2026-transformers
pip install -r requirements.txt
jupyter lab notebooks/
```

---

## What is in here

```
notebooks/    runnable Colab notebooks + utils.py
content/      synthetic Whitfield chart, cached model outputs
visuals/      header, dividers, conceptual diagrams (PNG + SVG)
docs/         GitHub Pages landing page
specs/        cell-by-cell design notes for each notebook
```

All clinical content is fully synthetic. No real patient data, even de-identified.
