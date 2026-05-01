# Transformer & LLM Basics with Relevance to Psychiatry

Resources from the **AI in Biomedical Sciences Workshop** at SOBP 2026.

**Landing page:** [grosenick-lab-cornell.github.io/sobp-2026-transformers](https://grosenick-lab-cornell.github.io/sobp-2026-transformers/)

---

## Resources

| # | Resource | Author | Open |
|---|---|---|---|
| 1 | **How LLMs see clinical text**: tokenization, positional encoding, context window | Logan Grosenick, PhD (Cornell) | [Open in Colab](https://colab.research.google.com/github/Grosenick-Lab-Cornell/sobp-2026-transformers/blob/main/notebooks/01_how_llms_see_clinical_text.ipynb) |
| 2 | **Using LLMs for psychiatric research**: RAG, schema-constrained extraction, four failure modes | Logan Grosenick, PhD (Cornell) | [Open in Colab](https://colab.research.google.com/github/Grosenick-Lab-Cornell/sobp-2026-transformers/blob/main/notebooks/02_using_llms_for_psychiatric_research.ipynb) |
| 3 | **Computable phenotyping with LLMs**: RDoC scoring, PHQ-9 estimation, three approaches in R | Thomas H. McCoy, MD (Harvard / MGH) | [Open in Colab](https://colab.research.google.com/github/Grosenick-Lab-Cornell/sobp-2026-transformers/blob/main/notebooks/McCoy_SOBP_2026.ipynb) |
| 4 | **Slurm Whisperer**: slash-command toolkit for driving SLURM clusters from Claude Code | Teddy Akiki, MD (Stanford) | [View on GitHub](https://github.com/Grosenick-Lab-Cornell/sobp-2026-transformers/tree/main/slurm-whisperer) |

---

## Run the notebooks

Notebooks 1 and 2 run on Colab's free T4 GPU with a small open-source model (Phi-3.5 mini Instruct) loaded locally. No API keys, no vendor accounts. Notebook 3 (McCoy) is in R; see notebook header for runtime notes.

Once a Colab notebook is open:

1. **Runtime → Change runtime type → T4 GPU** (for notebooks 1 and 2).
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

For Slurm Whisperer setup and use, see [`slurm-whisperer/README.md`](slurm-whisperer/README.md).

---

## What is in here

```
notebooks/         runnable Colab notebooks + utils.py
content/           synthetic Whitfield chart, cached model outputs
visuals/           header, dividers, conceptual diagrams (PNG + SVG)
slurm-whisperer/   Teddy Akiki's slash-command toolkit for SLURM clusters
docs/              GitHub Pages landing page
```

All clinical content is fully synthetic. No real patient data, even de-identified.

---

## Workshop organizers

Sophia Frangou, Martin Paulus, Logan Grosenick, Teddy Akiki, Thomas H. McCoy.
