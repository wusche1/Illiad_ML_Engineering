# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

2-day intensive AI safety course. Day 1 = ML engineering foundations. Day 2 = TBD.
Each lecture is a standalone Beamer presentation with an accompanying Jupyter notebook for exercises.

## Participants

~18 people. Mostly math/physics PhDs and MScs. High-caliber, theory-heavy.
Some have PyTorch experience, some are pure theory with unclear coding backgrounds.
All have linear algebra + gradient descent knowledge. Prepared to lean on AI coding agents.
Key takeaway: go deep on fewer things, not shallow on many. The "symmetry" and "first principles" framings resonate with this audience.

## Day 1 Schedule

### Morning Track A: `01_a_ml_foundations` (~4h)
For participants who need PyTorch basics. Notes written, exercises marked with `x`.

### Morning Track B: `01_b_practical_ml` (~4h)
For participants who already know PyTorch. Notes NOT yet written.

### Afternoon: `02_llm_architecture` (combined, ~4h)
Both tracks merge. Notes NOT yet written.

## Track Selection Quiz

`forms/track_selection.html` - self-contained HTML quiz for sorting into Track A vs B.
- 8 questions: 3 MC, 1 drag-and-drop (training loop ordering), 1 matching (architecture->code), 3 experience self-assessment
- Scoring: knowledge (5pt) normalized + experience (6pt), threshold at 7 -> Track B
- Works on mobile (touch drag support)

## Exercise System

### File structure

Each exercise lives in `lectures/XX_name/exercises/YY_name/` with three files:
- `notebook.ipynb` — the notebook students open (source of truth, edit directly)
- `utils.py` — test/check functions (print PASS/FAIL)
- `solutions.py` — `SOLUTIONS` dict mapping exercise IDs to full solution code strings

### Notebook cell pattern

Per exercise within a notebook, cells appear in this order:
1. **Markdown** — explanation + instructions
2. **Code cell** — skeleton with `# TODO`. Cell metadata must include `"exercise_id": "some_id"`
3. **Test cell** — calls check function from `utils.py` (e.g. `test_sgd(SGD)`)
4. **Hint/Solution markdown** — collapsible `<details>` blocks showing full solution class
5. **Visualization** — optional plot/output cell

The notebook also needs at the top:
1. **Colab badge** as first markdown cell (REQUIRED)
2. **Setup cell** — fetches `utils.py` from GitHub raw URL on Colab, uses importlib reload

### solutions.py format

```python
SOLUTIONS = {
    "exercise_id": """\
class Foo:
    def __init__(self, ...):
        ...
    def step(self):
        # the actual solution
""",
}
```

Keys must match the `exercise_id` metadata on the corresponding notebook code cells.

### Testing

`tests/test_notebooks.py` auto-discovers every exercise dir that has both `notebook.ipynb` and `solutions.py`. For each one it:
1. Reads the notebook
2. Swaps exercise code cells with the solution from `solutions.py` (matched by `exercise_id` metadata)
3. Executes the entire notebook
4. Fails if any cell errors or any output contains "FAIL"

Run with: `uv run pytest tests/ -v`

### Adding a new exercise

1. Create `lectures/XX_name/exercises/YY_name/`
2. Author `notebook.ipynb` directly in Jupyter/Colab following the cell pattern above
3. Write `utils.py` with test functions that print PASS/FAIL
4. Write `solutions.py` with a `SOLUTIONS` dict
5. Verify: `uv run pytest tests/ -v` should pick it up automatically

GitHub repo URL pattern for Colab setup cells: `https://raw.githubusercontent.com/wusche1/Illiad_ML_Engineering/main/lectures/XX_name/exercises/YY_name/utils.py`

## Build Commands

```bash
make lecture-01_a_ml_foundations   # Compile a single lecture
make all                           # Compile all lectures
make clean                         # Clean build artifacts
uv run pytest tests/ -v            # Test all exercise notebooks with solutions
uv run python scripts/syncing/main.py   # Run Zotero sync daemon
uv run python scripts/tools/rag.py "query"  # Search literature with RAG
```

## Repo Structure

- `lectures/XX_name/{notes.md, slides.tex}` - Lecture content
- `lectures/XX_name/exercises/YY_name/{notebook.ipynb, utils.py, solutions.py}` - Exercises
- `lectures/XX_name/claude_notes.md` - AI's research notes (only when explicitly instructed)
- `tests/test_notebooks.py` - Auto-discovers and tests all exercise notebooks
- `lib/{preamble.tex, packages.tex, settings.tex, metadata.tex}` - Shared LaTeX config
- `bib/` - Zotero-synced bibliography (`refs.bib`, per-paper folders with fulltext)
- `forms/` - HTML forms (track selection quiz)
- `scripts/` - sync pipeline, RAG tools
- `experiments/` - Cloned experiment repos (git-ignored)
- `scripts/config.yaml` - Zotero sync, extraction, embeddings config

## Key Rules

1. **Never modify `_fulltext.md` files** - Read-only source material
2. **Only write `slides.tex` when explicitly asked** - User writes notes, AI transforms to Beamer frames
3. **Put each sentence on a new line** in LaTeX files (improves diffs, invisible in PDF)
4. **Read existing slides before editing** to maintain context and consistency
5. **Verify citations** by reading the corresponding `bib/[key]/[key]_fulltext.md`
6. **Search literature** when answering questions about papers or verifying claims
7. **Notes are the source of truth** - slides should never contain information not in notes
8. **Never manually edit `bib/refs.bib`** - Auto-synced from Zotero. Use the Zotero skill to add papers.
9. **Avoid em-dashes (---)** - Use commas, parentheses, or separate sentences instead
10. **Use `[fragile]`** on frames with verbatim or code content
11. **Use `[allowframebreaks]`** on bibliography frames
12. **Keep frames concise** - Bullet points, not paragraphs
13. **New lectures**: Copy an existing lecture folder, update the title and Colab badge URL
14. **Notebooks must have the Colab badge** as the first markdown cell
15. **Shared metadata** lives in `lib/metadata.tex` - update there, not per-lecture
16. **Exercises are the main value prop** - Participants can read slides on their own

## Design Decisions

- Exercises are the main value prop; make them meaty
- Go deep on fewer things rather than shallow on many
- Frame content for a math/physics audience (first principles derivations, symmetry framings)
