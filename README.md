# mammo-multiimage

Study-level (2-view) extension of the single-image Model B mammography VLM
pipeline (`breast_cancer_qwen-B_clean.ipynb`). Groups VinDr-Mammo by
**`(study_id, laterality)`** so the CC and MLO views of the same breast are
fed to Qwen2.5-VL **together**, instead of independently.

## Why this grouping, and not something else

This directly answers the open questions raised in the meeting with Sonya:

| Question | Decision | Where in code |
|---|---|---|
| Same-breast pair vs. any-2-images? | `(study_id, laterality)` — same side's CC+MLO. "R vs L" comparison is a separate, later problem (asymmetry). | `grouping.py::build_laterality_groups` |
| Is BI-RADS/density duplicated across views? | Assumed identical for the same side; **asserted**, not trusted — logs a warning if a study ever disagrees. | `grouping.py::_check_birads_density_consistency` |
| Same finding on both views — one label or reasoned per-view? | **Option A**: independent per-view annotation (no cross-view lesion-identity matching). VinDr has no correspondence ID between findings across views, so true matching (Option B) would need a new heuristic. | `messages.py`, `schemas.py::FindingBox.view_position` |
| Should boxes carry an `[Image N]` prefix? | Yes — every box in the target text names its source image (`"...in Image 1 (CC) <box>...`). Kept explicit in the *text*, not just positional, so it survives group sizes other than 2. | `messages.py::_image_index_label` |
| "Tell the AI it's the same patient" | Explicit sentence in every user turn: *"Image 1 and Image 2 are the CC and MLO views... of the same patient."* | `messages.py::patient_context_sentence` |

## What's deferred (explicitly, not by omission)

- **Cross-laterality (R vs L) asymmetry comparison** — Sonya's "R has calc,
  L is normal" example. Needs a 4-view grouping and a way to phrase
  asymmetry as a supervised target. Not started here.
- **True cross-view lesion correspondence** (same physical mass linked
  across CC/MLO) — would need either new radiologist annotation or an
  IoU/geometry-based matching heuristic against VinDr's raw boxes.
- **Chain-of-thought / rationale generation** ("what process led to
  architectural distortion") — separate research direction, flagged in
  notes as "check it later."

## Open question for Sonya (not yet resolved)

Missing-view handling: if a laterality only has one of {CC, MLO}, this
code does **not** drop it — it builds a single-image example instead and
tags it `is_complete_pair: False` in the JSONL metadata, so it's easy to
filter or audit later. Confirm this is the right default vs. dropping
incomplete pairs entirely.

## Usage

```bash
pip install -e ".[dev]"
pytest                      # 12 unit tests, no GPU/data required

# Dry run (default) -- builds and logs counts, writes nothing:
python -m mammo_multiimage.cli \
    --data-root /gs/gsfs0/users/sohenry/vindr_mammo \
    --out-dir  /gs/gsfs0/users/kduong/lora_data/multiimage

# Actually write the JSONL files once reviewed:
python -m mammo_multiimage.cli \
    --data-root /gs/gsfs0/users/sohenry/vindr_mammo \
    --out-dir  /gs/gsfs0/users/kduong/lora_data/multiimage \
    --commit
```

## Package layout

```
mammo_multiimage/
├── schemas.py          # ViewRecord, FindingBox, LateralityGroup (typed, no raw dicts)
├── boxes.py            # coordinate scaling (unchanged math, isolated + tested)
├── grouping.py          # CSV rows -> LateralityGroup objects
├── messages.py         # Qwen chat-format message + target-text builders
├── dataset_builder.py  # LateralityGroup -> JSONL examples, write_split
└── cli.py               # end-to-end script, dry-run by default
tests/                   # pytest, synthetic fixtures, no data dependency
```

## Why this structure (for learning purposes)

This mirrors common "big tech" conventions for a small data-pipeline
package:
- **Typed dataclasses over dicts** — mistakes (wrong key name, wrong type)
  fail loudly during development instead of silently producing bad
  training data.
- **Pure functions, testable without GPUs or real data** — `boxes.py` and
  `messages.py` have zero I/O; every behavior decision (ordering, tagging,
  consistency checks) has a unit test using small synthetic fixtures.
- **Dry-run by default** — mirrors your own instinct in the draft notebook
  (data-writing calls commented out pending review); here it's a real flag
  instead of commented-out code, so it can't be forgotten or accidentally
  re-enabled.
- **Logging instead of print** — `logger.warning(...)` on the BI-RADS
  consistency check means a real disagreement surfaces in logs without
  crashing a long-running job.
