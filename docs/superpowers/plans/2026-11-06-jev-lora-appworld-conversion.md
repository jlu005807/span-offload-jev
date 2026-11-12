# Jev LoRA AppWorld Conversion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans or implement inline task-by-task with verification checkpoints. Steps use checkbox syntax for tracking.

**Goal:** Convert the generated AppWorld L0/L1/L2 train/dev labels into OpenJev JSONL samples that reproduce the selected V1 online `choice` request.

**Architecture:** Add a standalone standard-library converter under `scripts/jev_lora/`. It reads label JSONL and original trace data, validates task/k alignment, rebuilds context-mode state from trace metadata and pre-call messages, uses pinned V1 instructions and online criteria, skips masked labels, and writes `train.jsonl`, `dev.jsonl`, and a conversion summary to a fresh output directory. Deploy the same script to the remote LoRA project and validate the generated files with the real OpenJev loader.

**Tech Stack:** Python 3 standard library, pytest for unit tests, existing OpenJev `wev.data` loader for server-side acceptance.

---

### Task 1: Converter unit tests

**Files:**
- Create: `tests/test_jev_lora_conversion.py`
- Create: `scripts/jev_lora/convert_appworld_labels.py`

- [ ] Test L0/L1/L2 mapping and exclusion of unknown/mask-zero labels.
- [ ] Test online context state reconstruction, exact V1 request/criteria, and `_meta` placement.
- [ ] Test failures for mismatched task/k, missing goal, and initial-message prefix mismatch.
- [ ] Verify the V1 prompt+criteria SHA-256 matches the experiment report hash.

### Task 2: Implement converter

**Files:**
- Create: `scripts/jev_lora/convert_appworld_labels.py`

- [ ] Implement strict input validation and online context-mode state projection.
- [ ] Write fresh-output `train.jsonl`, `dev.jsonl`, and `conversion_summary.json` without overwriting existing outputs.

### Task 3: Run conversion and acceptance

**Files:**
- Remote output: `/share/home/shenwsh3/ozy/jev-appworld-lora/data/openjev-v1-context/`

- [ ] Run unit tests and prompt-hash check locally.
- [ ] Copy the converter to the LoRA project and run it on train/dev labels.
- [ ] Load both splits with OpenJev's `wev.data.load_items`; compare loaded/dropped counts with the conversion summary.
- [ ] Verify test data and raw traces were not modified.
