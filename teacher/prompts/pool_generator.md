# Teacher prompt v1 — candidate pool generator

You are the teacher of a small (1.7B) student model. Generate training
examples for next week's candidate pool.

## Input
- WEAKNESS PROFILE: {weakness_vector}   (per-skill error rates from this week's diagnostic)
- SKILL AXES: math, reasoning, instruction_following, recall, extraction, code
- COUNT: {n} examples

## Requirements
1. Bias generation toward high-error skills, but cover every axis (the QUBO
   does the selecting — your job is a *rich pool*, not a pre-selected batch).
2. Vary difficulty: ~30% at the student's current level, ~50% slightly above,
   ~20% stretch.
3. Vary surface form aggressively — the selection objective penalizes
   near-duplicates, so redundant phrasing wastes pool slots.
4. Each example is a (prompt, ideal_response) pair suitable for supervised
   fine-tuning in ChatML format.

## Output
JSON list, one object per example:
```json
{"prompt": "...", "response": "...",
 "skills": {"math": 0.8, "reasoning": 0.3},
 "difficulty": "at|above|stretch"}
```
Skill coverage values are 0–1 and should reflect how strongly the example
exercises each skill. Token counts and embeddings are computed downstream —
do not include them.
