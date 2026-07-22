# qurriculum

**Does quantum-curated curriculum teach a model faster?**

A teacher agent evals a small student model weekly, curates a pool of candidate
training examples, and selects each week's LoRA batch by solving a QUBO three
ways in parallel:

1. **greedy** — value/token ratio baseline
2. **classical** — simulated annealing
3. **quantum** — D-Wave Advantage annealer (Leap)

Three students train, one per solver — same teacher, same pool, same token
budget, same hyperparameters. The only experimental variable is the solver.
The deliverable is the longitudinal learning-curve comparison: every existing
quantum-selection paper scores the solver objective; this scores **downstream
student learning, closed-loop**, where this week's eval reweights next week's
QUBO.

## Frozen protocol (fixed at week 0, never changed)

| Component | Value |
|---|---|
| Student | Qwen3-1.7B base, QLoRA r=16, 4-bit |
| Token budget / batch | 100k |
| Static eval | 300 items, 6 skill axes, never trained on |
| Training | Unsloth on Colab T4, fixed hyperparams (`config/protocol.yaml`) |
| Solvers | greedy / SA / D-Wave, same Q matrix weekly |

## Repo layout

```
config/     frozen protocol + λ values
teacher/    teacher prompts (grader, pool generator, report-card writer)
qubo/       Q-matrix builder + the three solvers
train/      Colab QLoRA training script
eval/       static + diagnostic eval runners, weakness vectors
logs/       longitudinal data: per-week solutions, diffs, eval scores
posts/      weekly report cards
scripts/    toy tests, utilities
```

## Weekly loop

1. Diagnostic eval → per-student weakness vector **w**
2. Teacher generates shared candidate pool (~400 examples, tagged with skills, tokens, embeddings)
3. Build Q matrix per student, solve 3×, log solutions + selection diffs
4. Train 3 students (Colab, ~1 hr)
5. Static eval → append to `logs/curves.jsonl`
6. Report card → `posts/`

## QUBO

Binary x_i = include example i.

```
min  −Σ v_i x_i  +  λ₁ Σ_{i<j} sim(e_i,e_j) x_i x_j  +  λ₂ (Σ t_i x_i − B)²
```

v_i = Σ_k w_k·c_ik (weakness-weighted skill coverage), soft token budget B,
similarity pairs sparsified at sim > 0.7. See `qubo/build_qubo.py`.
