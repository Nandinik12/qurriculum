# qurriculum

**Does an agent-curated curriculum actually teach a model faster?**

A teacher agent evals a small student model weekly, generates a pool of
candidate training examples, and three students fine-tune in parallel on
batches selected three different ways from the *same* pool, under the *same*
token budget:

| Arm | Teacher signal | Selection |
|---|---|---|
| **random** | none | uniform random until budget filled (control) |
| **greedy** | weakness vector | value/token heuristic |
| **optimized** | weakness vector | global combinatorial optimization (QUBO via simulated annealing) |

The design isolates two questions nobody has answered longitudinally:

1. **Does closed-loop curation matter at all?** (random vs greedy)
2. **Does optimization add anything beyond a heuristic?** (greedy vs optimized)

"Closed-loop" means this week's eval reweights next week's selection: each
student's own weakness profile drives its own batch. Existing data-selection
work scores selection objectives or one-shot fine-tunes; this scores
**downstream student learning over a 16-week public run**.

## Frozen protocol (fixed at week 0, never changed)

| Component | Value |
|---|---|
| Student | Qwen3-1.7B base, QLoRA r=16, 4-bit |
| Token budget / batch | 100k |
| Static eval | 300 items, 6 skill axes, never trained on |
| Training | Unsloth on Colab T4, fixed hyperparams (`config/protocol.yaml`) |
| Arms | random / greedy / optimized, same pool weekly |

## Repo layout

```
config/     frozen protocol + λ values
teacher/    teacher prompts (grader, pool generator, report-card writer)
qubo/       Q-matrix builder + selection arms (random, greedy, SA, SQA, D-Wave)
train/      Colab QLoRA training script
eval/       static + diagnostic eval runners, weakness vectors
logs/       longitudinal data: per-week selections, diffs, eval scores
posts/      weekly report cards
scripts/    toy tests, utilities
```

## Weekly loop

1. Diagnostic eval → per-student weakness vector **w**
2. Teacher generates shared candidate pool (~400 examples, tagged with skills, tokens, embeddings)
3. Select batches (random / greedy / SA on the QUBO), log selections + diffs
4. Train 3 students (Colab, ~1 hr)
5. Static eval → append to `logs/curves.jsonl`
6. Report card → `posts/`

## The QUBO

Binary x_i = include example i.

```
min  −Σ v_i x_i  +  λ₁ Σ_{i<j} sim(e_i,e_j) x_i x_j  +  λ₂ (Σ t_i x_i − B)²
```

v_i = Σ_k w_k·c_ik (weakness-weighted skill coverage), soft token budget B,
similarity pairs sparsified at sim > 0.7. See `qubo/build_qubo.py`.

## Season 2 (planned)

Batch selection is a textbook QUBO, so the optimized arm generalizes to
quantum annealing. `qubo/solvers.py` already ships `solve_sqa` (simulated
quantum annealing, openjij) and `solve_quantum` (D-Wave Advantage) — quantum
students join the league next year.
