"""Week-0 smoke test: synthetic 50-candidate pool through the QUBO builder,
greedy, and classical solvers. No API keys, no GPU, runs anywhere.

    python scripts/toy_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "qubo"))

import numpy as np

from build_qubo import build_qubo, solution_stats
from solvers import solve_random, solve_greedy, solve_classical, selection_diff

SKILLS = ["math", "reasoning", "instruction_following", "recall", "extraction", "code"]


def make_toy_pool(n=50, seed=0):
    rng = np.random.default_rng(seed)
    pool = []
    for i in range(n):
        k = rng.choice(len(SKILLS), size=rng.integers(1, 3), replace=False)
        pool.append(
            {
                "id": f"toy-{i:03d}",
                "text": f"toy example {i}",
                "tokens": int(rng.integers(80, 600)),
                "skills": {SKILLS[j]: float(rng.uniform(0.3, 1.0)) for j in k},
                "embedding": rng.normal(size=32).tolist(),
            }
        )
    # inject near-duplicates to exercise the diversity penalty
    for i in (10, 20, 30):
        pool[i + 1]["embedding"] = (
            np.array(pool[i]["embedding"]) + rng.normal(scale=0.05, size=32)
        ).tolist()
    return pool


def main():
    pool = make_toy_pool()
    weakness = {s: w for s, w in zip(SKILLS, [0.8, 0.6, 0.3, 0.5, 0.2, 0.7])}
    budget = 4000
    lam1, lam2 = 1.0, 1e-6  # toy values — real λs come from the week-0 sweep

    Q = build_qubo(pool, weakness, budget, lam1, lam2)
    print(f"Q: {len(pool)} vars, {len(Q)} terms")

    rnd = solve_random(pool, Q, budget, seed=0)
    g = solve_greedy(pool, Q, weakness, budget)
    c = solve_classical(pool, Q, restarts=10)

    for r in (rnd, g, c):
        stats = solution_stats(pool, np.array(r.x), budget)
        print(
            f"{r.solver:9s} obj={r.objective:12.2f}  "
            f"n={stats['n_selected']:2d}  tokens={stats['tokens']:5d} "
            f"({stats['budget_utilization']:.0%} of budget)  {r.wall_time_s:.2f}s"
        )

    for a, b in ((rnd, g), (g, c)):
        d = selection_diff(a, b, pool)
        print(f"{d['pair']:24s} jaccard={d['jaccard']}, overlap={d['overlap']}")

    assert rnd.objective > c.objective, "optimized arm should beat random on objective"

    assert c.objective <= g.objective + 1e-9, "SA should never lose to greedy on objective"
    stats_c = solution_stats(pool, np.array(c.x), budget)
    assert 0.7 <= stats_c["budget_utilization"] <= 1.3, "budget term not binding — tune λ₂"
    print("OK")


if __name__ == "__main__":
    main()
