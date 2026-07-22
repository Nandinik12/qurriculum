"""Build the weekly Q matrix from a candidate pool + weakness vector.

Objective (minimization):
    -sum_i v_i x_i
    + lam1 * sum_{i<j} sim(e_i, e_j) x_i x_j      (only pairs with sim > threshold)
    + lam2 * (sum_i t_i x_i - B)^2                 (soft token budget)

The budget term expands to:
    lam2 * [ sum_i t_i^2 x_i  + 2 sum_{i<j} t_i t_j x_i x_j  - 2B sum_i t_i x_i ]  + const
(using x_i^2 = x_i for binaries).

Candidate format (candidates.json, one list):
    {"id": str, "text": str, "tokens": int,
     "skills": {skill_name: coverage 0-1}, "embedding": [float, ...]}

Weakness vector: {skill_name: error_rate 0-1}
"""

from __future__ import annotations

import numpy as np


def value_vector(candidates: list[dict], weakness: dict[str, float]) -> np.ndarray:
    """v_i = sum_k w_k * c_ik — weakness-weighted skill coverage."""
    return np.array(
        [sum(weakness.get(k, 0.0) * c for k, c in cand["skills"].items()) for cand in candidates]
    )


def similarity_pairs(candidates: list[dict], threshold: float) -> dict[tuple[int, int], float]:
    """Sparsified cosine-similarity pairs: only sim > threshold survive."""
    E = np.array([c["embedding"] for c in candidates], dtype=float)
    E = E / np.linalg.norm(E, axis=1, keepdims=True)
    S = E @ E.T
    n = len(candidates)
    return {
        (i, j): float(S[i, j])
        for i in range(n)
        for j in range(i + 1, n)
        if S[i, j] > threshold
    }


def build_qubo(
    candidates: list[dict],
    weakness: dict[str, float],
    budget: int,
    lam1: float,
    lam2: float,
    sim_threshold: float = 0.7,
) -> dict[tuple[int, int], float]:
    """Return Q as {(i, j): coeff} with i <= j; (i, i) entries are linear terms."""
    v = value_vector(candidates, weakness)
    t = np.array([c["tokens"] for c in candidates], dtype=float)
    Q: dict[tuple[int, int], float] = {}

    # Linear: -v_i + lam2 * (t_i^2 - 2*B*t_i)
    for i in range(len(candidates)):
        Q[(i, i)] = -v[i] + lam2 * (t[i] ** 2 - 2 * budget * t[i])

    # Quadratic: budget cross-terms 2*lam2*t_i*t_j on all pairs is dense —
    # keep it exact (it's rank-1, embeddable), plus sparse diversity terms.
    n = len(candidates)
    for i in range(n):
        for j in range(i + 1, n):
            Q[(i, j)] = 2 * lam2 * t[i] * t[j]

    for (i, j), sim in similarity_pairs(candidates, sim_threshold).items():
        Q[(i, j)] = Q.get((i, j), 0.0) + lam1 * sim

    return Q


def objective(Q: dict[tuple[int, int], float], x: np.ndarray) -> float:
    """Evaluate x^T Q x for a binary vector x."""
    total = 0.0
    for (i, j), coeff in Q.items():
        total += coeff * x[i] * x[j]
    return total


def solution_stats(candidates: list[dict], x: np.ndarray, budget: int) -> dict:
    picked = [c for c, xi in zip(candidates, x) if xi]
    tokens = sum(c["tokens"] for c in picked)
    return {
        "n_selected": len(picked),
        "tokens": tokens,
        "budget_utilization": round(tokens / budget, 4),
        "ids": [c["id"] for c in picked],
    }
