"""Three solvers, one Q matrix. All return SolverResult.

greedy    — value/token ratio, budget-capped, similarity-skip. The baseline.
classical — simulated annealing. Uses dwave-neal if installed, else the
            built-in numpy SA (identical interface, fine for toy runs).
quantum   — D-Wave Advantage QPU via Leap (needs dwave-ocean-sdk + DWAVE_API_TOKEN).

Every weekly run must log: solution, objective, wall time, and the selection
diff vs the other solvers (see log_run / selection_diff).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

from build_qubo import objective, similarity_pairs, solution_stats


@dataclass
class SolverResult:
    solver: str
    x: list[int]
    objective: float
    wall_time_s: float
    meta: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# 1. Greedy baseline
# --------------------------------------------------------------------------- #

def solve_greedy(
    candidates: list[dict],
    Q: dict[tuple[int, int], float],
    weakness: dict[str, float],
    budget: int,
    sim_threshold: float = 0.7,
) -> SolverResult:
    from build_qubo import value_vector

    t0 = time.perf_counter()
    v = value_vector(candidates, weakness)
    t = np.array([c["tokens"] for c in candidates], dtype=float)
    ratio = v / np.maximum(t, 1)
    order = np.argsort(-ratio)

    sim = similarity_pairs(candidates, sim_threshold)
    picked: list[int] = []
    used = 0
    for i in order:
        if used + t[i] > budget:
            continue
        if any(sim.get((min(i, j), max(i, j))) for j in picked):
            continue  # too similar to something already picked
        picked.append(int(i))
        used += t[i]

    x = np.zeros(len(candidates), dtype=int)
    x[picked] = 1
    return SolverResult(
        solver="greedy",
        x=x.tolist(),
        objective=objective(Q, x),
        wall_time_s=time.perf_counter() - t0,
    )


# --------------------------------------------------------------------------- #
# 2. Classical: simulated annealing
# --------------------------------------------------------------------------- #

def _sa_numpy(Q, n, sweeps=2000, t_hot=None, t_cold=0.1, rng=None):
    """Single SA run over a dense copy of Q. Returns (x, energy)."""
    rng = rng or np.random.default_rng()
    Qd = np.zeros((n, n))
    for (i, j), c in Q.items():
        Qd[i, j] = c
    Qd = Qd + np.triu(Qd, 1).T  # symmetrize for fast delta computation

    x = rng.integers(0, 2, n)
    # local field h_i = Qd[i,i] + sum_{j != i} Qd[i,j] x_j ; flip delta = (1-2x_i)*h_i
    if t_hot is None:
        t_hot = float(np.abs(Qd).max()) or 1.0
    temps = np.geomspace(t_hot, t_cold, sweeps)

    coupl = Qd.copy()
    np.fill_diagonal(coupl, 0.0)
    diag = np.diag(Qd).copy()
    field_ = coupl @ x + diag

    for T in temps:
        for i in rng.permutation(n):
            delta = (1 - 2 * x[i]) * field_[i]
            if delta <= 0 or rng.random() < np.exp(-delta / T):
                old = x[i]
                x[i] = 1 - x[i]
                field_ += coupl[:, i] * (x[i] - old)

    energy = float(x @ np.triu(Qd) @ x)
    return x, energy


def solve_classical(
    candidates: list[dict],
    Q: dict[tuple[int, int], float],
    restarts: int = 20,
    seed: int = 42,
) -> SolverResult:
    t0 = time.perf_counter()
    n = len(candidates)
    backend = "numpy-sa"
    best_x, best_e = None, np.inf

    try:
        import neal  # dwave-neal

        backend = "dwave-neal"
        sampler = neal.SimulatedAnnealingSampler()
        ss = sampler.sample_qubo(Q, num_reads=restarts, seed=seed)
        best = ss.first
        best_x = np.array([best.sample[i] for i in range(n)], dtype=int)
        best_e = float(best.energy)
    except ImportError:
        rng = np.random.default_rng(seed)
        for _ in range(restarts):
            x, e = _sa_numpy(Q, n, rng=rng)
            if e < best_e:
                best_x, best_e = x, e

    return SolverResult(
        solver="classical",
        x=best_x.tolist(),
        objective=best_e,
        wall_time_s=time.perf_counter() - t0,
        meta={"backend": backend, "restarts": restarts},
    )


# --------------------------------------------------------------------------- #
# 3. Simulated quantum annealing (no account needed)
# --------------------------------------------------------------------------- #

def solve_sqa(
    candidates: list[dict],
    Q: dict[tuple[int, int], float],
    num_reads: int = 20,
    seed: int = 42,
) -> SolverResult:
    """Path-integral Monte Carlo simulation of quantum annealing (openjij).

    The standard literature proxy for a quantum annealer: classically simulates
    the transverse-field tunneling dynamics that D-Wave hardware implements
    physically. Runs the 'quantum-dynamics' arm until real QPU access lands —
    then hardware joins as its own lineage and SQA-vs-QPU becomes a bonus
    comparison.
    """
    from openjij import SQASampler

    t0 = time.perf_counter()
    n = len(candidates)
    ss = SQASampler().sample_qubo(Q, num_reads=num_reads, seed=seed)
    best = ss.first
    x = np.array([int(best.sample[i]) for i in range(n)], dtype=int)
    return SolverResult(
        solver="sqa",
        x=x.tolist(),
        objective=float(best.energy),
        wall_time_s=time.perf_counter() - t0,
        meta={"backend": "openjij-SQA", "num_reads": num_reads},
    )


# --------------------------------------------------------------------------- #
# 4. Quantum: D-Wave Advantage via Leap
# --------------------------------------------------------------------------- #

def solve_quantum(
    candidates: list[dict],
    Q: dict[tuple[int, int], float],
    num_reads: int = 1000,
    use_hybrid_fallback: bool = True,
) -> SolverResult:
    """Requires dwave-ocean-sdk and DWAVE_API_TOKEN in the environment.

    Tries direct QPU embedding first; falls back to LeapHybridSampler if the
    embedding fails (dense budget cross-terms can make embedding tight).
    Log which path ran — it matters for the writeup.
    """
    t0 = time.perf_counter()
    n = len(candidates)

    try:
        from dwave.system import DWaveSampler, EmbeddingComposite

        sampler = EmbeddingComposite(DWaveSampler())
        ss = sampler.sample_qubo(Q, num_reads=num_reads, label="qurriculum-weekly")
        backend = ss.info.get("problem_id", "qpu")
        path = "qpu"
    except Exception as qpu_err:  # embedding failure, size, etc.
        if not use_hybrid_fallback:
            raise
        from dwave.system import LeapHybridSampler
        import dimod

        bqm = dimod.BinaryQuadraticModel.from_qubo(Q)
        ss = LeapHybridSampler().sample(bqm, label="qurriculum-weekly-hybrid")
        backend = f"hybrid (qpu failed: {type(qpu_err).__name__})"
        path = "hybrid"

    best = ss.first
    x = np.array([best.sample[i] for i in range(n)], dtype=int)
    return SolverResult(
        solver="quantum",
        x=x.tolist(),
        objective=float(best.energy),
        wall_time_s=time.perf_counter() - t0,
        meta={
            "path": path,
            "backend": str(backend),
            "num_reads": num_reads,
            "qpu_access_time_us": ss.info.get("timing", {}).get("qpu_access_time"),
        },
    )


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #

def selection_diff(a: SolverResult, b: SolverResult, candidates: list[dict]) -> dict:
    sa = {c["id"] for c, xi in zip(candidates, a.x) if xi}
    sb = {c["id"] for c, xi in zip(candidates, b.x) if xi}
    return {
        "pair": f"{a.solver}-vs-{b.solver}",
        "only_" + a.solver: sorted(sa - sb),
        "only_" + b.solver: sorted(sb - sa),
        "overlap": len(sa & sb),
        "jaccard": round(len(sa & sb) / max(len(sa | sb), 1), 4),
    }


def log_run(
    week: int,
    results: list[SolverResult],
    candidates: list[dict],
    budget: int,
    out_dir: str = "logs",
) -> Path:
    out = {
        "week": week,
        "budget": budget,
        "results": [
            {**asdict(r), "stats": solution_stats(candidates, np.array(r.x), budget)}
            for r in results
        ],
        "diffs": [
            selection_diff(results[i], results[j], candidates)
            for i in range(len(results))
            for j in range(i + 1, len(results))
        ],
    }
    path = Path(out_dir) / f"week{week:02d}_selection.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    return path
