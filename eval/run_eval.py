"""Eval harness: static eval (public curves) + diagnostic eval (QUBO feedback).

Static eval items (eval/static/static_eval.jsonl), one per line:
    {"id": str, "skill": str, "prompt": str,
     "check": "exact" | "numeric" | "judge",
     "answer": str,            # for exact/numeric
     "rubric": str}            # for judge

Scoring is deterministic wherever possible. Judge calls go through
teacher/judge.py with the student identity stripped (blind grading).

Outputs appended to logs/curves.jsonl:
    {"week": int, "student": "greedy"|"classical"|"quantum",
     "overall": float, "per_skill": {skill: acc}, "n_items": int}
"""

from __future__ import annotations

import json
import re
from pathlib import Path


def load_eval_set(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def check_exact(pred: str, answer: str) -> bool:
    return pred.strip().lower() == answer.strip().lower()


def check_numeric(pred: str, answer: str, tol: float = 1e-6) -> bool:
    nums = re.findall(r"-?\d+\.?\d*", pred.replace(",", ""))
    if not nums:
        return False
    try:
        return abs(float(nums[-1]) - float(answer)) <= tol
    except ValueError:
        return False


def score_item(item: dict, pred: str, judge_fn=None) -> bool:
    if item["check"] == "exact":
        return check_exact(pred, item["answer"])
    if item["check"] == "numeric":
        return check_numeric(pred, item["answer"])
    if item["check"] == "judge":
        if judge_fn is None:
            raise ValueError(f"item {item['id']} needs a judge but none provided")
        return judge_fn(item["prompt"], item["rubric"], pred)
    raise ValueError(f"unknown check type {item['check']}")


def run_eval(
    eval_set: list[dict],
    generate_fn,          # (prompt) -> str : the student model
    judge_fn=None,        # (prompt, rubric, pred) -> bool : blind teacher judge
) -> dict:
    per_skill: dict[str, list[bool]] = {}
    for item in eval_set:
        pred = generate_fn(item["prompt"])
        ok = score_item(item, pred, judge_fn)
        per_skill.setdefault(item["skill"], []).append(ok)

    skill_acc = {k: sum(v) / len(v) for k, v in per_skill.items()}
    n = sum(len(v) for v in per_skill.values())
    overall = sum(sum(v) for v in per_skill.values()) / n
    return {"overall": overall, "per_skill": skill_acc, "n_items": n}


def weakness_vector(diagnostic_scores: dict) -> dict[str, float]:
    """Error rates from a diagnostic run — this is w in the QUBO."""
    return {k: 1.0 - acc for k, acc in diagnostic_scores["per_skill"].items()}


def append_curve_point(week: int, student: str, scores: dict, path: str = "logs/curves.jsonl"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps({"week": week, "student": student, **scores}) + "\n")
