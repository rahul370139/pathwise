"""
eval_simulator.py — offline evaluation harness for the Career Simulator.

Proves the reliability story for the judges: it runs a small golden set through
the named agents and reports the metrics the rubric rewards:

    groundedness        fraction of scored answers the rubric actually supports
    refusal_correctness out-of-scope inputs are refused, not answered
    answer_relevance     planner produces gaps grounded in the JD
    latency_p95_sec      end-to-end responsiveness

Writes backend/data/eval/simulator_eval_report.json, which GET /api/simulator/eval
surfaces and the dashboard eval card renders.

Run (needs GROQ_API_KEY; uses Foundry IQ if configured, else Supabase):
    python backend/eval_simulator.py
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Dict, List

from career_simulator import (
    PlannerAgent,
    RetrievalAgent,
    InterviewerAgent,
    ScorerAgent,
    SafetyGuard,
)

OUT = Path(__file__).parent / "data" / "eval" / "simulator_eval_report.json"

# Inline fixtures avoid shipping PDF resumes. parsed_resume mirrors parse_resume().
GOLDEN: List[Dict] = [
    {
        "id": "ds-strong",
        "expect": "answer",
        "resume": {"headline": "ML Engineer", "skills": ["python", "pytorch", "sql", "mlops", "airflow"]},
        "jd": "Senior Data Scientist. Requirements: Python, statistics, A/B testing, "
              "machine learning, experimentation, SQL, stakeholder communication.",
        "strong_answer": "I design A/B tests by defining the metric, computing power and "
                         "sample size, randomizing at the user level, and guarding against "
                         "peeking with sequential testing; I validate SRM before reading results.",
    },
    {
        "id": "fe-partial",
        "expect": "answer",
        "resume": {"headline": "Backend Engineer", "skills": ["java", "spring", "postgres"]},
        "jd": "Frontend Engineer. Requirements: React, TypeScript, accessibility, "
              "state management, testing, performance optimization, CSS.",
        "strong_answer": "I manage state with hooks and context, memoize expensive renders, "
                         "and code-split routes; for accessibility I use semantic HTML and ARIA.",
    },
    {
        "id": "oos-short",
        "expect": "refuse",
        "resume": {"headline": "n/a", "skills": ["python"]},
        "jd": "hi",  # too short → SafetyGuard should refuse
        "strong_answer": "",
    },
]


async def _run_case(case: Dict) -> Dict:
    t0 = time.perf_counter()
    refusal = SafetyGuard.check_scope(case["jd"], case["resume"])
    refused = refusal is not None

    grounded_supported = None
    gaps_found = 0
    if not refused:
        plan = await PlannerAgent().run(case["resume"], case["jd"])
        gaps = plan.get("gaps", [])
        gaps_found = len(gaps)
        if gaps:
            name = gaps[0].get("name") if isinstance(gaps[0], dict) else str(gaps[0])
            retrieved = await RetrievalAgent().run(name, plan.get("target_role", ""))
            question = await InterviewerAgent().run(
                plan.get("target_role", ""), name, None, retrieved["context"]
            )
            scored = await ScorerAgent().run(name, question, case["strong_answer"], retrieved["context"])
            grounded_supported = bool(scored.get("supported"))

    return {
        "id": case["id"],
        "expect": case["expect"],
        "refused": refused,
        "gaps_found": gaps_found,
        "grounded_supported": grounded_supported,
        "latency_sec": round(time.perf_counter() - t0, 3),
    }


def _p95(values: List[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = min(len(s) - 1, int(round(0.95 * (len(s) - 1))))
    return round(s[k], 3)


async def main() -> None:
    results = [await _run_case(c) for c in GOLDEN]

    refusal_cases = [r for r in results if r["expect"] == "refuse"]
    refusal_correct = sum(1 for r in refusal_cases if r["refused"]) / len(refusal_cases) if refusal_cases else None

    answer_cases = [r for r in results if r["expect"] == "answer"]
    grounded = [r["grounded_supported"] for r in answer_cases if r["grounded_supported"] is not None]
    groundedness = (sum(1 for g in grounded if g) / len(grounded)) if grounded else None
    relevance = (sum(1 for r in answer_cases if r["gaps_found"] > 0) / len(answer_cases)) if answer_cases else None

    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scenario_count": len(GOLDEN),
        "metrics": {
            "groundedness": round(groundedness, 3) if groundedness is not None else None,
            "answer_relevance": round(relevance, 3) if relevance is not None else None,
            "refusal_correctness": round(refusal_correct, 3) if refusal_correct is not None else None,
            "latency_p95_sec": _p95([r["latency_sec"] for r in results]),
        },
        "results": results,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2))
    print(f"Wrote {OUT}")
    print(json.dumps(report["metrics"], indent=2))


if __name__ == "__main__":
    asyncio.run(main())
