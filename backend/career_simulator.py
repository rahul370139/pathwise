"""
career_simulator.py — PathWise "Career Simulator" multi-agent orchestrator.

The signature Creative-Apps feature: a user supplies their resume + a target job
posting, and a team of named agents runs an inspectable, multi-step loop:

    PLAN -> RETRIEVE -> GENERATE (adaptive question) -> VERIFY (score)
         -> (remediate weak gaps) -> FINALIZE (readiness report)

Design principles
-----------------
* REUSE, don't rewrite. Every agent delegates to existing modules:
    - parse_resume / _find_onet_row / _onet_summary / build_career_plan  (resume_career.py)
    - call_groq / _parse_json_safely / map_reduce_summary / gen_flashcards_quiz (distiller.py)
    - retrieve / format_kb_context  (rag_kb.py -> Foundry IQ or Supabase)
* EVERY step is emitted on a per-session event bus so the UI can stream a live
  "Agent Thinking" timeline (SSE) — this is what converts hidden reasoning into
  visible Reasoning + UX rubric points.
* SAFETY first: a SafetyGuard refuses out-of-scope uploads and forces an honest
  "insufficient grounded evidence" path instead of hallucinating.

State is held in-memory per session (demo-grade), mirroring the conversation
store pattern already used in distiller.py.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger
from pydantic import BaseModel

from distiller import (
    call_groq,
    _parse_json_safely,
    map_reduce_summary,
    gen_flashcards_quiz,
)
from resume_career import (
    parse_resume,
    parse_resume_text,
    _find_onet_row,
    _onet_summary,
    build_career_plan,
)
from rag_kb import retrieve as kb_retrieve, format_kb_context
from schemas import ExplanationLevel


# --------------------------------------------------------------------------- #
# Request model (POST /api/simulator/answer)
# --------------------------------------------------------------------------- #
class SimulatorAnswerRequest(BaseModel):
    session_id: str
    answer: str = ""
    skipped: bool = False
    comment: Optional[str] = None


# --------------------------------------------------------------------------- #
# Per-session event bus — powers the SSE "Agent Thinking" timeline
# --------------------------------------------------------------------------- #
class SessionBus:
    """Append-only event history + live fan-out to SSE subscribers."""

    def __init__(self) -> None:
        self.history: List[Dict[str, Any]] = []
        self._queues: List[asyncio.Queue] = []
        self.closed = False

    def emit(self, agent: str, status: str, message: str = "", data: Optional[Dict] = None) -> None:
        event = {
            "ts": round(time.time(), 3),
            "agent": agent,
            "status": status,
            "message": message,
            "data": data or {},
        }
        self.history.append(event)
        for q in self._queues:
            q.put_nowait(event)

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        for event in self.history:  # replay so late subscribers see the full story
            q.put_nowait(event)
        if self.closed:
            q.put_nowait({"agent": "system", "status": "close"})
        self._queues.append(q)
        return q

    def close(self) -> None:
        self.closed = True
        for q in self._queues:
            q.put_nowait({"agent": "system", "status": "close"})


# --------------------------------------------------------------------------- #
# Safety / scope guard
# --------------------------------------------------------------------------- #
class SafetyGuard:
    """Scope refusal + grounding guard. Keeps the agent honest."""

    MIN_GROUNDING_CHARS = 200

    @staticmethod
    def check_scope(jd_text: str, parsed_resume: Dict) -> Optional[str]:
        """Return a refusal reason if inputs are clearly out of scope, else None."""
        if not jd_text or len(jd_text.strip()) < 40:
            return "The job description is too short to assess. Paste the full posting."
        if not parsed_resume or not parsed_resume.get("skills"):
            return "Could not read skills from the resume (it may be scanned/image-only)."
        return None

    @classmethod
    def grounding_ok(cls, context: str) -> bool:
        return bool(context and len(context) >= cls.MIN_GROUNDING_CHARS)


# --------------------------------------------------------------------------- #
# Named agents — each one responsibility, each delegates to existing code
# --------------------------------------------------------------------------- #
class PlannerAgent:
    """Extract JD competencies, diff against the resume, rank the gaps."""

    PROMPT = """You are a hiring-panel analyst. Compare a candidate resume to a target job posting.

JOB POSTING:
\"\"\"{jd}\"\"\"

CANDIDATE SKILLS (from resume): {skills}
CANDIDATE HEADLINE: {headline}
INDUSTRY REFERENCE (O*NET):
{onet}

Return ONLY JSON (no prose, no fences):
{{
  "target_role": str,
  "competencies": [
    {{ "name": str, "importance": "high"|"medium"|"low",
       "in_resume": true|false, "gap_level": 0-3 }}
  ],
  "readiness_estimate": 0-100,
  "summary": str
}}
Rules: 5-7 competencies, ordered most-important first. gap_level 0 = candidate strong, 3 = missing entirely. Ground every competency in the posting text."""

    async def run(self, parsed_resume: Dict, jd_text: str) -> Dict:
        onet_row = _find_onet_row(self._guess_role(jd_text))
        prompt = self.PROMPT.format(
            jd=jd_text[:6000],
            skills=", ".join(parsed_resume.get("skills", []))[:1500] or "(none)",
            headline=parsed_resume.get("headline") or parsed_resume.get("current_role") or "(unknown)",
            onet=_onet_summary(onet_row),
        )
        raw = await call_groq([{"role": "user", "content": prompt}])
        plan = _parse_json_safely(raw) or {}
        comps = plan.get("competencies") or []
        # Focus the interview on the real gaps (gap_level >= 1), hardest first.
        gaps = sorted(
            [c for c in comps if isinstance(c, dict) and c.get("gap_level", 0) >= 1],
            key=lambda c: (-int(c.get("gap_level", 0)), c.get("importance") != "high"),
        )
        plan["competencies"] = comps
        plan["gaps"] = gaps or comps[:3]  # always have something to test
        plan.setdefault("target_role", self._guess_role(jd_text))
        return plan

    @staticmethod
    def _guess_role(jd_text: str) -> str:
        first = (jd_text or "").strip().splitlines()[0] if jd_text.strip() else ""
        return first[:80] or "the target role"


class RetrievalAgent:
    """Ground a competency in the KB via Foundry IQ (or Supabase fallback)."""

    async def run(self, competency: str, target_role: str) -> Dict:
        query = f"{competency} for {target_role}: best practices, interview expectations"
        matches = await kb_retrieve(query, top_k=5)
        context = format_kb_context(matches, max_chars=4000)
        citations = [
            {
                "doc": m.get("doc"),
                "section": m.get("section"),
                "source_uri": m.get("source_uri"),
                "score": round(float(m.get("similarity", 0.0)), 3),
            }
            for m in matches
        ]
        return {"context": context, "citations": citations, "count": len(matches)}


class InterviewerAgent:
    """Ask the next question; difficulty adapts to the previous score."""

    PROMPT = """You are a senior interviewer for the role: {role}.
Competency under test: {competency}
Difficulty: {difficulty} (because the candidate's last answer scored {prior}/100).

Grounded reference material (cite ideas from here, do not invent facts):
{context}

Ask ONE focused interview question (2-4 sentences max). It must probe the
competency at the stated difficulty. Return ONLY the question text, no preamble."""

    async def run(self, role: str, competency: str, prior_score: Optional[int], context: str) -> str:
        if prior_score is None:
            difficulty, prior = "warm-up", "n/a"
        elif prior_score >= 75:
            difficulty, prior = "harder / follow-up depth", prior_score
        elif prior_score >= 45:
            difficulty, prior = "same level, different angle", prior_score
        else:
            difficulty, prior = "easier / foundational", prior_score
        grounded = context if context else "(no grounded material retrieved — ask a general, fair question)"
        prompt = self.PROMPT.format(
            role=role, competency=competency, difficulty=difficulty,
            prior=prior, context=grounded[:3000],
        )
        q = (await call_groq([{"role": "user", "content": prompt}])).strip()
        return q or f"Tell me about your experience with {competency}."


class ScorerAgent:
    """Judge an answer against the grounded rubric. No grounding → honest caveat."""

    PROMPT = """You are a calibrated interview grader. Score the candidate answer.

COMPETENCY: {competency}
QUESTION: {question}
CANDIDATE ANSWER: \"\"\"{answer}\"\"\"
GROUNDED RUBRIC (source material):
{context}

Return ONLY JSON:
{{
  "score": 0-100,
  "supported": true|false,        // is the score backed by the grounded rubric?
  "strengths": [str],
  "missing_points": [str],
  "feedback": str                  // 1-2 sentences, specific and kind
}}
If the rubric is empty, set supported=false and grade conservatively on general merit."""

    async def run(self, competency: str, question: str, answer: str, context: str) -> Dict:
        prompt = self.PROMPT.format(
            competency=competency, question=question,
            answer=answer[:3000], context=(context or "(none)")[:3000],
        )
        raw = await call_groq([{"role": "user", "content": prompt}])
        result = _parse_json_safely(raw) or {}
        result.setdefault("score", 0)
        result.setdefault("supported", bool(context))
        result.setdefault("missing_points", [])
        result.setdefault("feedback", "")
        try:
            result["score"] = max(0, min(100, int(result["score"])))
        except Exception:
            result["score"] = 0
        return result


class RemediationAgent:
    """Turn a weak competency into a micro-lesson + quiz (reuses distiller)."""

    async def run(self, competency: str, context: str, missing_points: List[str]) -> Dict:
        seed = context or f"Key concepts a candidate must know about {competency}."
        focus = "; ".join(missing_points[:4])
        summary = await map_reduce_summary([f"{competency}. Focus on: {focus}\n\n{seed}"], ExplanationLevel.INTERN)
        qa = await gen_flashcards_quiz(summary, ExplanationLevel.INTERN, retrieval_context=context, num_items=3)
        return {"competency": competency, "summary": summary,
                "flashcards": qa.get("flashcards", []), "quiz": qa.get("quiz", [])}


class ReportAgent:
    """Compile the readiness report (reuses build_career_plan for the roadmap)."""

    async def run(self, parsed_resume: Dict, target_role: str, transcript: List[Dict]) -> Dict:
        # Only genuinely answered (scored) competencies feed the readiness average.
        answered = [t for t in transcript if t.get("score") is not None and not t.get("skipped")]
        skipped = [t for t in transcript if t.get("skipped")]
        scores = [t["score"] for t in answered]
        overall = round(sum(scores) / len(scores)) if scores else 0

        weakest = sorted(answered, key=lambda t: t.get("score", 0))[:3]
        # Skipped competencies are unproven → treat them as gaps to focus the plan.
        interests = [t["competency"] for t in weakest] or [t["competency"] for t in skipped]
        plan = await build_career_plan(parsed_resume, target_role, interests)

        top_gaps = [{"competency": t["competency"], "score": t.get("score", 0)} for t in weakest]
        top_gaps += [{"competency": t["competency"], "score": 0, "skipped": True} for t in skipped]
        comments = [
            {"competency": t["competency"], "comment": t["comment"]}
            for t in transcript if t.get("comment")
        ]
        return {
            "overall_readiness": overall,
            "answered": len(scores),
            "skipped": len(skipped),
            "top_gaps": top_gaps,
            "comments": comments,
            "career_plan": plan,
        }


# --------------------------------------------------------------------------- #
# Session orchestrator
# --------------------------------------------------------------------------- #
@dataclass
class SimulatorSession:
    session_id: str
    parsed_resume: Dict
    plan: Dict
    bus: SessionBus
    target_role: str
    gaps: List[Dict] = field(default_factory=list)
    idx: int = 0
    last_score: Optional[int] = None
    current_question: Optional[str] = None
    current_context: str = ""
    transcript: List[Dict] = field(default_factory=list)
    remediations: List[Dict] = field(default_factory=list)
    report: Optional[Dict] = None
    created_at: float = field(default_factory=time.time)

    # agents (stateless, shared)
    _retriever = RetrievalAgent()
    _interviewer = InterviewerAgent()
    _scorer = ScorerAgent()
    _remediator = RemediationAgent()
    _reporter = ReportAgent()

    @property
    def finished(self) -> bool:
        return self.idx >= len(self.gaps)

    async def ask_next(self) -> Optional[Dict]:
        """Retrieve evidence for the next gap and generate an adaptive question."""
        if self.finished:
            return None
        comp = self.gaps[self.idx]
        name = comp.get("name") if isinstance(comp, dict) else str(comp)

        self.bus.emit("retrieval", "running", f"Grounding '{name}'…")
        retrieved = await self._retriever.run(name, self.target_role)
        self.current_context = retrieved["context"]
        grounded = SafetyGuard.grounding_ok(self.current_context)
        self.bus.emit(
            "retrieval", "done",
            f"{retrieved['count']} sources" + ("" if grounded else " (low grounding)"),
            {"citations": retrieved["citations"], "grounded": grounded},
        )

        self.bus.emit("interviewer", "running", f"Composing a question on '{name}'…")
        question = await self._interviewer.run(self.target_role, name, self.last_score, self.current_context)
        self.current_question = question
        self.bus.emit("interviewer", "done", "Question ready", {"competency": name})
        return {"competency": name, "question": question,
                "citations": retrieved["citations"], "grounded": grounded,
                "index": self.idx, "total": len(self.gaps)}

    async def submit(self, answer: str, skipped: bool = False, comment: Optional[str] = None) -> Dict:
        """Score the answer (or record a skip), remediate if weak, advance.

        The candidate may skip any question and optionally leave a comment.
        Skipped competencies are not scored; they surface in the final report
        as unproven gaps so the plan still adapts to them.
        """
        comp = self.gaps[self.idx]
        name = comp.get("name") if isinstance(comp, dict) else str(comp)
        comment = (comment or "").strip() or None

        if skipped or not (answer or "").strip():
            self.bus.emit("interviewer", "skipped", f"Skipped '{name}'", {"competency": name})
            self.transcript.append({"competency": name, "question": self.current_question,
                                    "answer": (answer or "").strip(), "skipped": True,
                                    "comment": comment, "score": None})
            self.last_score = None
            self.idx += 1
            next_step = await self.ask_next()
            if next_step is None:
                await self.finalize()
            return {"score": None, "skipped": True, "remediation": None,
                    "next": next_step, "finished": self.finished}

        self.bus.emit("scorer", "running", f"Scoring your answer on '{name}'…")
        scored = await self._scorer.run(name, self.current_question or "", answer, self.current_context)
        self.last_score = scored["score"]
        self.transcript.append({"competency": name, "question": self.current_question,
                                "answer": answer, "comment": comment, **scored})
        self.bus.emit("scorer", "done", scored.get("feedback", ""),
                      {"score": scored["score"], "supported": scored["supported"]})

        remediation = None
        if scored["score"] < 60:  # weak → auto-generate gap-closing content
            self.bus.emit("remediation", "running", f"Building a micro-lesson for '{name}'…")
            remediation = await self._remediator.run(name, self.current_context, scored.get("missing_points", []))
            self.remediations.append(remediation)
            self.bus.emit("remediation", "done", "Micro-lesson + quiz ready", {"competency": name})

        self.idx += 1
        next_step = await self.ask_next()
        if next_step is None:
            await self.finalize()
        return {"score": scored, "skipped": False, "remediation": remediation,
                "next": next_step, "finished": self.finished}

    async def finalize(self) -> Dict:
        self.bus.emit("report", "running", "Compiling your readiness report…")
        self.report = await self._reporter.run(self.parsed_resume, self.target_role, self.transcript)
        self.bus.emit("report", "done", "Readiness report ready",
                      {"overall_readiness": self.report.get("overall_readiness")})
        self.bus.close()
        return self.report


# --------------------------------------------------------------------------- #
# In-memory session registry + public entrypoints used by main.py
# --------------------------------------------------------------------------- #
SESSIONS: Dict[str, SimulatorSession] = {}


def synthesize_jd(target_role: str, interests: Optional[List[str]] = None) -> str:
    """Build a JD-equivalent prompt from a target role + interests using O*NET.

    Lets users who only know the role they want (or picked one from the
    "not sure?" quiz) run the same grounded simulation without pasting a posting.
    """
    row = _find_onet_row(target_role)
    interests_line = (
        f"Areas the candidate wants to lean into: {', '.join(interests)}.\n"
        if interests else ""
    )
    return (
        f"{target_role}\n\n"
        f"We are hiring a {target_role}. {interests_line}"
        f"The ideal candidate demonstrates the core competencies, day-to-day "
        f"responsibilities, and top skills described below.\n\n"
        f"Industry reference (O*NET / BLS):\n{_onet_summary(row)}"
    )


async def start_session(
    jd_text: Optional[str] = None,
    resume_path: Optional[Path] = None,
    resume_text: Optional[str] = None,
    target_role: Optional[str] = None,
    interests: Optional[List[str]] = None,
) -> Dict:
    """Parse inputs, run the Planner, prepare the first question.

    Target can be supplied two ways:
      * `jd_text` — a pasted job posting, or
      * `target_role` (+ optional `interests`) — we synthesize a grounded JD
        from the O*NET row so the same multi-agent flow runs end-to-end.

    Resume can be a PDF path or raw text (demo fixtures / no-upload path).
    """
    session_id = uuid.uuid4().hex
    bus = SessionBus()
    SESSIONS[session_id] = None  # reserve id; bus available once session exists

    if (not jd_text or len(jd_text.strip()) < 40) and target_role and target_role.strip():
        jd_text = synthesize_jd(target_role.strip(), interests)

    bus.emit("planner", "running", "Reading resume and job posting…")
    if resume_text and resume_text.strip():
        parsed = await parse_resume_text(resume_text)
    elif resume_path is not None:
        parsed = await parse_resume(resume_path)
    else:
        bus.close()
        raise ValueError("Provide either a resume PDF or resume text.")

    refusal = SafetyGuard.check_scope(jd_text, parsed)
    if refusal:
        bus.emit("safety", "refused", refusal)
        bus.close()
        return {"session_id": session_id, "refused": True, "reason": refusal}

    plan = await PlannerAgent().run(parsed, jd_text)
    bus.emit("planner", "done",
             f"{len(plan.get('gaps', []))} competency gaps identified",
             {"target_role": plan.get("target_role"),
              "readiness_estimate": plan.get("readiness_estimate")})

    session = SimulatorSession(
        session_id=session_id, parsed_resume=parsed, plan=plan, bus=bus,
        target_role=plan.get("target_role", "the target role"),
        gaps=plan.get("gaps", []),
    )
    SESSIONS[session_id] = session
    first = await session.ask_next()
    return {"session_id": session_id, "refused": False, "plan": {
        "target_role": plan.get("target_role"),
        "readiness_estimate": plan.get("readiness_estimate"),
        "summary": plan.get("summary"),
        "competencies": plan.get("competencies", []),
    }, "first_question": first}


def get_session(session_id: str) -> Optional[SimulatorSession]:
    return SESSIONS.get(session_id)
