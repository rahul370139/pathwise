"""
onet_to_markdown.py — turn the O*NET/BLS career CSV into per-occupation
markdown "career briefs" for the knowledge base.

Why: the structured CSV (backend/data/onet_bls_trimmed.csv) stays the source of
truth for salary/growth numbers used by career_matcher + build_career_plan.
These markdown briefs make the SAME career facts *retrievable* in the Foundry
index, so when the simulator or Learn chat discusses a role, grounded career
context is available alongside the technical interview corpus.

Output: backend/knowledge_base/onet_careers/onet_<soc>_<slug>.md

Run:
    python3 scripts/onet_to_markdown.py
    python3 scripts/onet_to_markdown.py --out /custom/dir --limit 50
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = ROOT / "data" / "onet_bls_trimmed.csv"
DEFAULT_OUT = ROOT / "knowledge_base" / "onet_careers"

RIASEC = ["realistic", "investigative", "artistic", "social", "enterprising", "conventional"]
RIASEC_LABEL = {
    "realistic": "Realistic (hands-on, building)",
    "investigative": "Investigative (research, analysis)",
    "artistic": "Artistic (creative, design)",
    "social": "Social (helping, teaching)",
    "enterprising": "Enterprising (leading, persuading)",
    "conventional": "Conventional (organizing, detail)",
}


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", (text or "").lower()).strip("_")[:60] or "role"


def _money(v: str) -> str:
    try:
        return f"${int(float(v)):,}"
    except Exception:
        return "n/a"


def _top_interests(row: dict) -> list[str]:
    scored = []
    for k in RIASEC:
        try:
            scored.append((float(row.get(k, 0) or 0), k))
        except Exception:
            pass
    scored.sort(reverse=True)
    return [RIASEC_LABEL[k] for _, k in scored[:3]]


def render(row: dict) -> str:
    title = (row.get("title") or "Unknown role").strip()
    skills = [s.strip() for s in (row.get("top_skills") or "").split(",") if s.strip()]
    low, high = _money(row.get("salary_low", "")), _money(row.get("salary_high", ""))
    growth = (row.get("growth_pct") or "").strip()
    growth_txt = f"{growth}% projected growth" if growth else "growth data unavailable"

    lines = [
        f"# {title} — Career Brief",
        "",
        "**Type:** O*NET / BLS career reference",
        f"**SOC code:** {row.get('soc_code', 'n/a')}",
        f"**Typical salary band:** {low} – {high}",
        f"**Outlook:** {growth_txt}",
        "",
        "## Top skills",
    ]
    lines += [f"- {s}" for s in skills] or ["- (not specified)"]
    lines += [
        "",
        "## A day in the life",
        (row.get("day_in_life") or "Not specified.").strip(),
        "",
        "## Best-fit interests (RIASEC)",
    ]
    lines += [f"- {label}" for label in _top_interests(row)]
    lines += [
        "",
        "## How to use this brief",
        f"Use this as grounded context when assessing readiness for **{title}**: "
        f"compare a candidate's skills against the top skills above, anchor salary "
        f"and growth expectations to the band shown, and tailor interview focus to "
        f"the day-in-the-life responsibilities.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(DEFAULT_CSV))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--limit", type=int, default=0, help="0 = all rows")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    with open(args.csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if args.limit and i >= args.limit:
                break
            soc = _slug(row.get("soc_code", str(i)))
            name = _slug(row.get("title", ""))
            (out_dir / f"onet_{soc}_{name}.md").write_text(render(row), encoding="utf-8")
            written += 1

    print(f"Wrote {written} career briefs to {out_dir}")


if __name__ == "__main__":
    main()
