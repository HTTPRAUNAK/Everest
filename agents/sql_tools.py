"""
Deterministic tools (node 5). Thresholds come ONLY from rule_registry (R5);
the LLM never does arithmetic or judgement.

Teammate's SQL module is touched in exactly 5 places (all `db.*` calls below).
Assumed interface (rename here if theirs differ) - raw reads, Annex C columns:

    db.get_student(student_id)                 -> dict | None
    db.get_courses(programme)                  -> list[dict]   (course_code, course_name, ...)
    db.get_attendance(student_id, course_code) -> dict | None  (classes_held, classes_attended)
    db.get_results(student_id, course_code)    -> list[dict]   (exam_type, result, total_marks, ...)
    db.get_rules(parameter)                    -> list[dict]   (all rule_registry rows for that parameter)
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from datetime import date
from fractions import Fraction
from typing import Any, Optional

from models.schema import AppliedRule


def _db():
    """Load the SQL adapter only when a database-backed tool is invoked.

    The graph can therefore be assembled before the API/database package is
    installed, while a real tool invocation still fails clearly if it is absent.
    """
    from app.db import queries

    return queries

# rule_registry.parameter values. These are NAMES, never thresholds.
# Must match what your SQL teammate seeds in rule_registry.
PARAM = {
    "min_attendance": "min_attendance_pct",
    "placement_min_cgpa": "placement_min_cgpa",
    "placement_max_backlogs": "placement_max_active_backlogs",
}


@dataclass
class Ctx:
    student: dict
    as_of_date: date


@dataclass
class ToolResult:
    output: dict[str, Any]
    rules: list[AppliedRule] = field(default_factory=list)


def load_student(student_id: str) -> Optional[dict]:
    return _db().get_student(student_id)


# ---------------------------------------------------------------- rule helpers
def _d(v: Any) -> Optional[date]:
    try:
        return date.fromisoformat(str(v)[:10]) if v else None
    except ValueError:
        return None


def _in_scope(spec: Any, value: Any, batch: bool) -> bool:
    spec = str(spec or "ALL").strip()
    if spec.upper() == "ALL":
        return True
    if batch:
        m = re.fullmatch(r"(\d{4})\+", spec)
        if m:
            return int(value) >= int(m.group(1))
        return str(value) in {s.strip() for s in re.split(r"[;,]", spec)}
    return str(value).strip().lower() in {s.strip().lower() for s in re.split(r"[;,]", spec)}


def get_rule(ctx: Ctx, parameter: str) -> Optional[dict]:
    """The registry row in force on as_of_date for this student; latest effective_from wins."""
    ok = []
    for r in _db().get_rules(parameter):
        ef, et = _d(r.get("effective_from")), _d(r.get("effective_to"))
        if ef and ef > ctx.as_of_date:
            continue
        if et and et < ctx.as_of_date:
            continue
        if not _in_scope(r.get("scope_programmes"), ctx.student["programme"], False):
            continue
        if not _in_scope(r.get("scope_batches"), ctx.student["batch_year"], True):
            continue
        ok.append(r)
    return max(ok, key=lambda r: str(r.get("effective_from") or "")) if ok else None


def _frac(v: Any) -> Fraction:
    if isinstance(v, Fraction):
        return v
    return Fraction(re.sub(r"[^\d.\-]", "", str(v)))  # "75%" -> 75


def compare(actual: Any, op: str, value: Any) -> bool:
    """Exact comparison (Fractions, no float drift) so 'exactly at threshold' is right."""
    a, op = _frac(actual), op.strip()
    if op == "between":
        lo, hi = [_frac(x) for x in re.split(r"\s*(?:,|;|to)\s*", str(value).strip())]
        return lo <= a <= hi
    t = _frac(value)
    table = {">=": a >= t, ">": a > t, "<=": a <= t, "<": a < t, "==": a == t, "=": a == t}
    return table[op]


def _applied(rule: dict) -> AppliedRule:
    return AppliedRule(
        rule_id=rule["rule_id"],
        value=f"{rule['operator']} {rule['value']}",
        source_doc_id=rule["source_doc_id"],
        source_section=rule.get("source_section"),
    )


def _no_rule(parameter: str) -> ToolResult:
    return ToolResult({"error": "rule_not_found", "parameter": parameter})


# --------------------------------------------------------------- course lookup
_CODE = re.compile(r"^[A-Za-z]{2,4}\s?\d{3}[A-Za-z]?$")


def resolve_course(ctx: Ctx, text: str) -> tuple[Optional[dict], Optional[dict]]:
    """'CS201' or 'Data Structures' -> (course_row, None) or (None, error_dict)."""
    text = (text or "").strip()
    if not text:
        return None, {"error": "course_required"}
    courses = _db().get_courses(ctx.student["programme"])
    if _CODE.match(text):
        code = text.replace(" ", "").upper()
        for c in courses:
            if c["course_code"].upper() == code:
                return c, None
        return None, {"error": "course_not_found", "course": text}
    names = {c["course_name"].lower(): c for c in courses}
    low = text.lower()
    hits = [c for n, c in names.items() if low in n or n in low]
    if not hits:
        hits = [names[n] for n in difflib.get_close_matches(low, list(names), n=3, cutoff=0.6)]
    if len(hits) == 1:
        return hits[0], None
    if not hits:
        return None, {"error": "course_not_found", "course": text}
    return None, {"error": "course_ambiguous",
                  "candidates": [{"course_code": c["course_code"], "course_name": c["course_name"]} for c in hits]}


def _attendance(ctx: Ctx, course: str):
    c, err = resolve_course(ctx, course)
    if err:
        return None, None, err
    row = _db().get_attendance(ctx.student["student_id"], c["course_code"])
    if not row or not row.get("classes_held"):
        return c, None, {"error": "no_attendance_record", "course_code": c["course_code"]}
    return c, row, None


# ------------------------------------------------------------------------ tools
def get_student_profile(ctx: Ctx) -> ToolResult:
    s = ctx.student  # name / id deliberately left out (privacy, R7)
    return ToolResult({k: s[k] for k in ("programme", "batch_year", "current_semester", "cgpa", "active_backlogs")})


def get_attendance(ctx: Ctx, course: str) -> ToolResult:
    c, row, err = _attendance(ctx, course)
    if err:
        return ToolResult(err)
    held, att = row["classes_held"], row["classes_attended"]
    return ToolResult({"course_code": c["course_code"], "course_name": c["course_name"],
                       "classes_held": held, "classes_attended": att,
                       "attendance_pct": round(att * 100 / held, 2)})


def get_results(ctx: Ctx, course: str = "") -> ToolResult:
    code = ""
    if course:
        c, err = resolve_course(ctx, course)
        if err:
            return ToolResult(err)
        code = c["course_code"]
    rows = _db().get_results(ctx.student["student_id"], code)
    keep = ("course_code", "exam_session", "exam_type", "internal_marks", "external_marks",
            "total_marks", "max_marks", "result")
    return ToolResult({"results": [{k: r.get(k) for k in keep} for r in rows]})


def check_exam_eligibility(ctx: Ctx, course: str) -> ToolResult:
    c, row, err = _attendance(ctx, course)
    if err:
        return ToolResult(err)
    rule = get_rule(ctx, PARAM["min_attendance"])
    if not rule:
        return _no_rule(PARAM["min_attendance"])
    exact = Fraction(row["classes_attended"] * 100, row["classes_held"])
    ok = compare(exact, rule["operator"], rule["value"])
    return ToolResult({"course_code": c["course_code"], "attendance_pct": round(float(exact), 2),
                       "required": f"{rule['operator']} {rule['value']}",
                       "result": "ELIGIBLE" if ok else "NOT_ELIGIBLE", "rule_id": rule["rule_id"]},
                      [_applied(rule)])


def check_supplementary_eligibility(ctx: Ctx, course: str) -> ToolResult:
    c, err = resolve_course(ctx, course)
    if err:
        return ToolResult(err)
    rows = _db().get_results(ctx.student["student_id"], c["course_code"])
    regular = [r for r in rows if r.get("exam_type") == "REGULAR"]
    if not regular:
        return ToolResult({"error": "no_result_record", "course_code": c["course_code"]})
    failed = any(r["result"] in ("FAIL", "ABSENT") for r in regular)
    passed = any(r["result"] == "PASS" for r in rows)
    return ToolResult({"course_code": c["course_code"],
                       "regular_results": [r["result"] for r in regular],
                       "already_passed": passed,
                       "result": "ELIGIBLE" if failed and not passed else "NOT_ELIGIBLE"})


def check_placement_eligibility(ctx: Ctx, assume_cleared_backlogs: int = 0,
                                assume_cgpa: Optional[float] = None) -> ToolResult:
    """Also the what-if tool: pass assume_* to simulate; assumptions are echoed back."""
    s = ctx.student
    cleared = max(0, int(assume_cleared_backlogs or 0))
    cgpa = float(assume_cgpa) if assume_cgpa is not None else s["cgpa"]
    backlogs = max(0, s["active_backlogs"] - cleared)
    actuals = {"placement_min_cgpa": cgpa, "placement_max_backlogs": backlogs}

    checks, rules, missing = [], [], []
    for key, actual in actuals.items():
        rule = get_rule(ctx, PARAM[key])
        if not rule:
            missing.append(PARAM[key])
            continue
        checks.append({"rule_id": rule["rule_id"], "actual": actual,
                       "required": f"{rule['operator']} {rule['value']}",
                       "passed": compare(actual, rule["operator"], rule["value"])})
        rules.append(_applied(rule))
    if not checks:
        return _no_rule(", ".join(missing))

    assumptions = []
    if cleared:
        assumptions.append(f"{cleared} active backlog(s) cleared")
        if assume_cgpa is None:
            assumptions.append("CGPA unchanged by clearing the backlog(s)")
    if assume_cgpa is not None:
        assumptions.append(f"CGPA of {cgpa}")
    result = "UNDETERMINED" if missing else ("ELIGIBLE" if all(c["passed"] for c in checks) else "NOT_ELIGIBLE")
    return ToolResult({"result": result, "checks": checks, "unchecked_rules": missing,
                       "assumptions": assumptions}, rules)
