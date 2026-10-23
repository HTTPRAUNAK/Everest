"""
Tool registry + safe runner for the Tools node.

Security (R7): student_id is NEVER taken from the LLM's args. run_tool() gets it
from the request context, and any unexpected arg (including 'student_id') is dropped.
"""
from __future__ import annotations

import inspect
import time
from datetime import date

from agents import sql_tools as t
from models.schema import AppliedRule, ToolCall

TOOLS = {
    "get_student_profile": (t.get_student_profile,
        "Programme, batch, semester, CGPA, active backlogs. args: none"),
    "get_attendance": (t.get_attendance,
        "Attendance % for one course. args: course (code or name)"),
    "get_results": (t.get_results,
        "Exam results. args: course (optional, code or name)"),
    "check_exam_eligibility": (t.check_exam_eligibility,
        "Eligible to sit end-sem exam in a course (attendance rule). args: course"),
    "check_supplementary_eligibility": (t.check_supplementary_eligibility,
        "Eligible for the supplementary exam in a course. args: course"),
    "check_placement_eligibility": (t.check_placement_eligibility,
        "Placement eligibility; also what-if. args: assume_cleared_backlogs (int, optional), assume_cgpa (float, optional)"),
}


def tool_catalog() -> str:
    """Text block for the classify prompt."""
    return "\n".join(f"- {name}: {desc}" for name, (_, desc) in TOOLS.items())


def run_tool(name: str, args: dict, student_id: str | None, as_of_date: date) -> tuple[ToolCall, list[AppliedRule]]:
    t0 = time.perf_counter()
    clean: dict = {}
    output: dict
    rules: list[AppliedRule] = []
    try:
        if name not in TOOLS:
            raise KeyError("unknown_tool")
        fn = TOOLS[name][0]
        allowed = set(inspect.signature(fn).parameters) - {"ctx"}
        clean = {k: v for k, v in (args or {}).items() if k in allowed}
        if not student_id:
            output = {"error": "no_identity"}
        else:
            student = t.load_student(student_id)
            if not student:
                output = {"error": "student_not_found"}
            else:
                res = fn(t.Ctx(student=student, as_of_date=as_of_date), **clean)
                output, rules = res.output, res.rules
    except Exception as e:  # noqa: BLE001 - tool failures must not crash the graph
        output = {"error": f"{type(e).__name__}"}
    ms = int((time.perf_counter() - t0) * 1000)
    status = "error" if "error" in output else "ok"
    return ToolCall(tool=name, input=clean, output=output, status=status, ms=ms), rules
