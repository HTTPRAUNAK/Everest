"""SQLite queries and idempotent initialization for the Annex C schema."""
from __future__ import annotations

import json
import os
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATABASE_PATH = Path(
    os.getenv("DATABASE_PATH", str(PROJECT_ROOT / "data" / "university.db"))
)
STUDENT_JSON_PATH = Path(
    os.getenv("STUDENT_JSON_PATH", str(PROJECT_ROOT / "data" / "dtu_demo_students.json"))
)

_init_lock = threading.Lock()
_initialized_path: Path | None = None

_SCHEMA = """
CREATE TABLE IF NOT EXISTS students (
    student_id TEXT PRIMARY KEY CHECK (student_id GLOB 'S[0-9][0-9][0-9][0-9]'),
    full_name TEXT NOT NULL,
    programme TEXT NOT NULL,
    batch_year INTEGER NOT NULL,
    current_semester INTEGER NOT NULL CHECK (current_semester BETWEEN 1 AND 10),
    cgpa REAL CHECK (cgpa IS NULL OR cgpa BETWEEN 0.0 AND 10.0),
    active_backlogs INTEGER NOT NULL DEFAULT 0 CHECK (active_backlogs >= 0)
);

CREATE TABLE IF NOT EXISTS courses (
    course_code TEXT PRIMARY KEY,
    course_name TEXT NOT NULL,
    programme TEXT NOT NULL,
    semester INTEGER NOT NULL,
    credits INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS attendance (
    student_id TEXT NOT NULL REFERENCES students(student_id),
    course_code TEXT NOT NULL REFERENCES courses(course_code),
    classes_held INTEGER NOT NULL CHECK (classes_held > 0),
    classes_attended INTEGER NOT NULL CHECK (classes_attended BETWEEN 0 AND classes_held),
    PRIMARY KEY (student_id, course_code)
);

CREATE TABLE IF NOT EXISTS results (
    student_id TEXT NOT NULL REFERENCES students(student_id),
    course_code TEXT NOT NULL REFERENCES courses(course_code),
    exam_session TEXT NOT NULL,
    exam_type TEXT NOT NULL CHECK (exam_type IN ('REGULAR', 'SUPPLEMENTARY')),
    internal_marks INTEGER NOT NULL,
    external_marks INTEGER NOT NULL,
    total_marks INTEGER NOT NULL,
    max_marks INTEGER NOT NULL,
    result TEXT NOT NULL CHECK (result IN ('PASS', 'FAIL', 'ABSENT', 'DETAINED')),
    PRIMARY KEY (student_id, course_code, exam_session, exam_type),
    CHECK (total_marks = internal_marks + external_marks),
    CHECK (internal_marks BETWEEN 0 AND max_marks),
    CHECK (external_marks BETWEEN 0 AND max_marks),
    CHECK (total_marks BETWEEN 0 AND max_marks)
);

CREATE TABLE IF NOT EXISTS rule_registry (
    rule_id TEXT PRIMARY KEY,
    description TEXT NOT NULL,
    parameter TEXT NOT NULL,
    operator TEXT NOT NULL,
    value TEXT NOT NULL,
    scope_programmes TEXT NOT NULL DEFAULT 'ALL',
    scope_batches TEXT NOT NULL DEFAULT 'ALL',
    effective_from TEXT NOT NULL,
    effective_to TEXT,
    source_doc_id TEXT NOT NULL,
    source_section TEXT
);
"""


def initialize_database() -> None:
    """Create tables and insert available demo students without overwriting data."""
    global _initialized_path
    db_path = DATABASE_PATH.resolve()
    if _initialized_path == db_path:
        return

    with _init_lock:
        if _initialized_path == db_path:
            return
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(db_path) as connection:
            connection.executescript(_SCHEMA)
            if STUDENT_JSON_PATH.exists():
                records = json.loads(STUDENT_JSON_PATH.read_text(encoding="utf-8"))
                connection.executemany(
                    """INSERT OR IGNORE INTO students
                       (student_id, full_name, programme, batch_year,
                        current_semester, cgpa, active_backlogs)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    [
                        (
                            item["student_id"],
                            item["full_name"],
                            item["programme"],
                            item["batch_year"],
                            item["current_semester"],
                            item.get("cgpa"),
                            item["active_backlogs"],
                        )
                        for item in records
                    ],
                )
        _initialized_path = db_path


@contextmanager
def _connection() -> Iterator[sqlite3.Connection]:
    initialize_database()
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    try:
        yield connection
    finally:
        connection.close()


def _dict(row: sqlite3.Row | None) -> dict | None:
    return dict(row) if row is not None else None


def get_student(student_id: str) -> dict | None:
    with _connection() as connection:
        row = connection.execute(
            "SELECT * FROM students WHERE student_id = ?",
            (student_id.strip().upper(),),
        ).fetchone()
        return _dict(row)


def get_courses(programme: str) -> list[dict]:
    with _connection() as connection:
        rows = connection.execute(
            "SELECT * FROM courses WHERE programme = ? ORDER BY course_code",
            (programme,),
        ).fetchall()
        return [dict(row) for row in rows]


def get_attendance(student_id: str, course_code: str) -> dict | None:
    with _connection() as connection:
        row = connection.execute(
            "SELECT * FROM attendance WHERE student_id = ? AND course_code = ?",
            (student_id, course_code),
        ).fetchone()
        return _dict(row)


def get_results(student_id: str, course_code: str = "") -> list[dict]:
    with _connection() as connection:
        if course_code:
            rows = connection.execute(
                """SELECT * FROM results
                   WHERE student_id = ? AND course_code = ?
                   ORDER BY exam_session, exam_type""",
                (student_id, course_code),
            ).fetchall()
        else:
            rows = connection.execute(
                """SELECT * FROM results WHERE student_id = ?
                   ORDER BY exam_session, course_code, exam_type""",
                (student_id,),
            ).fetchall()
        return [dict(row) for row in rows]


def get_rules(parameter: str) -> list[dict]:
    with _connection() as connection:
        rows = connection.execute(
            "SELECT * FROM rule_registry WHERE parameter = ? ORDER BY effective_from",
            (parameter,),
        ).fetchall()
        return [dict(row) for row in rows]
