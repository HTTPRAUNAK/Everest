"""Small CLI for invoking the compiled assistant graph locally."""
from __future__ import annotations

import argparse
import os
import time
from datetime import date


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the Everest student-services agent")
    parser.add_argument("--student-id", help="Authenticated student ID for personal questions")
    parser.add_argument("--mock", action="store_true", help="Use deterministic local LLM fallbacks")
    parser.add_argument("--offline", action="store_true", help="Use cached embedding models only")
    parser.add_argument("question", nargs="+", help="Question to ask")
    args = parser.parse_args()

    if args.mock:
        os.environ["MOCK_LLM"] = "true"
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    # Import after flags are applied because config and retrieval read environment
    # settings when their modules load.
    from agents.graph import graph

    result = graph.invoke(
        {
            "question": " ".join(args.question),
            "student_id": args.student_id,
            "as_of_date": date.today(),
            "started_at": time.perf_counter(),
        }
    )
    answer_type = result.get("answer_type", "unknown")
    print(f"[{getattr(answer_type, 'value', answer_type)}]")
    print(result.get("answer", "No answer was produced."))


if __name__ == "__main__":
    main()
