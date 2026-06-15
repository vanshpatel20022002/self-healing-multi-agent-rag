"""Verify LangGraph workflow compiles and routes correctly."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.graph.workflow import build_graph, route_after_validation  # noqa: E402


def test_routing() -> None:
    assert route_after_validation({"validation_passed": True, "retry_count": 0}) == "end"
    assert route_after_validation({"validation_passed": False, "retry_count": 0}) == "retry"
    assert route_after_validation({"validation_passed": False, "retry_count": 3}) == "end"


def test_graph_compiles() -> None:
    app = build_graph()
    assert app is not None


def main() -> None:
    test_routing()
    test_graph_compiles()
    print("Workflow routing and compilation checks passed.")


if __name__ == "__main__":
    main()
