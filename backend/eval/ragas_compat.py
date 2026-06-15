"""Compatibility patches for Ragas optional Vertex AI imports."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock


def ensure_ragas_importable() -> None:
    if "langchain_community.chat_models.vertexai" not in sys.modules:
        sys.modules["langchain_community.chat_models.vertexai"] = MagicMock()
    if "langchain_community.llms" not in sys.modules:
        sys.modules["langchain_community.llms"] = MagicMock()
