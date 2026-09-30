"""Tiny helper for Server-Sent Events formatting."""
import json
from typing import Any


def format_sse(event: str, data: dict[str, Any]) -> str:
    """Return one SSE frame. JSON-encoding keeps newlines inside tokens safe."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
