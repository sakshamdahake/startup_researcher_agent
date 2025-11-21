# app/services/tracer.py
from typing import Optional, Dict, Any
from contextvars import ContextVar

# ContextVar is safe for threading and async contexts and is reliable in tests.
_current_tracer: ContextVar[Optional["ToolTracer"]] = ContextVar("_current_tracer", default=None)

class ToolTracer:
    """
    Minimal tool-level tracer interface. Tools call these methods if a tracer is available.
    Subclass this in tests or production to capture/forward events.
    """

    def on_tool_start(self, tool_name: str, payload: Dict[str, Any]) -> None:
        """Called at the beginning of a tool run."""
        pass

    def on_tool_progress(self, tool_name: str, payload: Dict[str, Any]) -> None:
        """Optional progress updates from a tool."""
        pass

    def on_tool_end(self, tool_name: str, result: Any) -> None:
        """Called on successful completion."""
        pass

    def on_tool_error(self, tool_name: str, error: Exception) -> None:
        """Called on exceptions raised inside tools."""
        pass

def set_current_tracer(tracer: Optional[ToolTracer]) -> None:
    """Set tracer for current context (thread/coroutine)."""
    _current_tracer.set(tracer)

def get_current_tracer() -> Optional[ToolTracer]:
    """Return the current tracer for this context, or None."""
    return _current_tracer.get()