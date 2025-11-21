import pytest
from app.services.tracer import set_current_tracer, get_current_tracer, ToolTracer
from app.services.agent_tools import run_pipeline
from app.services.tracer import ToolTracer as _ToolTracerBase

class CapturingTracer(_ToolTracerBase):
    def __init__(self):
        self.events = []
    def on_tool_start(self, tool_name, payload):
        self.events.append(("start", tool_name, payload))
    def on_tool_progress(self, tool_name, payload):
        self.events.append(("progress", tool_name, payload))
    def on_tool_end(self, tool_name, result):
        self.events.append(("end", tool_name, result))
    def on_tool_error(self, tool_name, error):
        self.events.append(("error", tool_name, str(error)))

def _unwrap_tool_callable(tool_obj):
    """
    Given a langchain @tool-wrapped object, return the underlying Python callable.
    Tries common attribute names across LangChain versions.
    """
    for attr in ("func", "function", "__wrapped__", "run", "callable"):
        candidate = getattr(tool_obj, attr, None)
        if candidate and callable(candidate):
            return candidate
    # fallback: if the tool_obj itself is callable (some versions), return it
    if callable(tool_obj):
        return tool_obj
    raise RuntimeError("Could not find underlying callable for tool object: %r" % (tool_obj,))

def test_run_pipeline_emits_tracer(monkeypatch):
    # monkeypatch ResearchAgent.run to return a simple object with model_dump
    class DummyReport:
        def model_dump(self):
            return {"title": "mock", "executive_summary": "x"}
    class DummyRA:
        def __init__(self, namespace=None):
            pass
        def run(self, query, pdf_paths=None):
            return DummyReport()

    # point the agent_tools factory at our DummyRA
    monkeypatch.setattr("app.services.agent_tools._research_agent_cls", DummyRA)

    # prepare tracer and register
    t = CapturingTracer()
    set_current_tracer(t)

    try:
        # unwrap the run_pipeline tool to call the underlying function directly
        callable_run_pipeline = _unwrap_tool_callable(run_pipeline)

        # call the underlying implementation (same signature as original)
        out = callable_run_pipeline("test-query", pdf_paths=[])
        assert isinstance(out, dict)

        # ensure tracer saw start and end
        names = [e[0] for e in t.events]
        assert "start" in names and "end" in names
    finally:
        # cleanup tracer
        set_current_tracer(None)