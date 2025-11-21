"""
LangChain create_agent runner with minimal, robust signature handling.

- Inspects the langchain.create_agent signature at runtime and passes only accepted kwargs.
- Attaches a SimpleTracer via CallbackManager (your environment uses langchain_core callbacks).
- Exposes create_agent_runner() and run_agent().
"""

from typing import Dict, List, Any, Optional
import time
import json
import logging
import inspect

from app.llm.openai_llm import get_llm
from app.services.agent_tools import TOOLS
from app.services.tracer import set_current_tracer, get_current_tracer, ToolTracer as _ToolTracerInterface

from langchain.agents import create_agent
from langchain_core.callbacks.manager import CallbackManager
from langchain_core.callbacks.base import BaseCallbackHandler

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class SimpleTracer(BaseCallbackHandler):
    def __init__(self):
        try:
            super().__init__()
        except Exception:
            pass
        self.events: List[Dict[str, Any]] = []

    def on_tool_start(self, tool_input: Dict[str, Any], **kwargs) -> None:
        try:
            if isinstance(tool_input, dict):
                name = tool_input.get("name") or tool_input.get("tool") or "<unknown>"
            else:
                name = getattr(tool_input, "name", str(tool_input))
        except Exception:
            name = "<unknown>"
        self.events.append({"type": "tool_start", "tool": name, "input": tool_input})

    def on_tool_end(self, output: Any, **kwargs) -> None:
        name = kwargs.get("tool") or kwargs.get("name") or "<unknown>"
        try:
            out_short = output if isinstance(output, (str, int, float)) else str(output)[:1000]
        except Exception:
            out_short = "<non-serializable-output>"
        self.events.append({"type": "tool_end", "tool": name, "output": out_short})

    def on_agent_end(self, **kwargs) -> None:
        self.events.append({"type": "agent_end", **(kwargs or {})})

    def tool_started(self, tool_input: Dict[str, Any], **kwargs):
        return self.on_tool_start(tool_input, **kwargs)

    def tool_finished(self, output: Any, **kwargs):
        return self.on_tool_end(output, **kwargs)


def _normalize_agent_output(result: Any) -> str:
    if result is None:
        return ""
    if isinstance(result, str):
        return result
    try:
        if hasattr(result, "model_dump"):
            dump = result.model_dump()
            for k in ("output", "text", "content", "result", "final"):
                if k in dump and dump[k]:
                    return str(dump[k])
            return json.dumps(dump, default=str)[:4000]
    except Exception:
        pass
    if hasattr(result, "content"):
        try:
            return str(getattr(result, "content"))
        except Exception:
            pass
    if hasattr(result, "generations"):
        try:
            gens = getattr(result, "generations")
            if gens and isinstance(gens, list):
                first = gens[0]
                if isinstance(first, list) and first:
                    return str(getattr(first[0], "text", first[0]))
                return str(getattr(first[0], "text", first[0]))
        except Exception:
            pass
    if isinstance(result, dict):
        for k in ("output", "text", "result", "final"):
            if k in result and result[k]:
                return str(result[k])
        try:
            return json.dumps(result, default=str)[:4000]
        except Exception:
            pass
    try:
        return str(result)
    except Exception:
        return "<unserializable-result>"


def _call_create_agent_with_supported_kwargs(llm, tools, system_prompt, cb_manager):
    """
    Inspect create_agent signature and call it with only supported kwargs.
    Returns the created agent.
    """
    sig = inspect.signature(create_agent)
    supported = set(sig.parameters.keys())

    candidates = {
        "model": llm,
        "llm": llm,
        "model_name": llm,
        "tools": tools,
        "tool_list": tools,
        "system_message": system_prompt,
        "system_prompt": system_prompt,
        "prompt": system_prompt,
        "callbacks": cb_manager,
        "callback_manager": cb_manager,
    }

    call_kwargs = {k: v for k, v in candidates.items() if k in supported and v is not None}

    try:
        return create_agent(**call_kwargs)
    except TypeError as e:
        logger.debug("create_agent(**call_kwargs) failed, trying positional fallbacks: %s", e)
        try:
            return create_agent(llm, tools)
        except Exception:
            pass
        try:
            return create_agent(llm, tools)
        except Exception:
            pass
        try:
            return create_agent(llm)
        except Exception as ee:
            raise RuntimeError(f"Failed to call create_agent with filtered kwargs and positional fallbacks: {ee}") from ee


def create_agent_runner(llm: Optional[Any] = None, tools: Optional[List[Any]] = None, system_prompt: Optional[str] = None):
    """
    Create and return a LangChain agent (modern API) with a SimpleTracer attached.
    This version explicitly passes the TOOLS export from app.services.agent_tools so
    the runtime can wire @tool-decorated functions into the agent execution graph.
    """
    from app.services.agent_tools import TOOLS as AGENT_TOOLS

    llm = llm or get_llm()
    tools = tools or AGENT_TOOLS

    if system_prompt is None:
        system_prompt = (
            "You are ResearchAgent (planner). Decide which tools to call and in which order "
            "to answer the user's research query. Prefer evidence-backed steps: search the web, "
            "fetch pages, ingest PDFs if available, index, retrieve, summarize chunks, synthesize and produce a final report. "
            "When a final report is requested, you may call 'run_pipeline' for a canonical pipeline."
        )

    tracer = SimpleTracer()
    cb_manager = CallbackManager([tracer])

    agent = _call_create_agent_with_supported_kwargs(llm=llm, tools=tools, system_prompt=system_prompt, cb_manager=cb_manager)

    setattr(agent, "_tracer", tracer)

    try:
        names = []
        for t in tools:
            name = getattr(t, "name", None) or getattr(t, "__name__", None) or str(t)
            names.append(name)
        setattr(agent, "registered_tools", names)
    except Exception:
        setattr(agent, "registered_tools", None)

    return agent


def _invoke_agent(agent: Any, query: str):
    try:
        return agent.invoke({"input": query})
    except Exception:
        pass
    try:
        return agent.invoke(query)
    except Exception:
        pass
    try:
        return agent(query)
    except Exception as e:
        raise RuntimeError("Agent invocation failed for all tried call styles.") from e


def run_agent(agent: Any, query: str, max_attempts: int = 1, verbose: bool = True) -> Dict[str, Any]:
    """
    Run agent, attach adapter to map agent._tracer -> tool tracer interface, run, then cleanup.
    """
    last_err = None

    class _Adapter:
        def __init__(self, cbhandler):
            self._cb = cbhandler

        def on_tool_start(self, tool_name: str, payload: Dict[str, Any]):
            try:
                if hasattr(self._cb, "on_tool_start"):
                    try:
                        self._cb.on_tool_start(tool_name, payload)
                    except TypeError:
                        self._cb.on_tool_start({"name": tool_name}, payload)
                else:
                    if hasattr(self._cb, "events"):
                        self._cb.events.append({"type": "tool_start", "tool": tool_name, "input": payload})
            except Exception:
                pass

        def on_tool_progress(self, tool_name: str, payload: Dict[str, Any]):
            try:
                if hasattr(self._cb, "events"):
                    self._cb.events.append({"type": "tool_progress", "tool": tool_name, "payload": payload})
            except Exception:
                pass

        def on_tool_end(self, tool_name: str, result: Any):
            try:
                if hasattr(self._cb, "on_tool_end"):
                    try:
                        self._cb.on_tool_end(tool_name, result)
                    except TypeError:
                        self._cb.on_tool_end({"name": tool_name}, result)
                else:
                    if hasattr(self._cb, "events"):
                        self._cb.events.append({"type": "tool_end", "tool": tool_name, "output": str(result)[:1000]})
            except Exception:
                pass

        def on_tool_error(self, tool_name: str, error: Exception):
            try:
                if hasattr(self._cb, "events"):
                    self._cb.events.append({"type": "tool_error", "tool": tool_name, "error": str(error)})
            except Exception:
                pass

    tracer_cb = getattr(agent, "_tracer", None)
    adapter = _Adapter(tracer_cb) if tracer_cb is not None else None

    set_current_tracer(adapter)

    try:
        for attempt in range(1, max_attempts + 1):
            try:
                start = time.time()
                raw = _invoke_agent(agent, query)
                duration = time.time() - start

                text = _normalize_agent_output(raw)
                out: Dict[str, Any] = {
                    "query": query,
                    "raw_result": raw,
                    "text": text,
                    "duration_sec": duration,
                }

                tracer = getattr(agent, "_tracer", None)
                if tracer is not None and getattr(tracer, "events", None) is not None:
                    out["trace"] = tracer.events

                if verbose:
                    logger.info("Agent completed in %.2fs", duration)
                    logger.info("Agent text (first 800 chars):\n%s", text[:800])

                return out

            except Exception as e:
                last_err = e
                logger.warning("Agent run failed on attempt %d/%d: %s", attempt, max_attempts, e)
                if attempt < max_attempts:
                    time.sleep(1 * attempt)
                    continue
                tracer = getattr(agent, "_tracer", None)
                trace = getattr(tracer, "events", None) if tracer is not None else None
                raise RuntimeError({"error": str(e), "trace": trace}) from e
    finally:
        set_current_tracer(None)

# CLI entrypoint
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run LangChain create_agent-based runner locally.")
    parser.add_argument("--query", "-q", type=str, required=True, help="Research query to run")
    parser.add_argument("--debug", action="store_true", help="Print more debug info")
    args = parser.parse_args()

    agent = create_agent_runner()
    try:
        result = run_agent(agent, args.query, verbose=not args.debug)
        print("=== Agent final text ===")
        print(result["text"])
        if "trace" in result:
            print("\n=== Agent trace ===")
            for ev in result["trace"]:
                print(ev)
    except Exception as exc:
        print("Agent run failed:", exc)
        raise