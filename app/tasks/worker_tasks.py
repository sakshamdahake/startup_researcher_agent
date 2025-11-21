# app/tasks/worker_tasks.py
import os
import json
import time
from typing import Any, Dict, List
from app.tasks.celery_app import celery_app
from app.services.tracer import set_current_tracer, get_current_tracer

import redis

# Import the run_pipeline tool callable (unwrap if decorated)
from app.services.agent_tools import run_pipeline as rc_run_pipeline_tool

# Redis connection used for pub/sub and store last state
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
_rds = redis.from_url(REDIS_URL, decode_responses=True)

# find callable behind tool decorator (same helper as in main)
def _unwrap_tool_callable(tool_obj):
    for attr in ("func", "function", "__wrapped__", "run", "callable"):
        candidate = getattr(tool_obj, attr, None)
        if candidate and callable(candidate):
            return candidate
    if callable(tool_obj):
        return tool_obj
    raise RuntimeError("Could not find callable for tool object")

_run_pipeline = _unwrap_tool_callable(rc_run_pipeline_tool)

def _publish(job_id: str, event: Dict[str, Any]) -> None:
    """
    Publish event to job channel AND store last state in job:{job_id}:state
    Event should be JSON-serializable.
    """
    channel = f"job:{job_id}"
    payload = dict(event)
    payload["ts"] = time.time()
    txt = json.dumps(payload, default=str)
    # pubsub
    try:
        _rds.publish(channel, txt)
    except Exception:
        # pubsub should be best-effort
        pass
    # store last state (for status endpoint)
    try:
        _rds.set(f"job:{job_id}:state", txt, ex=60*60)  # keep 1h by default
    except Exception:
        pass


@celery_app.task(bind=True, name="app.tasks.worker_tasks.run_research_job")
def run_research_job(self, job_id: str, query: str, pdf_paths: List[str], namespace: str, session_id: str = None):
    """
    Celery task: runs the canonical run_pipeline callable and publishes events as it runs.

    This task attaches a RedisTracer before calling the pipeline so that internal tracer events
    are forwarded to Redis pubsub (channel job:{job_id}). This enables SSE clients to see progress.
    """
    # small tracer that forwards events to _publish(job_id, event)
    class RedisTracer:
        def __init__(self, job_id: str):
            self.job_id = job_id

        def _send(self, payload: Dict[str, Any]):
            try:
                _publish(self.job_id, payload)
            except Exception:
                # never raise from tracer; swallow to avoid breaking pipeline
                pass

        # called when a tool is started; input can be any serializable payload
        def on_tool_start(self, tool_name, input_payload=None, **kwargs):
            self._send({"type": "tool_start", "tool": str(tool_name), "input": input_payload})

        def on_tool_progress(self, tool_name, progress_payload=None, **kwargs):
            payload = {"type": "tool_progress", "tool": str(tool_name)}
            if isinstance(progress_payload, dict):
                payload.update(progress_payload)
            else:
                payload["msg"] = str(progress_payload)
            self._send(payload)

        def on_tool_end(self, tool_name, output_payload=None, **kwargs):
            out_short = None
            try:
                if isinstance(output_payload, (str, int, float)):
                    out_short = output_payload
                else:
                    out_short = str(output_payload)[:1000]
            except Exception:
                out_short = "<non-serializable>"
            self._send({"type": "tool_end", "tool": str(tool_name), "output": out_short})

        def on_tool_error(self, tool_name, exc):
            self._send({"type": "tool_error", "tool": str(tool_name), "error": str(exc)})

    tracer = RedisTracer(job_id)

    # attach tracer so run_pipeline and tools can call get_current_tracer()
    try:
        set_current_tracer(tracer)
    except Exception:
        # If tracer module missing or fails, continue without it
        pass

    try:
        _publish(job_id, {"type": "step", "state": "started", "detail": "task started"})
        _publish(job_id, {"type": "progress", "pct": 1, "msg": "initializing pipeline"})

        # Run the pipeline (this may call tools/tracer internally)
        report_raw = _run_pipeline(query, pdf_paths=pdf_paths or [], namespace=namespace)

        # Normalize report to JSON-friendly dict
        if hasattr(report_raw, "model_dump"):
            report = report_raw.model_dump()
        elif hasattr(report_raw, "dict"):
            report = report_raw.dict()
        elif isinstance(report_raw, dict):
            report = report_raw
        else:
            try:
                report = json.loads(report_raw) if isinstance(report_raw, str) else {"raw": str(report_raw)}
            except Exception:
                report = {"raw": str(report_raw)}

        # Publish completion & report
        _publish(job_id, {"type": "progress", "pct": 100, "msg": "pipeline complete"})
        _publish(job_id, {"type": "done", "report": report})
        return {"status": "ok", "report_id": report.get("report_id") or report.get("id")}

    except Exception as exc:
        _publish(job_id, {"type": "error", "error": str(exc)})
        raise

    finally:
        # clear tracer for this context (best-effort)
        try:
            set_current_tracer(None)
        except Exception:
            pass