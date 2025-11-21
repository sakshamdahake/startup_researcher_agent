import time
import json
import uuid
import re
import traceback
from typing import Optional, Any, Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import redis
import os

from app.chat.session_manager import (
    create_session,
    append_message,
    get_history,
    store_report,
    get_report,
    list_reports_for_session,
)
from app.chat.qna import chat_reply
from app.services.agent_tools import run_pipeline as rc_run_pipeline_tool
from app.chat.qna import chat_stream

# Celery task
from app.tasks.worker_tasks import run_research_job

# Redis for pubsub & last-state
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
_rds = redis.Redis.from_url(REDIS_URL, decode_responses=True)

app = FastAPI(title="Startup Researcher API (minimal)", version="0.1")

# Helpers
CHUNK_ID_RE = re.compile(r"^chunk-[0-9a-fA-F\-]+$")

def _ensure_list_of_strings(x) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [str(i) for i in x]
    if isinstance(x, str):
        parts = [p.strip() for p in re.split(r"\n\s*\n|\r\n|\n|;|\||,", x) if p.strip()]
        return parts if len(parts) > 1 else [x.strip()]
    return [str(x)]

def _ensure_list_of_dicts(x) -> List[Dict[str, Any]]:
    if x is None:
        return []
    def _flatten(v):
        if isinstance(v, (list, tuple)):
            for i in v:
                yield from _flatten(i)
        else:
            yield v
    out: List[Dict[str, Any]] = []
    for item in _flatten(x):
        if isinstance(item, dict):
            out.append(item); continue
        if isinstance(item, str):
            s = item.strip()
            if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
                try:
                    parsed = json.loads(s)
                    if isinstance(parsed, dict):
                        out.append(parsed); continue
                    if isinstance(parsed, list):
                        out.extend(_ensure_list_of_dicts(parsed)); continue
                except Exception:
                    pass
            if CHUNK_ID_RE.match(s):
                out.append({"id": s}); continue
            if s.startswith("http://") or s.startswith("https://"):
                out.append({"url": s}); continue
            out.append({"value": s}); continue
        out.append({"value": str(item)})
    return out

def _resolve_chunk_ids(sources_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    resolved: List[Dict[str, Any]] = []
    for item in (sources_list or []):
        if not isinstance(item, dict):
            resolved.append(item); continue
        cid = item.get("id")
        if not cid or not isinstance(cid, str) or not CHUNK_ID_RE.match(cid):
            resolved.append(item); continue
        try:
            prefix = cid.split("-chunk-")[0] if "-chunk-" in cid else cid.split("-")[0]
            rep = get_report(prefix)
            for c in rep.get("chunks", []) or []:
                if c.get("id") == cid:
                    new_item = dict(item)
                    if "text" in c:
                        new_item["excerpt"] = c.get("text")
                    resolved.append(new_item)
                    break
            else:
                resolved.append(item)
        except Exception:
            resolved.append(item)
    return resolved

def _unwrap_tool_callable(tool_obj):
    for attr in ("func", "function", "__wrapped__", "run", "callable"):
        candidate = getattr(tool_obj, attr, None)
        if candidate and callable(candidate):
            return candidate
    if callable(tool_obj):
        return tool_obj
    raise RuntimeError("Could not find callable for tool object")

_run_pipeline = _unwrap_tool_callable(rc_run_pipeline_tool)

def _normalize_report_raw(report_raw: Any) -> Dict[str, Any]:
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
    report = dict(report or {})
    report["title"] = str(report.get("title") or report.get("name") or "Untitled Report")
    exec_sum = report.get("executive_summary") or report.get("summary") or ""
    if isinstance(exec_sum, list):
        exec_sum = "\n".join([str(i) for i in exec_sum])
    report["executive_summary"] = str(exec_sum)
    report["key_insights"] = _ensure_list_of_strings(report.get("key_insights") or report.get("insights") or "")
    evidence_raw = report.get("evidence") or []
    if isinstance(evidence_raw, list) and evidence_raw and isinstance(evidence_raw[0], str):
        report["evidence"] = [{"excerpt": s} for s in evidence_raw]
    else:
        report["evidence"] = evidence_raw if isinstance(evidence_raw, list) else []
    report["recommendations"] = _ensure_list_of_strings(report.get("recommendations") or report.get("next_steps") or "")
    report["sources"] = _ensure_list_of_dicts(report.get("sources") or report.get("references") or [])
    try:
        report["sources"] = _resolve_chunk_ids(report["sources"])
    except Exception:
        pass
    return report

# ----------------- Pydantic models -----------------
class CreateSessionResp(BaseModel):
    session_id: str

class ChatRequest(BaseModel):
    text: str
    use_retrieval: Optional[bool] = False
    report_id: Optional[str] = None

class ChatResponse(BaseModel):
    text: str
    sources: Optional[List[Dict[str, Any]]] = []
    report_id: Optional[str] = None

class ResearchRequest(BaseModel):
    query: str
    pdf_paths: Optional[List[str]] = []
    namespace: Optional[str] = "default"
    session_id: Optional[str] = None

# ----------------- endpoints -----------------

@app.get("/api/v1/chat_stream/{session_id}")
def api_chat_stream(session_id: str, text: str, use_retrieval: bool = False, report_id: Optional[str] = None):
    gen = chat_stream(session_id=session_id, user_text=text, use_retrieval=use_retrieval, report_id=report_id)
    return StreamingResponse(gen, media_type="text/event-stream")

@app.get("/ping")
def ping():
    return {"ok": True, "ts": time.time()}

@app.post("/api/v1/session", response_model=CreateSessionResp)
def api_create_session():
    sid = create_session()
    return CreateSessionResp(session_id=sid)

@app.get("/api/v1/session/{session_id}/history")
def api_history(session_id: str):
    try:
        return get_history(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="session not found")

@app.post("/api/v1/chat/{session_id}", response_model=ChatResponse)
def api_chat(session_id: str, req: ChatRequest):
    try:
        resp = chat_reply(session_id=session_id, user_text=req.text, use_retrieval=req.use_retrieval, report_id=req.report_id)
        return ChatResponse(text=resp["text"], sources=resp.get("sources", []), report_id=resp.get("report_id"))
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# ----------------- Async research: enqueue + stream -----------------
@app.post("/api/v1/research_async")
def api_research_async(req: ResearchRequest):
    """
    Enqueue a background research job. Returns job_id immediately.
    Client can then open SSE channel /api/v1/stream/{job_id}.
    """
    try:
        job_id = f"job-{uuid.uuid4().hex}"
        # enqueue the Celery task (async)
        run_research_job.delay(job_id, req.query, req.pdf_paths or [], req.namespace, req.session_id)
        return {"job_id": job_id}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/status/{job_id}")
def api_status(job_id: str):
    """
    Return last-known event for the job (if any).
    Reads Redis key job:{job_id}:state which worker sets.
    """
    try:
        txt = _rds.get(f"job:{job_id}:state")
        if not txt:
            return {"job_id": job_id, "state": None}
        return {"job_id": job_id, "state": json.loads(txt)}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

def event_stream(job_id: str):
    """
    Subscribe to Redis channel and yield Server-Sent Events.
    """
    pubsub = _rds.pubsub()
    channel = f"job:{job_id}"
    pubsub.subscribe(channel)
    try:
        for message in pubsub.listen():
            if message is None:
                continue
            if message.get("type") != "message":
                continue
            data = message.get("data")
            try:
                if isinstance(data, bytes):
                    data = data.decode("utf8")
            except Exception:
                pass
            yield f"data: {data}\n\n"
            try:
                obj = json.loads(data)
                if obj.get("type") in ("done", "error"):
                    break
            except Exception:
                pass
    finally:
        try:
            pubsub.unsubscribe(channel)
            pubsub.close()
        except Exception:
            pass

@app.get("/api/v1/stream/{job_id}")
def api_stream(job_id: str):
    return StreamingResponse(event_stream(job_id), media_type="text/event-stream")

# ----------------- legacy synchronous research endpoint -----------------
@app.post("/api/v1/research_sync")
def api_research_sync(req: ResearchRequest):
    """
    Blocking (synchronous) research call — existing behavior preserved.
    """
    try:
        report_raw = _run_pipeline(req.query, pdf_paths=req.pdf_paths or [], namespace=req.namespace)
        report = _normalize_report_raw(report_raw)
        if not isinstance(report.get("key_insights"), list) or not isinstance(report.get("sources"), list):
            raise HTTPException(status_code=500, detail="Failed to normalize report into expected schema.")
        storage_report = dict(report)
        if isinstance(storage_report.get("executive_summary"), list):
            storage_report["executive_summary"] = "\n".join(storage_report["executive_summary"])
        recs = storage_report.get("recommendations", "")
        if isinstance(recs, list):
            storage_report["recommendations"] = "\n".join([str(x) for x in recs])
        else:
            storage_report["recommendations"] = str(recs)
        rid = store_report(req.session_id, storage_report)
        return {"report_id": rid, "report": report}
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Report generation failed: {e}")

@app.get("/api/v1/report/{report_id}")
def api_get_report(report_id: str):
    try:
        r = get_report(report_id)
        return {"report_id": report_id, "report": r["report"], "chunks": r["chunks"]}
    except KeyError:
        raise HTTPException(status_code=404, detail="report not found")