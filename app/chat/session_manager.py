"""
In-memory session + report manager.

Minimal, thread-safe-ish store for local development.
Replace with Redis/Postgres for production.
"""

import uuid
import time
import threading
from typing import List, Dict, Any, Optional

_lock = threading.RLock()
_SESSIONS : Dict[str, Dict[str, Any]] = {}
_REPORTS: Dict[str, Dict[str, Any]] = {}

def create_session(user_id: Optional[str] = None) -> str:
    sid = f"session-{uuid.uuid4().hex}"
    with _lock:
        _SESSIONS[sid] = {
            "session-id": sid,
            "user-id": user_id,
            "history": [],  #list of {"role","text","ts"}
            "reports": [],  #list of report_ids generated in the session
            "created_at": time.time()
        }
    return sid

def append_message(session_id: str, role: str, text: str) -> None:
    with _lock:
        s = _SESSIONS.get(session_id)
        if s is None:
            raise KeyError(f"Session-id: {session_id} not found")
        s["history"].append({"role": role, "text": text, "ts": time.time()})
    
def get_history(session_id: str) -> List[Dict[str, Any]]:
    """
    Return session history (list). Raise KeyError if session not found.
    """
    with _lock:
        s = _SESSIONS.get(session_id)
        if s is None:
            raise KeyError(f"session {session_id} not found")
        hist = s.get("history")
        if hist is None:
            s["history"] = []
            hist = s["history"]
        return list(hist)
    
def store_report(session_id: Optional[str], report: Dict[str, Any]) -> str:
    """
    Store a report and associate it with session_id (if provided).
    Also creates simple chunking (paragraphs) for naive retrieval.
    Returns report_id.
    """
    rid = f"report-{uuid.uuid4().hex}"
    text = report.get("executive_summary", "") + "\n\n" + report.get("recommendations", "") if report else ""
    chunks: List[Dict[str, Any]] = []
    body = ""
    if isinstance(report, dict):
        pieces = []
        for k in ("executive_summary", "key_insights", "recommendations"):
            v = report.get(k)
            if not v:
                continue
            if isinstance(v, list):
                pieces.append("\n".join(v))
            else:
                pieces.append(str(v))
        body = "\n\n".join(pieces)
    else:
        body = str(report)
    
    raw_chunks = [c.strip() for c in body.split("\n\n") if c.strip()]
    if not raw_chunks:
        s = body or str(report)
        for i in range(0, len(s), 800):
            raw_chunks.append(s[i:i+800])

    for i, c in enumerate(raw_chunks):
        chunks.append({"id": f"{rid}-chunk-{i}", "text": c})

    entry = {
        "report_id": rid,
        "created_at": time.time(),
        "session_id": session_id,
        "report": report,
        "chunks": chunks,
    }

    with _lock:
        _REPORTS[rid] = entry
        if session_id:
            s = _SESSIONS.get(session_id)
            if s is not None:
                s["reports"].append(rid)
    return rid

def get_report(report_id: str) -> Dict[str, Any]:
    with _lock:
        r = _REPORTS.get(report_id)
        if r is None:
            raise KeyError(f"report {report_id} not found")
        return r

def list_reports_for_session(session_id: str) -> List[str]:
    with _lock:
        s = _SESSIONS.get(session_id)
        if s is None:
            raise KeyError(f"session {session_id} not found")
        return list(s["reports"])

def find_chunks_for_report(report_id: str) -> List[Dict[str, Any]]:
    r = get_report(report_id)
    return list(r.get("chunks", []))