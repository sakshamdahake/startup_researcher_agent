from typing import Optional, List, Dict, Any, Generator
from app.chat.session_manager import append_message, get_history, find_chunks_for_report
from app.llm.openai_llm import get_llm
from langchain_core.messages import HumanMessage
import re
import json
import time

try:
    from app.services.tools import DBRetrieverTool
    _vector_retriever = DBRetrieverTool()
except Exception:
    _vector_retriever = None

SYSTEM_PROMPT = (
    "You are ResearchAssistant. Be concise and evidence-backed. "
    "When evidence is provided, cite it in square brackets like [source-id]."
)

# LLM calling helpers

def _extract_text_from_resp(r: Any) -> str:
    """Normalize a chunk/response into a plain text fragment (best-effort)."""
    if r is None:
        return ""

    for attr in ("content", "text"):
        if hasattr(r, attr):
            val = getattr(r, attr)
            if isinstance(val, str):
                return val
            try:
                return str(val)
            except Exception:
                pass

    if isinstance(r, dict):
        if "delta" in r and isinstance(r["delta"], dict):
            d = r["delta"]
            for k in ("content", "text"):
                if k in d and isinstance(d[k], str):
                    return d[k]
        if "choices" in r:
            try:
                ch = r["choices"]
                if isinstance(ch, (list, tuple)) and ch:
                    first = ch[0]
                    if isinstance(first, dict):
                        for k in ("message", "delta", "text", "content"):
                            if k in first:
                                sub = first[k]
                                if isinstance(sub, dict):
                                    for kk in ("content", "text"):
                                        if kk in sub and isinstance(sub[kk], str):
                                            return sub[kk]
                                if isinstance(sub, str):
                                    return sub
                    if isinstance(first, dict) and "text" in first and isinstance(first["text"], str):
                        return first["text"]
            except Exception:
                pass
        for k in ("content", "text", "message"):
            if k in r and isinstance(r[k], str):
                return r[k]

    if isinstance(r, (list, tuple)):
        for item in r:
            if isinstance(item, (list, tuple)) and len(item) == 2 and isinstance(item[0], str):
                k, v = item
                if k.lower() in ("content", "text", "message") and isinstance(v, str):
                    return v
            if isinstance(item, dict):
                txt = _extract_text_from_resp(item)
                if txt:
                    return txt
            if isinstance(item, str):
                return item

    try:
        if hasattr(r, "to_dict"):
            d = r.to_dict()
            return _extract_text_from_resp(d)
        if hasattr(r, "__dict__"):
            return _extract_text_from_resp(vars(r))
    except Exception:
        pass

    try:
        return str(r)
    except Exception:
        return ""

def _call_llm(llm: Any, prompt_text: str) -> str:
    """
    Safe wrapper to call LLM synchronously. Tries several patterns for compatibility check.
    Returns the full assistant text.
    """
    try:
        r = llm(messages=[HumanMessage(prompt_text)])
        return _extract_text_from_resp(r).strip()
    except Exception:
        pass

    try:
        r = llm(prompt_text)
        return _extract_text_from_resp(r).strip()
    except Exception:
        pass

    try:
        if hasattr(llm, "invoke"):
            r = llm.invoke([HumanMessage(prompt_text)])
            return _extract_text_from_resp(r).strip()
        if hasattr(llm, "generate"):
            r = llm.generate([HumanMessage(prompt_text)])
            return _extract_text_from_resp(r).strip()
    except Exception as e:
        raise RuntimeError(f"LLM call failed: {e}") from e

    raise RuntimeError("LLM object has no compatible call method")


def _call_llm_stream(llm: Any, prompt_text: str):
    """
    Generator yielding text fragments only (best-effort streaming).
    Tries several streaming APIs and normalizes chunks using _extract_text_from_resp.
    If no streaming interface is present, yields the full sync output once.
    """
    stream_methods = [
        "stream", "stream_chat", "stream_messages", "stream_generate", "stream_completion",
        "stream_responses", "streaming", "_stream"
    ]

    for name in stream_methods:
        method = getattr(llm, name, None)
        if callable(method):
            try:
                for chunk in method(messages=[HumanMessage(prompt_text)]):
                    text = _extract_text_from_resp(chunk)
                    if text:
                        yield text
                return
            except TypeError:
                try:
                    for chunk in method(prompt_text):
                        text = _extract_text_from_resp(chunk)
                        if text:
                            yield text
                    return
                except Exception:
                    pass
            except Exception:
                continue

    for caller_name in ("invoke", "generate", "__call__", "call"):
        caller = getattr(llm, caller_name, None)
        if callable(caller):
            try:
                for chunk in caller([HumanMessage(prompt_text)], stream=True):  
                    text = _extract_text_from_resp(chunk)
                    if text:
                        yield text
                return
            except Exception:
                try:
                    for chunk in caller(prompt_text, stream=True):
                        text = _extract_text_from_resp(chunk)
                        if text:
                            yield text
                    return
                except Exception:
                    pass

    try:
        full = _call_llm(llm, prompt_text)
        if full:
            yield full
    except Exception:
        return

# Retrieval helpers
def _simple_score_text_score(q_terms: List[str], txt: str) -> int:
    txt_l = txt.lower()
    return sum(1 for t in q_terms if t in txt_l)


def _report_retrieval(report_id: str, query: str, top_k: int = 4) -> List[Dict[str, Any]]:
    """
    Try vector DB retrieval first (if retriever is available).
    Fallback to the in-memory chunk finder (find_chunks_for_report).
    Returned shape: list of dicts with keys: id, text, url (if known).
    """
    out: List[Dict[str, Any]] = []

    if _vector_retriever is not None:
        try:
            docs = _vector_retriever.run(query=query, k=top_k) or []
            for d in docs[:top_k]:
                if hasattr(d, "page_content"):
                    meta = getattr(d, "metadata", {}) or {}
                    out.append({
                        "id": meta.get("_id") or meta.get("id"),
                        "text": d.page_content,
                        "url": meta.get("source") or meta.get("url"),
                    })
                else:
                    text = d.get("fields", {}).get("chunk_text") or d.get("chunk_text") or d.get("page_content", "")
                    meta = d.get("metadata", {}) or {}
                    out.append({
                        "id": d.get("_id") or d.get("id"),
                        "text": text,
                        "url": meta.get("source") or meta.get("url"),
                    })
            if out:
                return out
        except Exception:
            pass

    if not report_id:
        return []

    chunks = find_chunks_for_report(report_id) or []
    if not chunks:
        return []

    q_terms = [t.lower() for t in re.findall(r"\w+", query) if len(t) > 2]
    if not q_terms:
        return [{"id": c.get("id"), "text": c.get("text"), "url": c.get("url")} for c in chunks[:top_k]]

    scored = []
    for c in chunks:
        txt = (c.get("text") or "")
        score = _simple_score_text_score(q_terms, txt)
        if score > 0:
            scored.append((score, c))
    scored.sort(key=lambda x: -x[0])
    if scored:
        return [{"id": c.get("id"), "text": c.get("text"), "url": c.get("url")} for _, c in scored[:top_k]]
    return [{"id": c.get("id"), "text": c.get("text"), "url": c.get("url")} for c in chunks[:top_k]]


# Prompt / formatting helpers
def _prepare_retrieved_section(chunks: List[Dict[str, Any]], max_chars: int = 1500) -> str:
    """
    Join top chunks into a single string for prompt injection.
    This is minimal: simply concatenates chunk texts with source ids.
    """
    out = []
    total = 0
    for c in chunks:
        txt = c.get("text", "")
        if not txt:
            continue
        if total + len(txt) > max_chars:
            break
        out.append(f"[{c.get('id')}] {txt}")
        total += len(txt)
    return "\n\n".join(out)


def build_prompt(history: List[Dict[str, Any]], retrieved_text: Optional[str], user_text: str) -> str:
    """
    Build a single prompt string (compatible with many client call patterns).
    Composition:
      - system
      - optional retrieved evidence block
      - recent history (last 8 turns)
      - user_text
    """
    parts = [f"System: {SYSTEM_PROMPT}"]
    if retrieved_text:
        parts.append("Retrieved evidence:\n" + retrieved_text)
    for h in history[-8:]:
        role = "User" if h["role"] == "user" else "Assistant"
        parts.append(f"{role}: {h['text']}")
    parts.append("User: " + user_text)
    parts.append("Assistant:")
    return "\n\n".join(parts)


# Primary chat functions
def chat_reply(session_id: str, user_text: str, use_retrieval: bool = False, report_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Synchronous chat reply used by API endpoints.
    Appends user message, optionally retrieves evidence, asks LLM, stores assistant message,
    and returns text + rich sources.
    Returned 'sources' list contains dicts: {id, url, excerpt}
    """
    try:
        history = get_history(session_id)
    except KeyError:
        raise

    append_message(session_id, "user", user_text)
    history = get_history(session_id)

    retrieved_chunks: List[Dict[str, Any]] = []
    retrieved_section = None
    if use_retrieval and report_id:
        retrieved_chunks = _report_retrieval(report_id=report_id, query=user_text, top_k=4)
        retrieved_section = _prepare_retrieved_section(retrieved_chunks)

    prompt = build_prompt(history=history, retrieved_text=retrieved_section, user_text=user_text)
    llm = get_llm()
    text = _call_llm(llm, prompt)

    append_message(session_id, "assistant", text)

    sources = []
    for c in (retrieved_chunks or []):
        sources.append({
            "id": c.get("id"),
            "url": c.get("url"),
            "excerpt": (c.get("text") or "")[:400]
        })

    return {"text": text, "sources": sources, "report_id": report_id}


def chat_stream(session_id: str, user_text: str, use_retrieval: bool = False, report_id: Optional[str] = None) -> Generator[str, None, None]:
    """
    Streaming generator for chat responses. Yields JSON-encoded event strings (ready to be used as SSE 'data: {...}\\n\\n').
    Events:
      - {"type":"start", "msg": "..."}
      - {"type":"token", "text":"..."}  (partial text pieces)
      - {"type":"progress", "pct":N, "msg":"..."}  (optional)
      - {"type":"done","text":"...","sources":[...]}
      - {"type":"error","error":"..."}
    Best-effort uses LLM streaming if available, otherwise emits start/progress/final.
    """
    try:
        history = get_history(session_id)
    except KeyError:
        yield f"data: {json.dumps({'type': 'error', 'error': 'session not found'})}\n\n"
        return

    append_message(session_id, "user", user_text)
    history = get_history(session_id)

    retrieved_chunks: List[Dict[str, Any]] = []
    retrieved_section = None
    if use_retrieval and report_id:
        try:
            retrieved_chunks = _report_retrieval(report_id=report_id, query=user_text, top_k=4)
            retrieved_section = _prepare_retrieved_section(retrieved_chunks)
            yield f"data: {json.dumps({'type':'progress','pct':5,'msg':'retrieved evidence','count': len(retrieved_chunks)})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type':'progress','pct':5,'msg':'retrieval_failed','error': str(e)})}\n\n"

    prompt = build_prompt(history=history, retrieved_text=retrieved_section, user_text=user_text)
    llm = get_llm()

    # start streaming
    yield f"data: {json.dumps({'type':'start','msg':'streaming started'})}\n\n"

    try:
        got_any = False
        text_accum = []
        for chunk in _call_llm_stream(llm, prompt):
            got_any = True
            if not isinstance(chunk, str):
                chunk = str(chunk)
            piece = chunk
            text_accum.append(piece)
            yield f"data: {json.dumps({'type':'token','text': piece})}\n\n"

        full_text = "".join(text_accum).strip()
        if not full_text:
            try:
                full_text = _call_llm(llm, prompt)
            except Exception as e:
                yield f"data: {json.dumps({'type':'error','error': str(e)})}\n\n"
                return

        append_message(session_id, "assistant", full_text)

        sources = []
        for c in (retrieved_chunks or []):
            sources.append({
                "id": c.get("id"),
                "url": c.get("url"),
                "excerpt": (c.get("text") or "")[:400]
            })

        yield f"data: {json.dumps({'type':'done','text': full_text, 'sources': sources, 'report_id': report_id})}\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'type':'error','error': str(e)})}\n\n"
        return