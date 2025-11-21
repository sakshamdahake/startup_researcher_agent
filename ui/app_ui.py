# startup_researcher/ui/app_ui.py
"""
Streamlit Chat UI for Startup Researcher

Features:
- create session
- chat streaming (SSE) for tokens
- simple progress/status panel for tool events
- optional retrieval (report_id)
- run synchronous research and view report
"""
import streamlit as st
import requests
import time
import json
import threading
from typing import Optional, Dict, Any, Iterator, List

st.set_page_config(page_title="Startup Researcher", layout="wide")

# --- configuration (secrets fallback) ---
try:
    API_BASE = st.secrets.get("API_BASE", "http://localhost:8000")
except Exception:
    API_BASE = "http://localhost:8000"

API_BASE = API_BASE.rstrip("/")

# endpoints
CREATE_SESSION_URL = f"{API_BASE}/api/v1/session"
CHAT_STREAM_ENDPOINT = f"{API_BASE}/api/v1/chat_stream"       # GET /chat_stream/{session_id}?text=...&use_retrieval=...&report_id=...
CHAT_SYNC_ENDPOINT = f"{API_BASE}/api/v1/chat"                # POST /chat/{session_id}
RESEARCH_SYNC_ENDPOINT = f"{API_BASE}/api/v1/research_sync"   # POST
LIST_REPORTS_ENDPOINT = f"{API_BASE}/api/v1/session"          # GET /session/{session_id}/history isn't for listing reports; session_manager exposes list_reports_for_session in backend; keep a quick view via /api/v1/report/{report_id}

# helpers
def json_safe(s: str) -> Any:
    try:
        return json.loads(s)
    except Exception:
        return s

# SSE parsing (simple)
def sse_event_lines(resp: requests.Response) -> Iterator[str]:
    """
    Yield full 'data' payloads as strings for each SSE event.
    Basic parser: accumulate lines starting with "data:" and yield on blank line separator.
    """
    buf_lines: List[str] = []
    for raw in resp.iter_lines(decode_unicode=True):
        if raw is None:
            continue
        line = raw.strip()
        if not line:
            # event boundary
            if buf_lines:
                yield "\n".join(buf_lines)
                buf_lines = []
            continue
        if line.startswith("data:"):
            buf_lines.append(line[len("data:"):].lstrip())
        # ignore other fields for now (id:, event:, etc.)
    # final
    if buf_lines:
        yield "\n".join(buf_lines)

def open_chat_stream(session_id: str, text: str, use_retrieval: bool=False, report_id: Optional[str]=None, timeout: int = 0):
    """
    Open SSE stream to chat_stream endpoint and yield decoded JSON events.
    Returns generator of parsed event dicts.
    """
    # build url
    params = {"text": text, "use_retrieval": "true" if use_retrieval else "false"}
    if report_id:
        params["report_id"] = report_id
    url = f"{CHAT_STREAM_ENDPOINT}/{session_id}"
    # stream
    try:
        with requests.get(url, params=params, stream=True, timeout=timeout if timeout>0 else None) as resp:
            resp.raise_for_status()
            for chunk in sse_event_lines(resp):
                # chunk may contain multiple JSON objects or a plain text token (we expect JSON)
                # try to parse each newline-separated JSON-like item
                # sometimes the event payload is already a JSON string; try that
                try:
                    parsed = json.loads(chunk)
                    yield parsed
                except Exception:
                    # chunk might contain multiple concatenated JSONs or plain text tokens
                    # try line-by-line parse
                    for line in chunk.splitlines():
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            yield json.loads(line)
                        except Exception:
                            # fallback: emit token event with text
                            yield {"type": "token", "text": line}
    except requests.RequestException as e:
        yield {"type": "error", "error": str(e)}
    except Exception as e:
        yield {"type": "error", "error": str(e)}

# --- session & history state ---
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "history" not in st.session_state:
    st.session_state.history = []   # list of {"role":"user"/"assistant"/"system","text":..., "ts": ...}
if "status_msgs" not in st.session_state:
    st.session_state.status_msgs = []  # for tool progress lines (most recent first)

# --- layout ---
left, right = st.columns([3, 1])

with right:
    st.markdown("## Controls")
    if st.button("Create new session"):
        try:
            r = requests.post(CREATE_SESSION_URL, timeout=10)
            r.raise_for_status()
            sid = r.json().get("session_id")
            st.session_state.session_id = sid
            st.success(f"Created session: {sid}")
        except Exception as e:
            st.error(f"Create session failed: {e}")

    st.markdown("**Session**")
    st.write(st.session_state.session_id or "No session yet")

    st.markdown("---")
    st.markdown("### Research")
    q_text = st.text_area("Research query", value="", key="research_query", height=80)
    if st.button("Run research (blocking)"):
        if not st.session_state.session_id:
            st.error("Create a session first.")
        elif not q_text.strip():
            st.warning("Please enter a research query.")
        else:
            with st.spinner("Running research... (this will block until complete)"):
                try:
                    payload = {"query": q_text, "pdf_paths": [], "session_id": st.session_state.session_id}
                    rr = requests.post(RESEARCH_SYNC_ENDPOINT, json=payload, timeout=120)
                    rr.raise_for_status()
                    data = rr.json()
                    # display the returned report summary
                    rep = data.get("report") or {}
                    st.markdown("#### Report")
                    st.write("**Title:**", rep.get("title"))
                    st.write("**Executive summary:**")
                    st.write(rep.get("executive_summary", ""))
                    st.write("**Key insights:**")
                    for ki in rep.get("key_insights", []):
                        st.write("-", ki)
                    # store minimal history reference
                    st.session_state.status_msgs.insert(0, {"ts": time.time(), "msg": f"Research completed: {rep.get('title')}"})
                except Exception as e:
                    st.error(f"Research failed: {e}")

    st.markdown("---")
    st.markdown("### Status (recent)")
    # show last 8 status messages
    for m in st.session_state.status_msgs[:8]:
        tstr = time.strftime("%H:%M:%S", time.localtime(m.get("ts", time.time())))
        st.write(f"[{tstr}] {m.get('msg')}")
    st.markdown("---")
    st.markdown("### Help")
    st.write("Use the chat form on the left. For retrieval, enter a `report_id` (from research response).")

# left: chat area
with left:
    st.markdown("# ResearchAssistant")
    box = st.container()

    def render_history():
        box.empty()
        with box.container():
            # title + messages
            st.write("### Conversation")
            for h in st.session_state.history[-200:]:
                role = h.get("role", "user")
                text = h.get("text", "")
                if role == "user":
                    st.markdown(f"**You:** {text}")
                elif role == "assistant":
                    st.markdown(f"**Assistant:** {text}")
                else:
                    st.markdown(f"*{role}:* {text}")

    render_history()

    st.markdown("---")

    # Chat form (clearing and submit handled safely)
    with st.form(key="chat_form", clear_on_submit=False):
        chat_text = st.text_input("Message", value="", key="chat_input")
        cols = st.columns([1, 1, 3])
        with cols[0]:
            use_retrieval = st.checkbox("Use retrieval", value=False, key="use_retrieval_box")
        with cols[1]:
            report_id = st.text_input("Report ID (optional)", value="", key="chat_report_id")
        with cols[2]:
            send = st.form_submit_button("Send")

    # send handling
    if send:
        if not st.session_state.session_id:
            st.warning("Create a session first (Control panel → Create new session).")
        elif not chat_text.strip():
            st.warning("Enter a message.")
        else:
            # append user turn
            st.session_state.history.append({"role": "user", "text": chat_text, "ts": time.time()})
            render_history()

            # area for assistant streaming
            assistant_area = st.empty()
            progress_area = st.empty()

            assistant_buf: List[str] = []

            # start streaming in a separate thread to keep UI responsive if desired
            # but we'll do a blocking iteration (streamlit run is single-threaded for UI updates).
            # The open_chat_stream generator yields events as dicts.
            for ev in open_chat_stream(
                session_id=st.session_state.session_id,
                text=chat_text,
                use_retrieval=use_retrieval,
                report_id=report_id or None,
                timeout=0,
            ):
                ev_type = ev.get("type")
                # handle progress & tool events
                if ev_type in ("tool_progress", "progress"):
                    # try to extract pct and message fields
                    pct = ev.get("pct")
                    msg = ev.get("msg") or ev.get("step") or ev.get("message")
                    ts = time.time()
                    label = f"progress: {pct}%" if pct is not None else f"{msg or ev}"
                    st.session_state.status_msgs.insert(0, {"ts": ts, "msg": label})
                    # render small progress bar + last messages
                    try:
                        if pct is not None:
                            progress_area.progress(min(max(int(pct), 0), 100))
                        progress_area.write(label)
                    except Exception:
                        progress_area.write(label)
                    continue

                if ev_type == "token":
                    token_text = ev.get("text", "")
                    assistant_buf.append(token_text)
                    assistant_area.markdown("**Assistant (streaming):** " + "".join(assistant_buf))
                elif ev_type == "done":
                    final = ev.get("text") or "".join(assistant_buf)
                    # append assistant message to history
                    st.session_state.history.append({"role": "assistant", "text": final, "ts": time.time()})
                    assistant_area.markdown("**Assistant:** " + final)
                    st.session_state.status_msgs.insert(0, {"ts": time.time(), "msg": "response done"})
                    break
                elif ev_type == "error":
                    assistant_area.markdown(f"**Error streaming:** {ev.get('error')}")
                    st.session_state.status_msgs.insert(0, {"ts": time.time(), "msg": f"stream error: {ev.get('error')}"})
                    break
                else:
                    # fallback: maybe we got a plain string
                    if isinstance(ev, str):
                        assistant_buf.append(ev)
                        assistant_area.markdown("**Assistant (streaming):** " + "".join(assistant_buf))
                    else:
                        # unknown event; log to status
                        st.session_state.status_msgs.insert(0, {"ts": time.time(), "msg": f"evt: {ev_type}"})

            # final render of history and clear input (safe to assign after form submit)
            render_history()
            try:
                st.session_state["chat_input"] = ""
            except Exception:
                # ignore if something odd happens
                pass

# footer: quick instructions
st.markdown("---")
st.markdown("Stream powered by your backend SSE endpoints. If you see errors, check backend logs.")