from typing import List, Dict, Optional, Any

from langchain.tools import tool

from app.services.tools import (
    WebSearchTool,
    FetchUrlTool,
    PDFLoaderTool,
    SplitAndEmbedTool,
    DBRetrieverTool,
    SummarizerTool,
    SynthesizerTool,
    ReportGeneratorTool,
)
from app.services.research_agent import ResearchAgent
from app.services.tracer import get_current_tracer

_research_agent_cls = ResearchAgent
_web_search_tool = WebSearchTool()
_fetch_url_tool = FetchUrlTool()
_pdf_loader_tool = PDFLoaderTool()
_split_and_embed_tool = SplitAndEmbedTool(namespace="default")
_db_retriever_tool = DBRetrieverTool()
_summarizer_tool = SummarizerTool()
_synthesizer_tool = SynthesizerTool()
_report_generator_tool = ReportGeneratorTool()


@tool
def run_pipeline(query: str, pdf_paths: Optional[List[str]] = None, namespace: str = "default") -> Dict[str, Any]:
    """
    Execute the full deterministic research pipeline and return FinalReport as a dict.
    Runs web search, fetch, PDF ingestion, chunking, embedding, retrieval, summarization,
    synthesis and final report generation. Emits tracer events if available.
    """
    tracer = get_current_tracer()
    tool_name = "run_pipeline"
    payload = {"query": query, "pdf_paths": pdf_paths or [], "namespace": namespace}

    if tracer:
        try:
            tracer.on_tool_start(tool_name, payload)
        except Exception:
            pass

    try:
        ra = _research_agent_cls(namespace=namespace)
        if tracer:
            try:
                tracer.on_tool_progress(tool_name, {"step": "starting", "msg": "initializing research agent"})
            except Exception:
                pass

        report = ra.run(query, pdf_paths=pdf_paths or [])

        if tracer:
            try:
                tracer.on_tool_progress(tool_name, {"step": "finished_pipeline", "msg": "pipeline completed"})
            except Exception:
                pass

        try:
            result_dict = report.model_dump()
        except Exception:
            if hasattr(report, "dict"):
                result_dict = report.dict()
            else:
                result_dict = getattr(report, "__dict__", report)

        if tracer:
            try:
                tracer.on_tool_end(tool_name, {"status": "ok", "title": (result_dict.get("title") if isinstance(result_dict, dict) else None)})
            except Exception:
                pass

        return result_dict

    except Exception as e:
        if tracer:
            try:
                tracer.on_tool_error(tool_name, e)
            except Exception:
                pass
        raise


@tool
def web_search(query: str, num: int = 10) -> List[Dict[str, str]]:
    """
    Search the web for relevant pages. Returns list of dicts with title, url, snippet.
    Emits tracer events for start and end (count).
    """
    tracer = get_current_tracer()
    tool_name = "web_search"
    if tracer:
        try:
            tracer.on_tool_start(tool_name, {"query": query, "num": num})
        except Exception:
            pass

    try:
        results = _web_search_tool.run(query=query, num=num)
        if tracer:
            try:
                tracer.on_tool_end(tool_name, {"status": "ok", "count": len(results)})
            except Exception:
                pass
        return results
    except Exception as e:
        if tracer:
            try:
                tracer.on_tool_error(tool_name, e)
            except Exception:
                pass
        raise


@tool
def fetch_url(url: str) -> Dict[str, Any]:
    """
    Fetch a webpage and return a small document dict: {page_content, metadata}.
    Emits tracer events for start/end and errors.
    """
    tracer = get_current_tracer()
    tool_name = "fetch_url"
    if tracer:
        try:
            tracer.on_tool_start(tool_name, {"url": url})
        except Exception:
            pass

    try:
        doc = _fetch_url_tool.run(url)
        result = {"page_content": getattr(doc, "page_content", ""), "metadata": getattr(doc, "metadata", {}) or {}}

        if tracer:
            try:
                tracer.on_tool_end(tool_name, {"status": "ok", "len": len(result["page_content"])})
            except Exception:
                pass

        return result
    except Exception as e:
        if tracer:
            try:
                tracer.on_tool_error(tool_name, e)
            except Exception:
                pass
        raise


@tool
def pdf_load(path_or_s3_key: str) -> List[Dict[str, Any]]:
    """
    Load a PDF and return a list of per-page document dicts.
    Each dict: {'page_content': str, 'metadata': {...}}. Emits tracer events.
    """
    tracer = get_current_tracer()
    tool_name = "pdf_load"
    if tracer:
        try:
            tracer.on_tool_start(tool_name, {"path": path_or_s3_key})
        except Exception:
            pass

    try:
        docs = _pdf_loader_tool.run(path_or_s3_key)
        out = [
            {"page_content": getattr(d, "page_content", ""), "metadata": getattr(d, "metadata", {}) or {}}
            for d in docs
        ]
        if tracer:
            try:
                tracer.on_tool_end(tool_name, {"status": "ok", "pages": len(out)})
            except Exception:
                pass
        return out
    except Exception as e:
        if tracer:
            try:
                tracer.on_tool_error(tool_name, e)
            except Exception:
                pass
        raise


@tool
def split_and_embed(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Split provided documents into chunks and upsert embeddings to vector store.
    Input: list of {'page_content', 'metadata'}. Returns list of inserted record metadata.
    Emits tracer events for counts.
    """
    tracer = get_current_tracer()
    tool_name = "split_and_embed"
    if tracer:
        try:
            tracer.on_tool_start(tool_name, {"documents": len(documents)})
        except Exception:
            pass

    try:
        from app.services.tools import Document  # avoid top-level import/type mismatch
        docs_objs = [Document(page_content=d["page_content"], metadata=d.get("metadata", {})) for d in documents]
        records = _split_and_embed_tool.run(docs_objs)

        if tracer:
            try:
                tracer.on_tool_end(tool_name, {"status": "ok", "inserted": len(records)})
            except Exception:
                pass

        return records
    except Exception as e:
        if tracer:
            try:
                tracer.on_tool_error(tool_name, e)
            except Exception:
                pass
        raise


@tool
def db_retrieve(query: str, top_k: int = 6) -> List[Dict[str, Any]]:
    """
    Retrieve top-k chunks from the vectorstore and return list of dicts:
    {id, page_content, metadata}. Emits tracer events.
    """
    tracer = get_current_tracer()
    tool_name = "db_retrieve"
    if tracer:
        try:
            tracer.on_tool_start(tool_name, {"query": query, "top_k": top_k})
        except Exception:
            pass

    try:
        docs = _db_retriever_tool.run(query=query, k=top_k)
        out: List[Dict[str, Any]] = []
        for d in docs:
            if hasattr(d, "page_content"):
                out.append({
                    "id": getattr(d, "metadata", {}).get("_id") or getattr(d, "metadata", {}).get("id"),
                    "page_content": d.page_content,
                    "metadata": getattr(d, "metadata", {}) or {}
                })
            else:
                fields = d.get("fields", {}) if isinstance(d, dict) else {}
                out.append({
                    "id": d.get("_id") or d.get("id"),
                    "page_content": fields.get("chunk_text") or d.get("chunk_text", ""),
                    "metadata": d.get("metadata", {}) or {}
                })

        if tracer:
            try:
                tracer.on_tool_end(tool_name, {"status": "ok", "retrieved": len(out)})
            except Exception:
                pass

        return out
    except Exception as e:
        if tracer:
            try:
                tracer.on_tool_error(tool_name, e)
            except Exception:
                pass
        raise


@tool
def summarize_chunk(chunk_text: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Summarize a text chunk. Returns {'summary': str, 'metadata': {...}}.
    Emits tracer events including summary length when available.
    """
    tracer = get_current_tracer()
    tool_name = "summarize_chunk"
    if tracer:
        try:
            tracer.on_tool_start(tool_name, {"len": len(chunk_text)})
        except Exception:
            pass

    try:
        summary = _summarizer_tool.run(chunk_text, metadata=metadata or {})
        out = {"summary": summary, "metadata": metadata or {}}

        if tracer:
            try:
                tracer.on_tool_end(tool_name, {"status": "ok", "summary_len": len(summary) if isinstance(summary, str) else None})
            except Exception:
                pass

        return out
    except Exception as e:
        if tracer:
            try:
                tracer.on_tool_error(tool_name, e)
            except Exception:
                pass
        raise


@tool
def synthesize(query: str, retrieved_docs: List[Dict[str, Any]], web_findings: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """
    Synthesize retrieved documents and web findings into a single string output.
    Returns {'synthesis': str}. Emits tracer events.
    """
    tracer = get_current_tracer()
    tool_name = "synthesize"
    if tracer:
        try:
            tracer.on_tool_start(tool_name, {"query": query, "docs": len(retrieved_docs)})
        except Exception:
            pass

    try:
        web_findings = web_findings or []
        synth = _synthesizer_tool.run(query=query, retrieved_docs=retrieved_docs, web_findings=web_findings)
        out = {"synthesis": synth}

        if tracer:
            try:
                tracer.on_tool_end(tool_name, {"status": "ok", "len": len(synth) if isinstance(synth, str) else None})
            except Exception:
                pass

        return out
    except Exception as e:
        if tracer:
            try:
                tracer.on_tool_error(tool_name, e)
            except Exception:
                pass
        raise


@tool
def generate_report(query: str, syntheses: List[Dict[str, Any]], sources_metadata: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Produce a FinalReport JSON object and return it as a plain dict.
    Emits tracer events with the final title when available.
    """
    tracer = get_current_tracer()
    tool_name = "generate_report"
    if tracer:
        try:
            tracer.on_tool_start(tool_name, {"query": query, "syntheses": len(syntheses), "sources": len(sources_metadata)})
        except Exception:
            pass

    try:
        report_obj = _report_generator_tool.run(query=query, syntheses=syntheses, sources_metadata=sources_metadata)
        try:
            data = report_obj.model_dump()
        except Exception:
            data = report_obj.dict() if hasattr(report_obj, "dict") else getattr(report_obj, "__dict__", report_obj)

        if tracer:
            try:
                tracer.on_tool_end(tool_name, {"status": "ok", "title": data.get("title") if isinstance(data, dict) else None})
            except Exception:
                pass

        return data
    except Exception as e:
        if tracer:
            try:
                tracer.on_tool_error(tool_name, e)
            except Exception:
                pass
        raise


TOOLS = [
    run_pipeline,
    web_search,
    fetch_url,
    pdf_load,
    split_and_embed,
    db_retrieve,
    summarize_chunk,
    synthesize,
    generate_report,
]


__all__ = [
    "run_pipeline",
    "web_search",
    "fetch_url",
    "pdf_load",
    "split_and_embed",
    "db_retrieve",
    "summarize_chunk",
    "synthesize",
    "generate_report",
    "TOOLS",
]