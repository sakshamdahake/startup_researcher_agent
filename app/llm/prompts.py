from typing import List, Any
from pydantic import BaseModel, Field

from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    FewShotChatMessagePromptTemplate,
)
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

### System / role prompts ###

researcher_system = SystemMessagePromptTemplate.from_template(
    """You are ResearchBot, an expert AI research assistant specialized in concise,
evidence-backed market / technology research and report generation.

Behavioral rules:
- Prioritize factual accuracy and cite sources when available.
- When you use retrieved documents, summarize the exact evidence and include the source id or metadata.
- If the answer is not contained in the retrieved material, say "Insufficient data" rather than inventing facts.
- Produce outputs that are structured and easy to read by a product manager."""
)

summarizer_system = SystemMessagePromptTemplate.from_template(
    "You are a concise summarizer. Produce short, clear summaries with a 2-3 sentence topline and 3 bullet takeaways."
)

synthesizer_system = SystemMessagePromptTemplate.from_template(
    """You are a synthesis engine that merges evidence from multiple sources.
Your job is to combine retrieved docs, web results and API outputs into a single coherent
section with: (1) key insight, (2) supporting evidence list, (3) confidence score (low/med/high),
and (4) suggested follow-ups or gaps."""
)

### Retrieval / chunk summarization prompt ###

chunk_summary_user = HumanMessagePromptTemplate.from_template(
    """Below is an extracted document chunk (or multiple short chunks) with metadata.

Chunk:
{chunk_text}

Metadata (if available):
{metadata}

Task:
1) Write a short 1-2 sentence summary (topline).
2) Produce 3 short bullet points for the most important facts or claims in the chunk.
3) If the chunk contains an explicit date, number, or named entity, highlight it.

Output format:
Topline: <one sentence>
- <bullet1>
- <bullet2>
- <bullet3>

Only output the topline and bullets (no extra prose)."""
)

chunk_summary_prompt = ChatPromptTemplate.from_messages(
    [summarizer_system, chunk_summary_user]
)

### Multi-Source synthesis prompt ###

synthesis_user = HumanMessagePromptTemplate.from_template(
    """You are given:
- A user query: {query}
- Retrieved documents (each with id, text, and metadata): {retrieved_docs}
- Web/API findings (short list): {web_findings}

Task:
1) Provide a single short paragraph "Key insight" that answers the user query using the evidence.
2) Provide a numbered list "Evidence" with up to 5 items. For each item include:
   - source_id or source label
   - 1 short quote/claim from the document
   - 1 sentence explaining how it supports the insight.
3) Give a confidence label in one word: low / medium / high and a one-line justification.
4) Suggest up to 3 follow-up research steps or clarifying questions.

Important:
- Use only the information in the retrieved documents and web_findings.
- If evidence is missing, say "Insufficient evidence" and list what you'd need.

Format:
Key insight:
    <one paragraph>

Evidence:
    1. [source_id] "<quote or claim>" — <explanation>
    ...

Confidence: <low/medium/high> — <one-line justification>

Follow-ups:
    - <item1>
    - <item2>
"""
)

synthesis_prompt = ChatPromptTemplate.from_messages(
    [researcher_system, synthesis_user]
)

### Final report structured output ###

class FinalReport(BaseModel):
    """Structured final report we want from the LLM for downstream code to consume."""
    title: str = Field(description="A short, SEO-friendly title for the report")
    executive_summary: str = Field(description="1-3 sentence executive summary")
    key_insights: List[str] = Field(description="Top 5 key insights")
    evidence: List[dict] = Field(
        description="List of evidence objects containing source_id, excerpt, explanation"
    )
    confidence: str = Field(description="low/medium/high")
    recommendations: List[str] = Field(description="Tactical recommendations or next steps")
    sources: List[dict] = Field(description="List of sources (id, url, type, metadata)")

final_answer_system = SystemMessagePromptTemplate.from_template(
    "You are ResearchBot. Output must conform exactly to the FinalReport JSON schema: title, executive_summary, key_insights, evidence, confidence, recommendations, sources."
)

final_report_user = HumanMessagePromptTemplate.from_template(
    """Produce a FinalReport (JSON) for the query: {query}

Inputs:
- Syntheses: {syntheses}
- Retrieved source metadata: {sources_metadata}

Rules:
- Fill all fields in the FinalReport model.
- Evidence entries must reference source ids that exist in sources_metadata.
- Keep text concise; executive_summary <= 3 sentences.
- Return only valid JSON (no explanatory text).

Return: a JSON object matching FinalReport.
"""
)

final_report_prompt = ChatPromptTemplate.from_messages(
    [final_answer_system, final_report_user]
)

### Few-shots example helper and prompt ###

# few-shot example: use ChatPromptTemplate for the example pair
few_shot_example_prompt = ChatPromptTemplate.from_messages(
    [
        HumanMessagePromptTemplate.from_template("{input}"),
        # AI response example — we can place it using a HumanMessagePromptTemplate too
        HumanMessagePromptTemplate.from_template("{output}")
    ]
)

EXAMPLES = [
    {
        "input": "Summarize the following chunk about product-market fit: 'Users reported 60% retention after 30 days...' ",
        "output": (
            "Topline: Users have strong retention signals for month 1.\n\n"
            "- Retention: 60% day-30 retention reported in the study.\n"
            "- Cohort detail: retention stronger in users acquired via organic channels.\n"
            "- Implication: product meets core need for a specific cohort."
        )
    },
    {
        "input": "Summarize the chunk: 'DeepSeek-V3 was released in December 2024 and targets...' ",
        "output": (
            "Topline: DeepSeek-V3 is a late-2024 release focused on search-specialized LLM workloads.\n\n"
            "- Launch: Released December 2024.\n"
            "- Positioning: Focused on retrieval-augmented tasks.\n"
            "- Implication: Useful as a base for vector-backed semantic search tasks."
        )
    }
]

few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=few_shot_example_prompt,
    examples=EXAMPLES
)

### Helpers ###
# NOTE: .format(...) returns the full prompt filled with variables (string form).
# You can pass the returned string to your LLM or wrap it in a HumanMessage if needed.

def get_chunk_summary_prompt(chunk_text: str, metadata: str = "") -> str:
    return chunk_summary_prompt.format(chunk_text=chunk_text, metadata=metadata)

def get_synthesis_prompt(query: str, retrieved_docs: Any, web_findings: Any) -> str:
    # repr(...) keeps the structure readable for the LLM; you can change to JSON dumps if preferred
    return synthesis_prompt.format(
        query=query,
        retrieved_docs=repr(retrieved_docs),
        web_findings=repr(web_findings),
    )

def get_final_report_prompt(query: str, syntheses: Any, sources_metadata: Any) -> str:
    return final_report_prompt.format(
        query=query,
        syntheses=repr(syntheses),
        sources_metadata=repr(sources_metadata),
    )