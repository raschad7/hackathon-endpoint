from langchain.prompts import PromptTemplate

from backend.config import get_settings
from backend.services.database import collection_is_empty, get_vectorstore
from backend.services.llm import get_llm

_PROMPT_TEMPLATE = """{system_prompt}

You must answer using only the retrieved context below.
If the context includes a process, show it as ordered steps.
If the context includes requirements, fees, durations, or warnings, organize them under short Arabic headings.
Do not mention information that does not appear in the context.

Context:
---------
{context}
---------

Question: {question}

Answer (in Arabic Markdown):"""


def _build_prompt() -> PromptTemplate:
    settings = get_settings()
    return PromptTemplate(
        template=_PROMPT_TEMPLATE,
        input_variables=["system_prompt", "context", "question"],
        partial_variables={"system_prompt": settings.system_prompt},
    )


def _build_retriever():
    settings = get_settings()
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": settings.retrieval_k,
            "fetch_k": settings.retrieval_fetch_k,
            "lambda_mult": 0.35,
        },
    )
    return vectorstore, retriever


def _format_context(documents: list) -> str:
    formatted_sections = []
    for index, doc in enumerate(documents, start=1):
        title = doc.metadata.get("heading_path") or doc.metadata.get("section_title") or f"مقطع {index}"
        formatted_sections.append(f"[المصدر {index}] {title}\n{doc.page_content}")
    return "\n\n".join(formatted_sections)


def _build_source_snippet(text: str, limit: int) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 1].rstrip() + "…"


def _format_sources(documents: list) -> list[str]:
    settings = get_settings()
    sources = []

    for doc in documents:
        title = doc.metadata.get("heading_path") or doc.metadata.get("section_title") or "مصدر"
        snippet = _build_source_snippet(doc.page_content, settings.source_excerpt_chars)
        sources.append(f"{title}: {snippet}")

    return sources


def _normalize_answer(answer: str) -> str:
    return answer.replace("\\n", "\n").replace("/n", "\n").strip()


def query_rag(question: str) -> dict:
    """Run a RAG query: retrieve relevant chunks and generate an answer."""
    vectorstore, retriever = _build_retriever()

    if collection_is_empty(vectorstore):
        return {
            "answer": "عذراً، لم يتم تحميل أي بيانات بعد. يرجى التواصل مع إدارة النظام.",
            "sources": [],
        }

    retrieved_docs = retriever.invoke(question)

    if not retrieved_docs:
        return {
            "answer": "عذراً، لا تتوفر لدي معلومات كافية للإجابة على هذا السؤال. يرجى التواصل مع بلدية الخليل مباشرة.",
            "sources": [],
        }

    prompt = _build_prompt().format(
        context=_format_context(retrieved_docs),
        question=question,
    )
    response = get_llm().invoke(prompt)
    answer = response.content if hasattr(response, "content") else str(response)

    return {
        "answer": _normalize_answer(answer),
        "sources": _format_sources(retrieved_docs),
    }
