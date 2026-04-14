from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from backend.config import get_settings
from backend.services.database import get_vectorstore, collection_is_empty
from backend.services.llm import get_llm

_PROMPT_TEMPLATE = """{system_prompt}

Context:
---------
{{context}}
---------

Question: {{question}}

Answer (in Arabic):"""


def _build_prompt() -> PromptTemplate:
    settings = get_settings()
    template = _PROMPT_TEMPLATE.format(system_prompt=settings.system_prompt)
    return PromptTemplate(template=template, input_variables=["context", "question"])


def query_rag(question: str) -> dict:
    """Run a RAG query: retrieve relevant chunks and generate an answer."""
    vectorstore = get_vectorstore()

    if collection_is_empty(vectorstore):
        return {
            "answer": "عذراً، لم يتم تحميل أي بيانات بعد. يرجى التواصل مع إدارة النظام.",
            "sources": [],
        }

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    chain = RetrievalQA.from_chain_type(
        llm=get_llm(),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": _build_prompt()},
    )

    result = chain.invoke({"query": question})

    sources = [doc.page_content[:200] for doc in result.get("source_documents", [])]

    return {"answer": result["result"], "sources": sources}
