import logging

from fastapi import APIRouter, HTTPException

from backend.api.schemas import QueryRequest, QueryResponse
from backend.services.rag_service import query_rag

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/query", response_model=QueryResponse)
async def query_endpoint(body: QueryRequest):
    try:
        result = query_rag(body.question)
    except Exception as e:
        logger.exception("RAG query failed")
        raise HTTPException(status_code=502, detail="حدث خطأ أثناء معالجة طلبك. يرجى المحاولة لاحقاً.")
    return QueryResponse(**result)
