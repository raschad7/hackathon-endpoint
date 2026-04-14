import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from backend.api.routes import router
from backend.config import get_settings
from backend.services.database import ingest_document

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    api_key = get_settings().openai_api_key
    logger.info("OpenAI API key loaded — last 4 chars: ...%s", api_key[-4:] if len(api_key) >= 4 else "TOO SHORT")
    logger.info("Starting up — checking for document ingestion...")
    try:
        ingest_document()
    except Exception:
        logger.exception("Document ingestion failed — server will start without data.")
    yield


app = FastAPI(title="RAG Backend", version="1.0.0", lifespan=lifespan)
app.include_router(router)
