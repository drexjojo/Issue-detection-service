"""FastAPI application factory with lifespan-managed resources.

On startup:
- Opens ChromaDB vector store (auto-persisted, skips indexing if data exists)
- Connects to local Klaire MCP and Linear MCP servers
- Indexes issues from any source not yet in the vector store
- Initializes the LLM-powered analysis pipeline (local Ollama)
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.clients.linear_mcp_client import LinearMCPClient
from src.clients.local_mcp_client import LocalMCPClient
from src.config import get_settings
from src.core.classifier import Classifier
from src.core.duplicate_detector import DuplicateDetector
from src.core.extractor import Extractor
from src.core.generator import Generator
from src.core.llm_client import OllamaClient
from src.core.pipeline import AnalysisPipeline
from src.embeddings.store import EmbeddingStore

logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()

    # 1. Open ChromaDB vector store (auto-persisted)
    embedding_store = EmbeddingStore(
        model_name=settings.embedding_model,
        persist_dir="data/chroma_db",
    )

    # 2. Connect to local ticketing MCP server
    logger.info("Connecting to Klaire MCP server...")
    local_client = LocalMCPClient(settings.mcp_server_script)
    await local_client.connect()

    # 3. Connect to Linear MCP server
    logger.info("Connecting to Linear MCP server...")
    linear_client = LinearMCPClient(
        settings.linear_mcp_url, settings.linear_api_key
    )
    await linear_client.connect()

    # 4. Index local issues if not already in ChromaDB
    if not embedding_store.has_source("local"):
        logger.info("Indexing local issues...")
        local_issues = await local_client.list_all_issues()
        embedding_store.index_issues(local_issues, source="local")
    else:
        logger.info("Local issues already indexed in ChromaDB")

    # 5. Index Linear issues if not already in ChromaDB
    if not embedding_store.has_source("linear"):
        logger.info("Indexing Linear issues...")
        linear_issues = await linear_client.list_all_issues()
        embedding_store.index_issues(linear_issues, source="linear")
    else:
        logger.info("Linear issues already indexed in ChromaDB")

    # 6. Initialize LLM pipeline (local Ollama)
    logger.info("Initializing LLM pipeline...")
    ollama = OllamaClient(base_url=settings.ollama_base_url)
    pipeline = AnalysisPipeline(
        classifier=Classifier(ollama, settings.ollama_model),
        extractor=Extractor(ollama, settings.ollama_model),
        generator=Generator(ollama, settings.ollama_model),
        duplicate_detector=DuplicateDetector(
            embedding_store=embedding_store,
            reranker_model=settings.reranker_model,
            threshold=settings.similarity_threshold,
            max_results=settings.max_duplicates,
        ),
    )
    logger.info("LLM pipeline initialized! Service ready!")

    app.state.pipeline = pipeline
    app.state.local_client = local_client
    app.state.linear_client = linear_client
    app.state.ollama = ollama

    yield

    # Cleanup
    await ollama.close()
    await local_client.close()
    await linear_client.close()


app = FastAPI(
    title="Issue Detection Service",
    description="Intelligent issue detection with MCP integration and semantic duplicate detection with Klaire and Linear.",
    version="0.1.0",
    lifespan=lifespan,
)

# Import routes after app creation to avoid circular imports
from src.api.routes import router

app.include_router(router)
