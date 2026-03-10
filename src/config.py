from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Linear integration
    linear_api_key: str

    # Local ticketing service
    ticketing_base_url: str = "https://klaire-ticket-service.web.app"

    # MCP server
    mcp_server_script: str = "src/mcp_server/server.py"

    # LLM config (Ollama local)
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b"

    # Embedding & reranking config
    embedding_model: str = "all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    similarity_threshold: float = 0.5
    max_duplicates: int = 5

    # Linear MCP
    linear_mcp_url: str = "https://mcp.linear.app/mcp"

    model_config = {"env_file": ".env", "extra": "ignore"}


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
