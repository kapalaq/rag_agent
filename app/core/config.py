"""Configuration management for RAG agent using Pydantic.

Config is loaded and used by pydantic for
data leakage prevention and convenience.
"""

import os
from pydantic import Field, ValidationError, field_validator
from pydantic_core.core_schema import FieldValidationInfo
from pydantic_settings import BaseSettings


class RAGConfig(BaseSettings):
    """Configuration for the RAG agent"""

    anthropic_api_key: str = Field(..., description="Anthropic API key")
    tavily_api_key: str = Field(..., description="Tavily API key")
    llm_model: str = Field(default="claude-3-sonnet-20240229", description="Model name to use")
    llm_temperature: float = Field(default=0.1, ge=0.0, description="Temperature of the LLM model")
    embedding_model: str = Field(default="intfloat/multilingual-e5-large", description="Embedding model to use")
    embedding_device: str = Field(default="cpu", description="Device to use")
    chunk_size: int = Field(default=1000, gt=0, description="Size of text chunks")
    chunk_overlap: int = Field(default=200, ge=0, description="Overlap between chunks")
    max_retrieval_depth: int = Field(default=3, gt=0, description="Maximum retrieval depth")
    top_k_retrieval: int = Field(default=5, gt=0, description="Top K results to retrieve")
    compression_enabled: bool = Field(default=True, description="Enable compression")
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Similarity threshold")
    vector_store_path: str = Field(default="./vector_store", description="Path to vector store")
    verbose: bool = Field(default=True, description="Enable verbose logging")

    class Config:
        """Basic configuration for the RAGConfig class"""
        env_file = os.path.join(os.getcwd(), "settings", ".env")
        env_file_encoding = "utf-8"
        case_sensitive = False
        env_prefix = ""

    @field_validator('chunk_overlap')
    def validate_chunk_overlap(self, v, info: FieldValidationInfo):
        """Ensure chunk_overlap is not larger than chunk_size"""
        if 'chunk_size' in info.data and v >= info.data['chunk_size']:
            raise ValidationError('chunk_overlap must be less than chunk_size')
        return v

    @field_validator('anthropic_api_key')
    def validate_api_keys(self, v):
        """Ensure API keys are not empty"""
        if not v or not v.strip():
            raise ValidationError('API key cannot be empty')
        return v.strip()

    @field_validator('embedding_model')
    def validate_embeddings_model(self, v):
        """Ensure embeddings model is not empty"""
        if not v or not v.strip():
            raise ValidationError('Embeddings model cannot be empty')
        return v.strip()

    @field_validator('embedding_device')
    def validate_embedding_device(self, v):
        """Ensure embedding device is correct"""
        if not v in ['cpu', 'gpu']:
            raise ValidationError('Embeddings device should be either cpu or gpu')
        return v

    @field_validator('vector_store_path')
    def validate_vector_store_path(self, v):
        """Ensure vector store path is not empty"""
        if not v or not v.strip():
            raise ValidationError('Vector store path cannot be empty')
        return v.strip()


def get_config() -> RAGConfig:
    """Alternative factory function for backwards compatibility"""
    return RAGConfig()


if __name__ == "__main__":
    # Ensuring working directory
    print(os.getcwd())

    # Functionality test
    config = RAGConfig()
    print(config.model_dump_json(indent=2))

    # Or create with explicit values
    config_explicit = RAGConfig(
        anthropic_api_key="your-anthropic-key",
        openai_api_key="your-openai-key",
        chunk_size=800,
        verbose=False
    )
