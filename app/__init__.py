"""
RAG Agent Package
A modular RAG (Retrieval-Augmented Generation) agent using LangChain and LangGraph.
"""

from app.core.agent import RAGAgent
from app.graph.workflow import create_workflow
from app.store.vector_store import VectorStoreManager
from app.utils.logging_config import setup_logging
from app.utils.query_analyzer import QueryAnalyzer
from app.utils.summary_manager import SummaryManager
from app.utils.document_processor import DocumentProcessor

__version__ = "1.0.0"
__all__ = [
    "RAGAgent",
    "create_workflow",
    "VectorStoreManager",
    "QueryAnalyzer",
    "SummaryManager",
    "DocumentProcessor",
    "setup_logging"
]
