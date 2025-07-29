"""State management for the RAG agent workflow.

This schema allows code to effectively pass
Agent States' info through the LangGraph nodes.
"""
from dataclasses import dataclass
from typing import List, Dict, Any, TypedDict

from langchain.schema import Document


@dataclass
class AgentState(TypedDict):
    """State for the RAG agent graph"""
    question: str
    final_answer: str
    search_queries: List[str]
    documents_path: str
    success: bool
    retrieved_docs: List[Document]
    additional_info: str
    sources: List[str]
    retries: int
    validation: str
    retrieval_depth: int
    confidence_score: float
    needs_refinement: bool

