"""State management for the RAG agent workflow.

This schema allows code to effectively pass
Agent States' info through the LangGraph nodes.
"""
from dataclasses import dataclass
from typing import List, Dict, Any, TypedDict, Annotated
import operator
from langchain.schema import Document


@dataclass
class AgentState(TypedDict):
    """State for the RAG agent graph"""
    messages: Annotated[List[Dict[str, Any]], operator.add]
    question: str
    context: List[Document]
    retrieved_docs: List[Document]
    answer: str
    retrieval_depth: int
    search_queries: List[str]
    confidence_score: float
    sources: List[Dict[str, Any]]
    needs_refinement: bool
    number_of_retries: int
    documents_path: str
