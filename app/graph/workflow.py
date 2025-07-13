"""LangGraph workflow definition."""

from typing import Dict, Any
import logging

from langgraph.graph import StateGraph, END
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langgraph.graph.state import CompiledStateGraph

from app.core.agent import RAGAgent
from app.core.state import AgentState


logger = logging.getLogger(__name__)


def create_workflow(agent) -> CompiledStateGraph:
    """Create the LangGraph workflow"""
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("analyze_query", lambda state: _analyze_query_node(state, agent))
    workflow.add_node("init_vectorstores", lambda state: _init_vectorstore_node(state, agent))
    workflow.add_node("generate", lambda state: _generate_node(state, agent))
    workflow.add_node("evaluate", lambda state: _evaluate_node(state, agent))
    workflow.add_node("refine", lambda state: _refine_node(state, agent))

    # Set entry point
    workflow.set_entry_point("analyze_query")

    # Add edges
    workflow.add_edge("analyze_query", "init_vectorstores")
    workflow.add_edge("init_vectorstores", "generate")
    workflow.add_edge("generate", "evaluate")
    workflow.add_conditional_edges(
        "evaluate",
        lambda state: _should_refine(state),
        {
            "refine": "refine",
            "end": END
        }
    )
    workflow.add_edge("refine", "retrieve")

    return workflow.compile()


async def _analyze_query_node(state: AgentState, agent: RAGAgent) -> Dict[str, Any]:
    """Analyze the query complexity and determine retrieval strategy"""
    question = state["question"]

    # Use query analyzer
    analysis = await agent.query_analyzer.analyze_query(question)

    return {
        "search_queries": analysis["search_queries"],
        "retrieval_depth": 0
    }


async def _retrieve_summary_node(state: AgentState, agent: RAGAgent) -> Dict[str, Any]:
    """Retrieve summary of documents from FAISS"""
    top_docs = []
    for query in state["search_queries"]:
        docs = agent.summary_vectorstore.retrieve(query)
        top_docs.extend(docs)

    # Extract context information
    context = [
        {
            "content": docx.page_content[:300] + "...",
            "metadata": docx.metadata,
            "relevance_score": getattr(docx, 'relevance_score', 0)
        }
        for docx in top_docs
    ]

    # Extract sources information
    sources = {
        docx.metadata.get("source")
        for docx in top_docs
        if docx.metadata.get("source") is not None
    }

    return {
        "retrieved_docs": top_docs,
        "context": context,
        "sources": sources
    }

async def _retrieve_chunks_node(state: AgentState, agent: RAGAgent) -> Dict[str, Any]:
    """Retrieve chunks of documents from FAISS"""
    top_docs = []
    for query in state["search_queries"]:
        docs = agent.chunks_vectorstore.retrieve(query, k=20)
        filtered_chunks = [
            chunk for chunk in docs
            if chunk.metadata.get("source") in state["sources"]
        ]
        top_docs.extend(filtered_chunks)

    # Extract context information
    context = [
        {
            "content": docx.page_content[:300] + "...",
            "metadata": docx.metadata,
            "relevance_score": getattr(docx, 'relevance_score', 0)
        }
        for docx in top_docs
    ]

    return {
        "retrieved_docs": top_docs,
        "context": context
    }

async def _init_vectorstore_node(state: AgentState, agent: RAGAgent) -> Dict[str, Any]:
    """Initialize vector store and add documents in it. Or just load."""
    documents_path = state["documents_path"]
    agent.init_vectorstores(documents_path)

    return {}

async def _generate_node(state: AgentState, agent) -> Dict[str, Any]:
    """Generate answer using LangChain with retrieved context"""
    question = state["question"]
    context_docs = state["retrieved_docs"]

    if not context_docs:
        return {
            "answer": "I couldn't find relevant information to answer your question.",
            "confidence_score": 0.0
        }

    # Format context
    context_text = "\n\n".join([
        f"Document {i + 1}:\n{doc.page_content}"
        for i, doc in enumerate(context_docs)
    ])

    # Create prompt template
    prompt = ChatPromptTemplate.from_template("""
    You are a helpful AI assistant. Use the following context to answer the question comprehensively.

    Important instructions:
    1. Base your answer primarily on the provided context
    2. If the context doesn't contain sufficient information, clearly state what's missing
    3. Provide specific details and examples when available
    4. Maintain accuracy and avoid speculation beyond the context

    Context:
    {context}

    Question: {question}

    Answer:
    """)

    # Generate answer
    chain = prompt | agent.llm | StrOutputParser()
    answer = await chain.ainvoke({
        "context": context_text,
        "question": question
    })

    # Simple confidence scoring based on context relevance
    confidence = min(len(context_docs) / agent.config.top_k_retrieval, 1.0)

    return {
        "answer": answer,
        "confidence_score": confidence
    }


async def _evaluate_node(state: AgentState, agent) -> Dict[str, Any]:
    """Evaluate if the answer needs refinement"""
    answer = state["answer"]
    confidence = state.get("confidence_score", 0.0)
    retrieval_depth = state.get("retrieval_depth", 0)

    # Determine if refinement is needed
    needs_refinement = (
            confidence < 0.7 and
            retrieval_depth < agent.config.max_retrieval_depth - 1 and
            any(phrase in answer.lower() for phrase in [
                "don't have enough information",
                "context doesn't contain",
                "insufficient information",
                "cannot answer"
            ])
    )

    return {
        "needs_refinement": needs_refinement,
        "retrieval_depth": retrieval_depth + 1
    }


async def _refine_node(state: AgentState, agent) -> Dict[str, Any]:
    """Refine the query for better retrieval"""
    question = state["question"]
    answer = state["answer"]

    # Generate refined query based on the gaps in current answer
    prompt = ChatPromptTemplate.from_template("""
    The original question was: "{question}"

    The current answer is: "{answer}"

    Based on the current answer, generate 1-2 refined search queries that would help 
    fill in the missing information. Focus on the specific gaps mentioned in the answer.

    Return only the refined queries, one per line.
    """)

    chain = prompt | agent.llm | StrOutputParser()
    refined_queries_text = await chain.ainvoke({
        "question": question,
        "answer": answer
    })

    refined_queries = [q.strip() for q in refined_queries_text.split('\n') if q.strip()]

    return {
        "search_queries": refined_queries[:2]  # Limit to 2 refined queries
    }


def _should_refine(state: AgentState) -> str:
    """Determine if refinement is needed"""
    needs_refinement = state.get("needs_refinement", False)
    return "refine" if needs_refinement else "end"
