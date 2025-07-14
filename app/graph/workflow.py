"""LangGraph workflow definition."""

from typing import Dict, Any
import logging

from langgraph.graph import StateGraph, END
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langgraph.graph.state import CompiledStateGraph

from app.core.state import AgentState


logger = logging.getLogger(__name__)

def create_workflow(agent) -> CompiledStateGraph:
    """Create the LangGraph workflow"""
    workflow = StateGraph(AgentState)

    # Async wrappers for async nodes
    async def analyze_query_node(state): return await _analyze_query_node(state, agent)
    async def init_vectorstores_node(state): return await _init_vectorstore_node(state, agent)
    async def summary_retrieve_node(state): return await _retrieve_summary_node(state, agent)
    async def chunk_retrieve_node(state): return await _retrieve_chunks_node(state, agent)
    async def web_retrieve_node(state): return await _retrieve_web_node(state, agent)
    async def generate_node(state): return await _generate_node(state, agent)
    async def validate_node(state): return await _validate_node(state, agent)
    async def evaluate_node(state): return await _evaluate_node(state, agent)
    async def refine_node(state): return await _refine_node(state, agent)

    # Add nodes
    workflow.add_node("analyze_query", analyze_query_node)
    workflow.add_node("init_vectorstores", init_vectorstores_node)
    workflow.add_node("summary_retrieve", summary_retrieve_node)
    workflow.add_node("chunk_retrieve", chunk_retrieve_node)
    workflow.add_node("web_retrieve", web_retrieve_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("validate", validate_node)
    workflow.add_node("evaluate", evaluate_node)
    workflow.add_node("refine", refine_node)

    # Set entry point
    workflow.set_entry_point("analyze_query")

    # Add edges
    workflow.add_edge("analyze_query", "init_vectorstores")
    workflow.add_conditional_edges(
        "init_vectorstores",
        lambda state: state["success"],
        {
            True: "summary_retrieve",
            False: END,
        }
    )
    workflow.add_conditional_edges(
        "summary_retrieve",
        lambda state: _check_retrieve(state, agent),
        {
            "Done": "chunk_retrieve",
            "Repeat": "summary_retrieve",
            "Failed": "web_retrieve",
        }
    )
    workflow.add_conditional_edges(
        "chunk_retrieve",
        lambda state: _check_retrieve(state, agent),
        {
            "Done": "generate",
            "Repeat": "chunk_retrieve",
            "Failed": "web_retrieve"
        }
    )
    workflow.add_conditional_edges(
        "web_retrieve",
        lambda state: _check_retrieve(state, agent),
        {
            "Done": "generate",
            "Repeat": "web_retrieve",
            "Failed": END
        }
    )
    workflow.add_edge("generate", "validate")
    workflow.add_conditional_edges(
        "validate",
        lambda state: state["success"],
        {
            True: "evaluate",
            False: "generate"
        }
    )
    workflow.add_conditional_edges(
        "evaluate",
        lambda state: _should_refine(state),
        {
            "end": END,
            "refine": "refine"
        }
    )
    workflow.add_conditional_edges(
        "refine",
        lambda state: _should_retry(state, agent),
        {
            "retry": "summary_retrieve",
            "end": END
        }
    )

    return workflow.compile()


async def _analyze_query_node(state: AgentState, agent) -> Dict[str, Any]:
    """Analyze the query complexity and determine retrieval strategy"""
    question = state["question"]

    # Use query analyzer
    analysis = await agent.query_analyzer.analyze_query(question)

    logger.info("Query has been analyzed.")

    return {
        "search_queries": analysis["search_queries"]
    }


async def _init_vectorstore_node(state: AgentState, agent) -> Dict[str, Any]:
    """Initialize vector store and add documents in it. Or just load."""
    try:
        documents_path = state["documents_path"]
        await agent.init_vectorstores(documents_path)

    except KeyError:
        return {
            "success": False,
            "final_answer": "Sorry! But you or our engineers forget to pass docs path :("
        }

    except Exception as e:
        logger.error("There is a problem in vector store initialization:", e)
        return {
            "success": False,
            "final_answer": "Sorry! There is a problem in vector store initialization. Please try again later."
        }

    logger.info("Query has been analyzed.")

    return {
        "success": True
    }


async def _retrieve_summary_node(state: AgentState, agent) -> Dict[str, Any]:
    """Retrieve summary of documents from FAISS"""
    top_docs = agent.retrieve_summary(state["search_queries"])

    retries = state.get("retries", 0)
    max_retries = agent.get_max_retries()
    retries = retries if retries < max_retries else 0

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

    logger.info("Summaries has been retrieved.")

    return {
        "retrieved_docs": top_docs,
        "additional_info": context,
        "sources": sources,
        "retries": retries + 1
    }


async def _retrieve_chunks_node(state: AgentState, agent) -> Dict[str, Any]:
    """Retrieve chunks of documents from FAISS"""
    top_docs = agent.retrieve_chunks(state["search_queries"], state["sources"])

    retries = state.get("retries", 0)
    max_retries = agent.get_max_retries()
    retries = retries if retries < max_retries else 0

    # Extract context information
    context = [
        {
            "content": docx.page_content[:300] + "...",
            "metadata": docx.metadata,
            "relevance_score": getattr(docx, 'relevance_score', 0)
        }
        for docx in top_docs
    ]

    logger.info("Chunks has been retrieved.")

    return {
        "retrieved_docs": top_docs,
        "context": context,
        "retries": retries + 1
    }


async def _retrieve_web_node(state: AgentState, agent) -> Dict[str, Any]:
    top_docs = agent.retrieve_web(state["search_queries"])
    retrieved_docs = state.get("retrieved_docs", [])
    retrieved_docs.extend(top_docs)

    retries = state.get("retries", 0)
    max_retries = agent.get_max_retries()
    retries = retries if retries < max_retries else 0

    logger.info("Web pages has be scraped.")

    return {
        "retrieved_docs": retrieved_docs,
        "retries": retries + 1,
        "final_answer": "Sorry! There are no documents in my knowledge base to answer this question :("
    }


async def _generate_node(state: AgentState, agent) -> Dict[str, Any]:
    """Generate answer using LangChain with retrieved context"""
    question = state["question"]
    context_docs = state["retrieved_docs"]
    validation = state.get("validation", "")

    retries = state.get("retries", 0)
    max_retries = agent.get_max_retries()
    retries = retries if retries < max_retries else 0

    # Format context
    context_text = "\n\n".join([
        f"Document {i + 1}:\n{doc.page_content}"
        for i, doc in enumerate(context_docs)
    ])
    if len(validation) > 0:
        context_text += ("\n\n" + validation)

    # Create prompt template
    prompt = ChatPromptTemplate.from_template("""
    You are a helpful assistant. Use the following context to answer the question comprehensively.

    Important instructions:
    1. Base your answer primarily on the provided context
    2. If the context doesn't contain sufficient information, clearly state what's missing
    3. Provide specific details and examples when available
    4. Maintain accuracy and avoid speculation beyond the context
    5. Do not include any unnecessary sentences, answer the question only

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

    logger.info(f"Answer has been created.")

    return {
        "final_answer": answer,
        "confidence_score": confidence,
        "retries": retries + 1
    }


async def _validate_node(state: AgentState, agent) -> Dict[str, Any]:
    """Check answer on hallucinations."""
    question = state["question"]
    context_docs = state["retrieved_docs"]
    answer = state["final_answer"]

    # Format context
    context_text = "\n\n".join([
        f"Document {i + 1}:\n{doc.page_content}"
        for i, doc in enumerate(context_docs)
    ])

    # Create prompt template
    prompt = ChatPromptTemplate.from_template("""
    The original question was: "{question}"
    
    The context was: "{context}"
    
    The current answer is: "{answer}"

    Examine answer on hallucinations: make sure each statement has a direct source from the provided documents.
    
    Return every mistake or hallucination, one per line.
    As a first line write either 'n', if there is no hallucination, or 'y', if there are hallucinations.
    """)

    # Generate answer
    chain = prompt | agent.llm | StrOutputParser()
    answer = await chain.ainvoke({
        "question": question,
        "context": context_text,
        "answer": answer
    })

    logger.info(f"Validation has been done.")

    return {
        "validation": answer,
        "success": answer[0] == 'n' or state["retries"] == 2
    }


async def _evaluate_node(state: AgentState, agent) -> Dict[str, Any]:
    """Evaluate if the answer needs refinement"""
    answer = state["final_answer"]
    confidence = state.get("confidence_score", 0.0)
    retrieval_depth = state.get("retrieval_depth", 0)

    # Determine if refinement is needed
    # Should add fuzzy match instead or any other algorithm
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

    logger.info("Evaluation has been done.")

    return {
        "needs_refinement": needs_refinement,
        "retrieval_depth": retrieval_depth + 1
    }


async def _refine_node(state: AgentState, agent) -> Dict[str, Any]:
    """Refine the query for better retrieval"""
    question = state["question"]
    answer = state["final_answer"]

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

    logger.info("Refinement has been done.")

    return {
        "search_queries": refined_queries[:2]  # Limit to 2 refined queries
    }


def _check_retrieve(state: AgentState, agent) -> str:
    """Check whether retrieve was successful"""
    docs = state.get("retrieved_docs", [])
    max_retries = agent.get_max_retries()
    retries = state.get("retries", max_retries)

    if len(docs) > 0:
        return "Done"
    elif retries < agent.max_retries and len(docs) == 0:
        return "Repeat"
    else:
        return "Failed"


def _should_refine(state: AgentState) -> str:
    """Determine if refinement is needed"""
    needs_refinement = state.get("needs_refinement", False)
    return "refine" if needs_refinement else "end"


def _should_retry(state: AgentState, agent) -> str:
    """Determine if another loop is needed"""
    search_queries = state["search_queries"]
    retrieval_depth = state["retrieval_depth"]

    return "retry" if len(search_queries) > 0 and retrieval_depth < agent.get_max_retrieval_depth() else "end"
