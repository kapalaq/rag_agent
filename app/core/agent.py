"""Main RAG Agent implementation."""
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

from langchain.embeddings import HuggingFaceEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain.memory import ConversationBufferMemory

from app.core.config import RAGConfig
from app.graph.workflow import create_workflow
from app.utils.summary_manager import SummaryManager
from app.utils.document_processor import DocumentProcessor
from app.store.vector_store import VectorStoreManager
from app.utils.query_analyzer import QueryAnalyzer


logger = logging.getLogger(__name__)


class RAGAgent:
    """Main RAG Agent using LangChain and LangGraph"""

    def __init__(self, config: RAGConfig):
        self.config = config

        self.llm = ChatAnthropic(
            anthropic_api_key=config.anthropic_api_key,
            model_name=config.llm_model,
            temperature=0.1
        )

        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.config.embedding_model,
            model_kwargs = {'device': self.config.embedding_device}
        )

        self.summary_vectorstore = VectorStoreManager(config, "faiss_summary")
        self.chunks_vectorstore = VectorStoreManager(config, "faiss_chunks")

        self.document_processor = DocumentProcessor(config)
        self.summary_manager = SummaryManager(config)
        self.query_analyzer = QueryAnalyzer(self.llm)

        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        self.graph = create_workflow(self)

    @staticmethod
    def _list_documents(path: str, extensions=None):
        if extensions is None:
            extensions = [".md", ".txt", ".pdf", ".docx", ".doc"]

        folder = Path(path)
        return sorted([f for f in folder.glob("*") if f.suffix.lower() in extensions])

    def init_vectorstores(self, path_to_docs: str):
        """Init and add documents to both Summary and Chunks Layers in FAISS."""
        if (self.summary_vectorstore.exist() and
            self.chunks_vectorstore.exist()):
            self.load_vectorstore()
            return

        logger.info(f"Adding {len(path_to_docs)} documents to vector stores")

        docs = []
        summaries = []
        for docx in RAGAgent._list_documents(path_to_docs):
            # Process documents
            langchain_doc = self.document_processor.load_document(docx)

            summary = self.summary_manager.generate_summary(langchain_doc)

            # Collect documents
            docs.append(langchain_doc)
            summaries.append(summary)

        chunks = self.document_processor.generate_chunks(docs)

        # Delete vectorstores if exists
        self.summary_vectorstore.delete()
        self.chunks_vectorstore.delete()

        # Recreate vectorstores
        self.summary_vectorstore.create(summaries)
        self.chunks_vectorstore.create(chunks)

        logger.info(f"Added {len(chunks)} chunks and {len(summaries)} of documents to FAISS")

    def load_vectorstore(self):
        """Load vector store from disk"""
        self.summary_vectorstore.load()
        self.chunks_vectorstore.load()

        logger.info(f"Vector stores loaded")

    async def query(self, question: str) -> Dict[str, Any]:
        """Query the RAG agent"""
        logger.info(f"Processing query: {question}")

        initial_state = {
            "messages": [],
            "question": question,
            "context": [],
            "retrieved_docs": [],
            "answer": "",
            "retrieval_depth": 0,
            "search_queries": [],
            "confidence_score": 0.0,
            "sources": [],
            "needs_refinement": False
        }

        # Run the graph
        result = await self.graph.ainvoke(initial_state)

        # Store in memory
        self.memory.chat_memory.add_user_message(question)
        self.memory.chat_memory.add_ai_message(result["answer"])

        return {
            "question": question,
            "answer": result["answer"],
            "sources": result["sources"],
            "search_queries": result["search_queries"],
            "confidence_score": result["confidence_score"],
            "num_retrieved_docs": len(result["context"])
        }
