"""Main RAG Agent implementation."""

from pathlib import Path
from typing import List, Dict, Any, Tuple
import logging

from langchain.embeddings import HuggingFaceEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain.memory import ConversationBufferMemory
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.documents import Document
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from agent.core.config import RAGConfig
from agent.utils.summary_manager import SummaryManager
from agent.utils.document_processor import DocumentProcessor
from agent.store.vector_store import VectorStoreManager
from agent.utils.query_analyzer import QueryAnalyzer


logger = logging.getLogger(__name__)

class RAGAgent:
    """Main RAG Agent using LangChain and LangGraph"""

    def __init__(self):
        self.config = RAGConfig()

        self.llm = ChatAnthropic(
            anthropic_api_key=self.config.anthropic_api_key.get_secret_value(),
            model_name=self.config.llm_model.get_secret_value(),
            temperature=0.1
        )

        self.web = TavilySearchResults(
            tavily_api_key=self.config.tavily_api_key.get_secret_value(),
            k=self.config.top_k_retrieval
        )

        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.config.embedding_model.get_secret_value(),
            model_kwargs = {'device': self.config.embedding_device}
        )

        self.summary_vectorstore = VectorStoreManager(self.config, "faiss_summary")
        self.chunks_vectorstore = VectorStoreManager(self.config, "faiss_chunks")

        self.document_processor = DocumentProcessor(self.config)
        self.summary_manager = SummaryManager(self.config)
        self.query_analyzer = QueryAnalyzer(self.llm)

        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

    async def should_retrieve(self, query: str) -> Tuple[bool, Any]:
        """LLM will decide should it retrieve any docs from vectorstore"""
        prompt = PromptTemplate.from_template(
            "You are a RAG (Retrieval Augmented Generation) agent. "
            "Your work is to answer on user queries in accordance "
            "with provided documents. However, it is possible that "
            "user will send some vague and non-specific queries, "
            "such as 'hello', 'what can you do?'. This doesn't mean "
            "that you don't **have** enough context to answer. It means "
            "that you **don't need new** context to answer query. "
            "Thus, it is crucial separate such question from "
            "requiring retrieval ones for cost efficiency sake.\n"
            "You will be provided with query in tag (<query></query>), "
            "return one word on the following question: "
            "'Whether should I retrieve documents to answer this query?': Yes or No."
            "No dot or any character should be added except the word 'Yes' or word 'No'."
            "No 'but' or any other additional thoughts."
            "\n\n"
            "<query>{query}</query>"
            "\n\n"
            "Answer:"
        )
        chain = LLMChain(llm=self.llm, prompt=prompt)

        answer = chain.invoke({"query": query})

        if "yes" in answer['text'].lower():

            logger.info("Yes, we need to retrieve.")

            return True, ""

        logger.info("No, we don't need to retrieve.")

        prompt = PromptTemplate.from_template(
            "You are a RAG (Retrieval Augmented Generation) agent. "
            "Your work is to answer on user queries in accordance "
            "with provided documents. These documents is retrieved "
            "from local vectorstore and user can add documents "
            "in it using or interface's 'Upload' button near query "
            "entering box. Also, he can check what kind of documents "
            "LLM can access using button 'Documents' next to 'Upload'. "
            "Be polite and ask for help if it is needed. "
            "With this information provided, answer succinctly "
            "following query:\n\n"
            "<query>{query}</query>"
        )

        chain = LLMChain(llm=self.llm, prompt=prompt)

        answer = chain.invoke({"query": query})

        logger.info("Answer generated")

        return False, answer['text']


    @staticmethod
    def _list_documents(path: str, extensions=None) -> List[str]:
        """List all documents with the specified folder."""
        if extensions is None:
            extensions = [".md", ".txt", ".pdf", ".docx", ".doc"]

        folder = Path(path)
        return sorted([f for f in folder.glob("*") if f.suffix.lower() in extensions])

    async def init_vectorstores(self, path_to_docs: str) -> None:
        """Init and add documents to both Summary and Chunks Layers in FAISS."""
        if (self.summary_vectorstore.exist() and
            self.chunks_vectorstore.exist()):
            self.load_vectorstore()
            return

        logger.info(f"Adding {len(path_to_docs)} documents to vector stores")
        logger.info(path_to_docs)

        docs = []
        summaries = []
        for docx in RAGAgent._list_documents(path_to_docs):
            # Process documents
            langchain_doc = self.document_processor.load_document(docx)

            summary = await self.summary_manager.generate_summary(langchain_doc)

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

    def load_vectorstore(self) -> None:
        """Load vector store from disk"""
        self.summary_vectorstore.load()
        self.chunks_vectorstore.load()

        logger.info(f"Vector stores loaded")

    def retrieve_summary(self, queries: List[str]) -> List[Document]:
        """Retrieve summary of documents from FAISS"""
        top_docs = []
        for query in queries:
            try:
                docs = self.summary_vectorstore.retrieve(query, self.config.top_k_retrieval)
                top_docs.extend(docs)
            except Exception as e:
                logger.error("There is a problem with retrieving summary:", e)

        return top_docs

    def retrieve_chunks(self, queries: List[str], sources: List[str]) -> List[Document]:
        """Retrieve chunks of documents from FAISS"""
        top_docs = []
        for query in queries:
            docs = self.chunks_vectorstore.retrieve(query, self.config.top_k_retrieval * len(sources))
            filtered_chunks = [
                chunk for chunk in docs
                if chunk.metadata.get("source") in sources
            ]
            top_docs.extend(filtered_chunks)

        return top_docs

    def retrieve_web(self, queries: List[str]) -> List[Document]:
        """Retrieve documents from the web using Tavily."""
        top_docs = []
        for query in queries:
            try:
                docs = self.web.invoke({"query": query})
                web_results = "\n".join([d["content"] for d in docs])
                web_results = Document(page_content=web_results)
                top_docs.append(web_results)
            except Exception as e:
                logger.error("There is a problem with web searching documents:", e)

        return top_docs

    def get_max_retries(self) -> int:
        """Get the max retries."""
        return self.config.max_retries

    def get_max_retrieval_depth(self) -> int:
        """Get the max retrieval depth."""
        return self.config.max_retrieval_depth

    def store_messages(self, question: str, answer: str) -> None:
        """Store messages in memory of the agent for chat history."""
        # Store in memory
        self.memory.chat_memory.add_user_message(question)
        self.memory.chat_memory.add_ai_message(answer)


if __name__ == "__main__":
    import os
    agent = RAGAgent()
    print(os.getcwd())
