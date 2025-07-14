"""Vector Store Management utilities."""

import logging
from typing import List, Optional
import os
import shutil

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

from app.core.config import RAGConfig


logger = logging.getLogger(__name__)

class VectorStoreManager:
    def __init__(
        self,
        config: RAGConfig,
        persist_path: Optional[str] = None,
    ):
        self.persist_path = os.path.join(config.vector_store_path, persist_path)
        self.embedding = HuggingFaceEmbeddings(
            model_name=config.embedding_model.get_secret_value()
        )
        self.vectorstore: Optional[FAISS] = None

        self.vectorstore = None  # will initialize on create/add

    def exist(self):
        """Is vector store exist"""
        return os.path.exists(self.persist_path)

    def create(self, documents: List[Document]):
        """Create and save a new FAISS index."""
        self.vectorstore = FAISS.from_documents(documents, embedding=self.embedding)
        self.vectorstore.save_local(self.persist_path)
        logger.info(f"Index created and saved at '{self.persist_path}'.")

    def add(self, documents: List[Document]):
        """Add documents to an existing index."""
        if self.vectorstore is None:
            if os.path.exists(self.persist_path):
                self.load()
            else:
                self.create(documents)
                return

        self.vectorstore.add_documents(documents)
        self.vectorstore.save_local(self.persist_path)

        logger.info("Documents added and index updated.")

    def delete(self):
        """Delete the FAISS index directory."""
        if os.path.exists(self.persist_path):
            shutil.rmtree(self.persist_path)
            self.vectorstore = None
            logger.info(f"Index at '{self.persist_path}' deleted.")
        else:
            logger.info("No index to delete.")

    def load(self):
        """Load FAISS index from disk."""
        self.vectorstore = FAISS.load_local(
            self.persist_path,
            embeddings=self.embedding,
            allow_dangerous_deserialization=True
        )
        logger.info(f"Loaded index from '{self.persist_path}'.")

    def retrieve(self, query: str, k: int = 3) -> List[Document]:
        """Perform similarity search on the index."""
        if self.vectorstore is None:
            logger.error("No FAISS index loaded.")
            raise ValueError("No FAISS index loaded.")
        return self.vectorstore.similarity_search(query, k=k)
