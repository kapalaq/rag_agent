"""Document processing utilities."""

import logging
from pathlib import Path
from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader
)

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """handle document processing and chunking"""

    SUPPORTED_EXTENSIONS = {'.md', '.pdf', '.txt', '.doc', '.docx'}

    def __init__(self, config):
        self.config = config
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap
        )

    @staticmethod
    def _combine_documents(documents: List[Document]) -> Document:
        """Combine multiple parts of a document into one."""
        combined_text = "\n\n".join(docx.page_content for docx in documents)
        combined_metadata = documents[0].metadata.copy() if documents else {}
        return Document(page_content=combined_text, metadata=combined_metadata)

    @staticmethod
    def load_document(file_path: str) -> Document:
        """Load document based on file extension"""
        file_path = Path(file_path)

        if file_path.suffix.lower() not in DocumentProcessor.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")

        try:
            if file_path.suffix.lower() == '.pdf':
                loader = PyPDFLoader(str(file_path))
            elif file_path.suffix.lower() in ['.txt', '.md']:
                loader = TextLoader(str(file_path), encoding='utf-8')
            elif file_path.suffix.lower() in ['.doc', '.docx']:
                loader = UnstructuredWordDocumentLoader(str(file_path))
            else:
                raise ValueError(f"Unsupported file type: {file_path.suffix}")

            documents = loader.load()

            logger.info(f"Loaded {len(documents)} pages from {file_path}")
            return DocumentProcessor._combine_documents(documents)

        except Exception as e:
            logger.error(f"Failed to load document {file_path}: {e}")

    def generate_chunks(self, documents: List[Document]) -> List[Document]:
        """Chunk documents"""
        return self.text_splitter.split_documents(documents)
