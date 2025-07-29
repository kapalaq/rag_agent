"""Summary management utilities."""

import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, Any

from langchain_anthropic import ChatAnthropic
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from agent.core.config import RAGConfig

logger = logging.getLogger(__name__)

class SummaryManager:
    """Manages document summaries with caching"""

    def __init__(self, config: RAGConfig):
        self.config = config
        self.summaries_dir = Path(config.vector_store_path) / Path("summaries")
        self.summaries_dir.mkdir(parents=True, exist_ok=True)


        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size * 4,
            chunk_overlap=config.chunk_overlap
        )
        self.llm = ChatAnthropic(
            api_key=config.anthropic_api_key.get_secret_value(),
            model=config.llm_model.get_secret_value(),
            temperature=config.llm_temperature
        )

    def _get_document_hash(self, document: Document) -> str:
        """Generate hash for document content"""
        content = document.page_content[:200] + str(document.metadata)
        return hashlib.md5(content.encode()).hexdigest()

    def _get_summary_path(self, doc_hash: str) -> Path:
        """Get path for summary file"""
        return self.summaries_dir / Path(f"{doc_hash}.json")

    def _summary_exists(self, doc_hash: str) -> bool:
        """Check if summary already exists"""
        return self._get_summary_path(doc_hash).exists()

    def _load_summary(self, doc_hash: str) -> Dict[str, Any]:
        """Load existing summary from file"""
        summary_path = self._get_summary_path(doc_hash)
        with open(summary_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _save_summary(self, doc_hash: str, summary_data: Dict[str, Any]):
        """Save summary to file"""
        summary_path = self._get_summary_path(doc_hash)
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)

    async def generate_summary(self, document: Document) -> Document: # Can be more effective
        """Generate summary for a document"""
        doc_hash = self._get_document_hash(document)

        # Check if summary already exists
        if self._summary_exists(doc_hash):
            logger.info(f"Loading existing summary for document {doc_hash[:8]}...")
            summary_data = self._load_summary(doc_hash)
            return Document(
                page_content=summary_data['summary'],
                metadata=summary_data['metadata']
            )

        # Generate new summary
        logger.info(f"Generating new summary for document {doc_hash[:8]}...")

        # Divides long documents into several summaries
        documents = self.splitter.split_documents([document])
        summaries = []
        for docx in documents:
            prompt = f"""Please provide a comprehensive summary of the following document. 
            Focus on key concepts, main arguments, and important details that would be useful for retrieval.
            The document may contain multiple pages or sections - please synthesize the content into a cohesive summary.

            Document metadata: {docx.metadata}

            Document content:
            {docx.page_content}

            Summary:"""

            try:
                response = await self.llm.ainvoke(prompt)
                summary = response.content.strip()
                summaries.append(summary)

            except Exception as e:
                logger.error(f"Failed to generate summary: {e}")
                raise

        if len(summaries) == 0:
            logger.info(f"There is a problem in summary generation!")
            raise ValueError(f"There are no summaries available for document {doc_hash[:8]}")

        elif len(summaries) > 1: # Last overall summarization
            prompt = f"""Please provide a comprehensive summary of the following document. 
            Focus on key concepts, main arguments, and important details that would be useful for retrieval.
            The document may contain multiple pages or sections - please synthesize the content into a cohesive summary.

            Document metadata: {document.metadata}

            Document content:
            {'\n\n'.join(summaries)}

            Summary:"""

            try:
                response = await self.llm.ainvoke(prompt)
                summary = response.content.strip()

            except Exception as e:
                logger.error(f"Failed to generate summary: {e}")
                raise

        else:
            summary = summaries[0]

        # Save summary
        summary_data = {
            'doc_hash': doc_hash,
            'summary': summary,
            'metadata': document.metadata,
            'original_length': len(document.page_content),
            'summary_length': len(summary)
        }
        self._save_summary(doc_hash, summary_data)

        logger.info(
            f"Generated and saved summary for document {doc_hash[:8]} "
            f"(original: {len(document.page_content)} chars, summary: "
            f"{len(summary)} chars)")

        return Document(page_content=summary, metadata=document.metadata)
