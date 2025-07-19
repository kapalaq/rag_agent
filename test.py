"""Basic usage example."""

import asyncio

from agent import RAGAgent
from agent import setup_logging


async def main():
    """Example usage of the RAG agent"""

    setup_logging()
    
    # Initialize agent
    agent = RAGAgent()

    documents = "./documents"

    question = "What is attention"

    result = await agent.query(question, documents)

    print(f"\n{'=' * 80}")
    print(f"Question: {result['question']}")
    print(f"Answer: {result['answer']}")
    print(f"Confidence: {result['confidence_score']:.2f}")
    print(f"Search queries used: {result['search_queries']}")
    print(f"Documents retrieved: {result['num_retrieved_docs']}")
    print(f"Sources: {result['sources']}")
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(main())