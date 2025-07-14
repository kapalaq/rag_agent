"""Basic usage example."""

import asyncio
from app import RAGAgent
from app import setup_logging


async def main():
    """Example usage of the RAG agent"""

    setup_logging()
    
    # Initialize agent
    agent = RAGAgent()

    documents = "./documents"

    question = "What is attention"

    result = await agent.query(question, documents)

    print("Result is:", result, flush=True)

    agent.store_messages(question, result["final_answer"])

if __name__ == "__main__":
    asyncio.run(main())