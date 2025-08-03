"""Main file that opens endpoint for RAG agent querying"""

import json
import logging
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from agent.core.rag_agent import RAGAgent
from agent.graph.workflow import create_workflow
from agent.utils.logging_config import setup_logging

app = FastAPI()
setup_logging()

logger = logging.getLogger(__name__)

# Load once
agent = RAGAgent()

class QueryRequest(BaseModel):
    """Request query fields"""
    question: str
    documents_path: str

@app.post("/query")
async def query_agent(request: QueryRequest):
    """Main function that process queries"""
    question, documents_path = request.question, request.documents_path

    logger.info(f"Processing query: {question}")

    await agent.init_vectorstores(documents_path)

    initial_state = {
        "question": question,
        "documents_path": documents_path,
        "final_answer": "",
    }

    graph = create_workflow(agent)

    # Run the graph
    async def event_stream():
        result = None
        async for state in graph.astream(initial_state):
            for node_name, node_json in state.items():
                if "additional_info" in node_json:
                    yield f"data: {json.dumps({'type': 'update', 'content': node_json['additional_info']})}\n\n"
                if "final_answer" in node_json:
                    result = node_json
        if result:
            agent.store_messages(question, result["final_answer"])
            yield f"data: {json.dumps({'type': 'final', 'data': result})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
