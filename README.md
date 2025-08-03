<!--
from nltk.corpus.reader import documentsfrom agent_container.agent.core.rag_agent import RAGAgentfrom agent_container.agent.graph.workflow import create_workflow
-->


# 🧠 RAG Agent using LangGraph & LangChain

This project is a **Retrieval-Augmented Generation (RAG) Agent** built with:

- 🔁 **LangGraph**: for orchestrating recursive retrieval using conditional edges  
- 🦜 **LangChain**: to integrate vector stores, embeddings, memory, and agents  
- 🧭 **Tavily Search**: to augment answers with live web results  
- 🤖 **Anthropic Claude API**: as the core LLM (e.g., Claude 3.5)  
- 🔍 **FAISS**: for efficient local semantic retrieval  
- 💡 **HuggingFace Embeddings**: to embed documents and queries for semantic search

---

## ✨ Features

- **Agent-centric architecture**: all logic encapsulated in a single `Agent` class
Note: for modulation purposes, main 'run' function has be depreciated.
- Basic interface:  
    ```python
    from agent_container.agent import RAGAgent
    from agent_container.agent import create_workflow

    agent = RAGAgent()
    graph = create_workflow(agent)
    
    # Init vectorstore
    documents_path = "/docs/path"
    agent.init_vectorstores(documents_path)
    
    # All initial state params can be taken from 
    # agent_container/agent/core/state.py
    # Note: question and documents_path are required
    initial_state = {...}
  
    # Run graph
    ans = graph.invoke(initial_state)
    ```
- Recursive document retrieval via **LangGraph** conditional edges
- Seamless integration with **Tavily Search API** for real-time information
- Modular design for easy **extensibility and integration**

---

## 🗂️ Project Structure
```graphql
├── agent_container/
│   ├── log/
│   │   └──rag_agent.log    # Log file
│   ├── .dockerignore
│   ├── Dockerfile      # Dockerfile for RAG agent container
│   ├── requrements.txt
│   ├── vector_store    # Vector stores folder. Will be created on initialization
│   ├── main.py
│   └── agent/          # Main Agent module: query processing & orchestration
│       ├── documents/      # Source documents to embed
│       ├── utils/
│       │   ├── document_processor.py   # Read/chunk documents tool
│       │   ├── summary_manager.py      # Summary generating tool
│       │   ├── query_analyzer.py       # Query analysis and breakdown tool
│       │   └── logging_config.py       # Logging setup
│       ├── store/
│       │   └── vector_store.py     # FAISS-based vector store logic using LangChain
│       ├── graph/
│       │   └── workflow.py     # LangGraph setup with conditional edges
│       └── core/
│           ├── config.py       # Pydantic config facade
│           ├── state.py        # Graph variables/states
│           ├── agent.py        # Agent class
│           └── settings/
│                   ├── .env.template   # Helps to locate .env file
│                   └── .env            # API keys, settings, and constants
├── app/
│   ├── .dockerignore
│   ├── .env            # API keys, settings, and constants
│   ├── .env.template   # Helps to locate .env file
│   ├── chainlit.md     # Chanilit UI FAQ page
│   ├── Dockerfile      # Dockerfile for Chainlit UI container
│   ├── requirements.txt
│   ├── ui.py           # Chainlit UI main code
│   └── main.py         # FastAPI BackEnd main code
│
├── docker_compose.yml  # Docker-compose to up all project containers at once
├── requirements.txt    # Main requirements file to setup venv and run code locally
├── .gitignore
├── graphics/           # Image folder
└── README.md
```

---

## 🚀 Getting Started
### 1. Install Requirements
```bash
pip install -r requirements.txt
```
Make sure the following packages are included:
- ```langchain```
- ```langgraph```
- ```faiss-cpu or faiss-gpu```
- ```huggingface_hub```
- ```anthropic```

Will be in the future version, now we use LangChain module TavilySearch
- ```tavily-python```

### 2. Set Environment Variables
1. Paste your variables into .env that can be created in **the same folder and the same format** as _.env.template_

### 3. Set up Chainlit
1. Go to /app and open the console
2. Write ```chainlit create-secret```
3. Write ```CHAINLIT_AUTH_SECRET=...``` into your .env file

### 4. Make sure documents are set up
1. Go to /agent_container/agent
2. Create folder /documents
3. Add all documents you want to use

### 5. Start app
1. If you want to start in containers
   1. Go to _/RAGAgent_
   2. Open _/app/.env_ and write ```AGENT_URL=http://rag_agent:8000/query```
   2. Open console and write: ```docker compose up```
2. If you want to start it locally:
   1. Go to _/agent_container_ folder
   2. Open console and write: ```uvicorn main:app --host=localhost --ip=8000```
   3. Go to _/app_ folder
   4. Open _.env_ and write ```AGENT_URL=http://localhost:8000/query```
   5. Open console and write: ```uvicorn main:app --host=localhost --ip=8888```
3. Go to ```http://localhost:8888``` and have fun :)
---

## 🧠 Workflow

- Uses **LangChain FAISS** to perform local document retrieval
- Embeds documents and queries using **HuggingFace** embedding models
- Implements recursive retrieval with **LangGraph** conditional edge logic
- Falls back to **Tavily Search** when local results are insufficient
- Combines all context and queries **Anthropic Claude API** for final response and summaries
![img.png](graphics/img.png)

---

## 🛠️ Customization
All files are modules within my flow, so you there are no direct dependancies.
Follow functions names and return type, you can change every part.
For example:
- VectorStore can be created using llama_index
- Workflow can be redirected or new nodes/edges initialized
- Document preprocessing customized and different extensions added

---

## 📌 TODO
- Introduce feedback loop (Reinforcement Learning)
- Support multi-turn conversational memory
- Support documents filtering by masks
- Better evaluation technique (fuzzy match)
- Workflow nuances (retries, loops, web_search)
- Add automatization on docker
- Prompt Injection security

---

Main developer: @kapalaq