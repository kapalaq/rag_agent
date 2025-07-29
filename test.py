"""Basic usage example."""

if __name__ == "__main__":
    import requests

    url = "http://localhost:8000/query"
    payload = {
        "question": "What is an attention",
        "documents_path": "agent/documents",
        "history": []
    }

    response = requests.post(url, json=payload)

    print(response.json())
