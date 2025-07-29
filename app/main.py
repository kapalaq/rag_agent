"""A main server"""

from fastapi import FastAPI
from chainlit.utils import mount_chainlit

app = FastAPI()

@app.get("/status")
def status():
    """Get status of the server"""
    return {"message": "API is running"}

mount_chainlit(app=app, target="ui.py", path="/")
