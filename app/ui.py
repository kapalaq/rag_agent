"""Chainlit UI endpoint that will directly send requests to the RAG Agent"""


import json
import os
import httpx
import chainlit as cl


@cl.password_auth_callback
async def on_authorize(username: str, password: str):
    if username == os.getenv("ADMIN_ID") and password == os.getenv("ADMIN_PASSWORD"):
        return cl.User(identifier='admin', metadata={'role': 'ADMIN'})


@cl.on_chat_start
async def on_chat_start():
    """Handle chat start"""
    app_user = cl.user_session.get("user")

    msg = cl.Message(f"Hello {app_user.identifier}")
    await msg.send()


@cl.on_message
async def on_msg(msg: cl.Message):
    """Handle messages by sending them into LLM container"""
    ans = cl.Message("Knocking on agent's door.")
    await ans.send()

    try:
        async with httpx.AsyncClient(timeout=47) as client:
            mode = "POST"
            url = os.getenv("AGENT_URL")
            content = {"question": msg.content, "documents_path": os.getenv("DOCUMENTS_PATH")}
            async with client.stream(mode, url, json=content) as response:
                async for line in response.aiter_lines():
                    if line.strip() == "":
                        continue
                    try:
                        event = json.loads(line.replace("data: ", ""))

                        if event["type"] == "update":
                            print(event["content"])
                            ans.content = event["content"]
                            await ans.update()
                        elif event["type"] == "final":
                            final_data = event["data"]
                            ans.content = final_data['final_answer']
                            await ans.update()
                    except Exception as e:
                        ans.content = f"Parse error:, {e}, Line was:, {line}"

    except httpx.ConnectError:
        ans.content = "Sorry! There is a problem connecting to our agent. Please try again later."
        await ans.update()
