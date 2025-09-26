import os
import time
import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from dotenv import load_dotenv
from pymongo import MongoClient
from bson import ObjectId

# LangChain / Google
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain.prompts import PromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.tools import Tool
from langchain_google_community import GoogleSearchAPIWrapper
from langchain_core.callbacks.base import BaseCallbackHandler
from google.api_core.exceptions import InternalServerError

# Local chat storage helpers
from chatbot import (
    ensure_indexes,
    create_chat,
    append_message,
    get_chat,
    list_chats,
)


# ---------------------------
# Pydantic models
# ---------------------------
class CreateChatResponse(BaseModel):
    chat_id: str


class ChatItem(BaseModel):
    id: str
    title: str
    updated_at: Optional[str] = None


class ListChatsResponse(BaseModel):
    items: List[ChatItem]


class ChatMessage(BaseModel):
    role: str
    content: str
    ts: Optional[str] = None


class ChatDetail(BaseModel):
    id: str
    title: str
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    messages: List[ChatMessage]


class SendMessageRequest(BaseModel):
    content: str


class SendMessageResponse(BaseModel):
    reply: str


class Settings(BaseModel):
    temperature: float


class SettingsUpdate(BaseModel):
    temperature: Optional[float] = None


# ---------------------------
# FastAPI lifespan: init and teardown
# ---------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- startup ---
    load_dotenv()
    google_api_key = os.getenv("GOOGLE_API_KEY")
    mongo_conn_string = os.getenv("MONGO_CONNECTION_STRING")
    google_cse_id = os.getenv("GOOGLE_CSE_ID")

    if not all([google_api_key, mongo_conn_string, google_cse_id]):
        raise RuntimeError(
            "Please set GOOGLE_API_KEY, MONGO_CONNECTION_STRING, and GOOGLE_CSE_ID in the .env file."
        )

    os.environ["GOOGLE_API_KEY"] = google_api_key
    os.environ["GOOGLE_CSE_ID"] = google_cse_id

    # Embeddings (static)
    embedding_function = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Mongo
    DB_NAME = "coffee_db"
    COLLECTION_NAME = "knowledge_collection"
    INDEX_NAME = "coffee_index"

    client = MongoClient(mongo_conn_string)
    app.state.mongo_client = client
    collection = client[DB_NAME][COLLECTION_NAME]
    app.state.chats_col = client[DB_NAME]["chats"]
    ensure_indexes(app.state.chats_col)

    # Vector store / Retriever
    vector_store = MongoDBAtlasVectorSearch(
        collection=collection, embedding=embedding_function, index_name=INDEX_NAME
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    # Save primitives in app.state
    app.state.retriever = retriever
    app.state.embedding_function = embedding_function
    app.state.temperature = 0.4

    def build_agent_for_temperature(temp: float):
        llm_local = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=float(max(0.0, min(1.0, temp))),
            convert_system_message_to_human=True,
        )

        rag_prompt_local = PromptTemplate(
            template=(
                "Answer the user's question based ONLY on the following context:\n\n{context}\n\nQuestion: {input}"
            ),
            input_variables=["context", "input"],
        )
        rag_chain_base = create_stuff_documents_chain(llm_local, rag_prompt_local)
        rag_chain_local = create_retrieval_chain(app.state.retriever, rag_chain_base)

        def run_rag_tool(query_dict_or_str: dict | str) -> str:
            query = (
                query_dict_or_str[list(query_dict_or_str.keys())[0]]
                if isinstance(query_dict_or_str, dict)
                else query_dict_or_str
            )
            result = rag_chain_local.invoke({"input": query})
            return result.get("answer", "I couldn't find a specific answer in my knowledge base.")

        rag_tool_local = Tool(
            name="coffee_knowledge_base",
            func=run_rag_tool,
            description=(
                "Your primary tool. Use this to answer any questions about coffee brewing, origins, "
                "processing, history, roasting, and techniques. This is your own expert knowledge."
            ),
        )

        search_wrapper_local = GoogleSearchAPIWrapper()
        web_search_tool_local = Tool(
            name="google_search",
            func=search_wrapper_local.run,
            description=(
                "A web search tool. Use this to find current or specific information on COFFEE-RELATED topics that "
                "are not in the knowledge base, like latest coffee prices, a specific new cafe, or recent competition results."
            ),
        )

        tools_local = [rag_tool_local, web_search_tool_local]

        agent_prompt_template = """
        You are a world-class Coffee Master and sommelier named 'Barista Bot'.
        You are an expert on all things coffee and your goal is to provide accurate, helpful information about coffee with a friendly, educational, and passionate tone.

        You have access to two tools:
        1. `coffee_knowledge_base`: Your internal, expert knowledge base. Always try this tool FIRST for any coffee-related question.
        2. `Google Search`: A web search tool.
        
        Here are your strict rules:
        - Rule 1: For any question you receive, you MUST use the `coffee_knowledge_base` tool first.
        - Rule 2: If the `coffee_knowledge_base` tool provides a complete answer, use it directly.
        - Rule 3: If the `coffee_knowledge_base` tool does not provide a sufficient answer BUT the question is CLEARLY ABOUT COFFEE, you are then allowed to use `Google Search` to find the missing information.
        - Rule 4: If the user's question is NOT related to coffee, politely decline: "As a Coffee Master, my focus is solely on the fascinating world of coffee. I can't answer questions on that topic."

        Think step-by-step and provide a final, comprehensive answer to the user.

        Begin!

        Question: {input}
        Thought: {agent_scratchpad}
        """
        agent_prompt_local = PromptTemplate.from_template(agent_prompt_template)
        agent_local = create_tool_calling_agent(llm_local, tools_local, agent_prompt_local)
        app.state.agent_executor = AgentExecutor(
            agent=agent_local, tools=tools_local, verbose=False, handle_parsing_errors=True
        )
        app.state.llm_for_title = llm_local

    # Build initial agent
    build_agent_for_temperature(app.state.temperature)

    # helper to modify agent on temperature change
    app.state._rebuild_agent = build_agent_for_temperature

    # yield to run app
    try:
        yield
    finally:
        # --- shutdown ---
        client: MongoClient = app.state.mongo_client
        if client:
            client.close()


# App
app = FastAPI(title="Coffee Chat AI API", version="0.1.0", lifespan=lifespan)

# CORS (allow Angular dev server)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200", "http://127.0.0.1:4200", "*"],  # adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Globals stored in app.state (typing aid)
class AppState:
    mongo_client: Optional[MongoClient] = None
    chats_col = None
    agent_executor: Optional[AgentExecutor] = None
    retriever = None
    embedding_function = None
    temperature: float
    _rebuild_agent = None
    llm_for_title = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chats", response_model=CreateChatResponse)
def create_chat_endpoint():
    chat_id = str(create_chat(app.state.chats_col))
    return CreateChatResponse(chat_id=chat_id)


@app.get("/chats", response_model=ListChatsResponse)
def list_chats_endpoint(limit: int = 10):
    items = list_chats(app.state.chats_col, limit=limit)

    def to_item(doc) -> ChatItem:
        return ChatItem(
            id=str(doc["_id"]),
            title=doc.get("title", "(sin título)"),
            updated_at=doc.get("updated_at").isoformat() if doc.get("updated_at") else None,
        )

    return ListChatsResponse(items=[to_item(doc) for doc in items])


@app.get("/chats/{chat_id}", response_model=ChatDetail)
def get_chat_endpoint(chat_id: str):
    doc = get_chat(app.state.chats_col, chat_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Chat not found")

    def to_msg(m) -> ChatMessage:
        return ChatMessage(
            role=m.get("role", "assistant"),
            content=m.get("content", ""),
            ts=m.get("ts").isoformat() if m.get("ts") else None,
        )

    return ChatDetail(
        id=str(doc["_id"]),
        title=doc.get("title", "(sin título)"),
        created_at=doc.get("created_at").isoformat() if doc.get("created_at") else None,
        updated_at=doc.get("updated_at").isoformat() if doc.get("updated_at") else None,
        messages=[to_msg(m) for m in doc.get("messages", [])],
    )


@app.post("/chats/{chat_id}/messages", response_model=SendMessageResponse)
def send_message_endpoint(chat_id: str, body: SendMessageRequest):
    # Validate chat exists
    doc = get_chat(app.state.chats_col, chat_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Chat not found")

    user_query = (body.content or "").strip()
    if not user_query:
        raise HTTPException(status_code=400, detail="content is required")

    # Save user turn
    append_message(app.state.chats_col, chat_id, "user", user_query)

    # Perform agent call with retry
    max_retries = 4
    retry_delay = 10

    for attempt in range(max_retries):
        try:
            response = app.state.agent_executor.invoke({"input": user_query})
            bot_text = response.get("output", "")
            # Save assistant turn
            append_message(app.state.chats_col, chat_id, "assistant", bot_text)

            # Auto-generate title if missing and we have at least two messages (user+assistant)
            try:
                doc2 = get_chat(app.state.chats_col, chat_id)
                if doc2 and (not doc2.get("title") or doc2.get("title") == "Coffee Chat"):
                    msgs = doc2.get("messages", [])
                    if len(msgs) >= 2:
                        user_first = msgs[0].get("content", "")
                        assistant_first = msgs[1].get("content", "")
                        prompt = (
                            "Given the following first user message and assistant reply from a coffee chat, "
                            "propose a short, catchy English title (max 6 words) without quotes.\n\n"
                            f"User: {user_first}\nAssistant: {assistant_first}"
                        )
                        try:
                            title_resp = app.state.llm_for_title.invoke(prompt)
                            new_title = title_resp.content.strip() if hasattr(title_resp, "content") else str(title_resp).strip()
                            if new_title:
                                app.state.chats_col.update_one({"_id": ObjectId(chat_id)}, {"$set": {"title": new_title}})
                        except Exception:
                            pass
            except Exception:
                pass
            return SendMessageResponse(reply=bot_text)
        except InternalServerError:
            if attempt == max_retries - 1:
                msg = "Estoy teniendo problemas para conectar con mi inteligencia. Inténtalo más tarde."
                append_message(app.state.chats_col, chat_id, "assistant", msg)
                return SendMessageResponse(reply=msg)
            time.sleep(retry_delay)
        except Exception as e:
            err_msg = f"Ocurrió un error inesperado: {e}. Por favor, intenta reformular."
            append_message(app.state.chats_col, chat_id, "assistant", err_msg)
            return SendMessageResponse(reply=err_msg)


# Run with: uvicorn api:app --reload


class SSECallbackHandler(BaseCallbackHandler):
    def __init__(self, queue: asyncio.Queue, loop: asyncio.AbstractEventLoop):
        self.queue = queue
        self.loop = loop
        self._last_tool: Optional[str] = None

    def _emit(self, event: str, data: str):
        # Called from worker threads; schedule thread-safe put into asyncio queue
        self.loop.call_soon_threadsafe(self.queue.put_nowait, (event, data))

    def on_chain_start(self, serialized, inputs, **kwargs):
        # High-level reasoning start (kept concise)
        self._emit("verbose", "Starting reasoning…")

    def on_chain_end(self, outputs, **kwargs):
        # Final status so the UI can swap out typing indicator
        self._emit("verbose", "Analyzing...")

    def on_tool_start(self, serialized, input_str, **kwargs):
        name = ((serialized or {}).get("name") or self._last_tool or "herramienta").lower()
        self._last_tool = name
        if name == "google_search":
            # Friendlier copy for web search
            self._emit("verbose", "Searching the web…")
        elif name == "coffee_knowledge_base":
            self._emit("verbose", "Consulting knowledge base…")
        else:
            self._emit("verbose", f"Using {name}…")

    def on_tool_end(self, output, **kwargs):
        name = (self._last_tool or "herramienta").lower()
        if name == "google_search":
            self._emit("verbose", "Search done. Analyzing information…")
        elif name == "coffee_knowledge_base":
            self._emit("verbose", "Knowledge retrieved. Drafting answer…")
        else:
            self._emit("verbose", f"{name} done.")

    # If the underlying LLM streams tokens through callbacks, forward them
    def on_llm_new_token(self, token: str, **kwargs):
        self._emit("delta", token)


@app.get("/chats/{chat_id}/stream")
async def stream_message(chat_id: str, content: str):
    # Validate chat exists
    doc = get_chat(app.state.chats_col, chat_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Chat not found")

    user_query = (content or "").strip()
    if not user_query:
        raise HTTPException(status_code=400, detail="content is required")

    # Save user turn
    append_message(app.state.chats_col, chat_id, "user", user_query)

    queue: asyncio.Queue[tuple[str, str]] = asyncio.Queue()
    loop = asyncio.get_running_loop()
    handler = SSECallbackHandler(queue, loop)
    final_text_parts: list[str] = []

    async def event_generator():
        # Small helper to format SSE
        def sse(event: str, data: str) -> str:
            # Ensure single-line data; if multi-line, split
            if "\n" in data:
                return "".join([f"event: {event}\ndata: {line}\n\n" for line in data.splitlines()])
            return f"event: {event}\ndata: {data}\n\n"

        # Prime an initial verbose (no await, as we might not be in queue consumer yet)
        queue.put_nowait(("verbose", "Reading your message…"))

        try:
            def run_agent():
                # Invoke with callbacks so we get verbose/tool and token events
                try:
                    # Note: token streaming depends on LLM support
                    resp = app.state.agent_executor.invoke({"input": user_query}, config={"callbacks": [handler]})
                except Exception as e:
                    # Forward error as verbose and re-raise
                    try:
                        loop.call_soon_threadsafe(queue.put_nowait, ("verbose", f"Error: {e}"))
                    except Exception:
                        pass
                    raise
                return resp

            task = loop.run_in_executor(None, run_agent)

            while True:
                try:
                    event, data = await asyncio.wait_for(queue.get(), timeout=0.1)
                    if event == "delta":
                        final_text_parts.append(data)
                    yield sse(event, data)
                except asyncio.TimeoutError:
                    if task.done():
                        break

            resp = await task
            bot_text = resp.get("output", "")
            if not final_text_parts:
                # If we didn't receive token deltas, send the full text now
                yield sse("delta", bot_text)
            # Mark done and persist assistant turn
            full_text = "".join(final_text_parts) or bot_text
            append_message(app.state.chats_col, chat_id, "assistant", full_text)

            # Auto-generate title if missing and we have at least two messages
            try:
                doc2 = get_chat(app.state.chats_col, chat_id)
                if doc2 and (not doc2.get("title") or doc2.get("title") == "Coffee Chat"):
                    msgs = doc2.get("messages", [])
                    if len(msgs) >= 2:
                        user_first = msgs[0].get("content", "")
                        assistant_first = msgs[1].get("content", "")
                        prompt = (
                            "Given the following first user message and assistant reply from a coffee chat, "
                            "propose a short, catchy English title (max 6 words) without quotes.\n\n"
                            f"User: {user_first}\nAssistant: {assistant_first}"
                        )
                        try:
                            title_resp = app.state.llm_for_title.invoke(prompt)
                            new_title = title_resp.content.strip() if hasattr(title_resp, "content") else str(title_resp).strip()
                            if new_title:
                                app.state.chats_col.update_one({"_id": ObjectId(chat_id)}, {"$set": {"title": new_title}})
                        except Exception:
                            pass
            except Exception:
                pass

            yield sse("done", "ok")
        except Exception as e:
            yield sse("verbose", f"Error during generation: {e}")
            yield sse("done", "error")

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.get("/settings", response_model=Settings)
def get_settings():
    return Settings(temperature=float(getattr(app.state, "temperature", 0.4)))


@app.post("/settings", response_model=Settings)
def update_settings(body: SettingsUpdate):
    # Update temperature and rebuild agent if provided
    if body.temperature is not None:
        t = float(max(0.0, min(1.0, body.temperature)))
        app.state.temperature = t
        # rebuild agent with new temp
        rebuilder = getattr(app.state, "_rebuild_agent", None)
        if callable(rebuilder):
            rebuilder(t)
    return Settings(temperature=float(app.state.temperature))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
