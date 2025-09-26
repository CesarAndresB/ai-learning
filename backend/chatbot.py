import os
import time
from datetime import datetime, timezone
from dotenv import load_dotenv
from pymongo import MongoClient, ASCENDING
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
from google.api_core.exceptions import InternalServerError

# ---------------------------
# Utilidades de almacenamiento de chats
# ---------------------------

def ensure_indexes(chats_col):
    # Índice para ordenar recientes y búsquedas futuras
    chats_col.create_index([("updated_at", ASCENDING)], name="updated_at_idx")

def create_chat(chats_col, title="Conversación de café"):
    now = datetime.now(timezone.utc)
    doc = {
        "title": title,
        "created_at": now,
        "updated_at": now,
        "messages": []
    }
    res = chats_col.insert_one(doc)
    return res.inserted_id

def append_message(chats_col, chat_id, role, content):
    now = datetime.now(timezone.utc)
    update = {
        "$push": {"messages": {"role": role, "content": content, "ts": now}},
        "$set": {"updated_at": now}
    }
    chats_col.update_one({"_id": ObjectId(chat_id)}, update)

def get_chat(chats_col, chat_id):
    return chats_col.find_one({"_id": ObjectId(chat_id)})

def list_chats(chats_col, limit=10):
    return list(chats_col.find({}, {"messages": 0}).sort("updated_at", -1).limit(limit))

def print_chat(chat):
    print("\n================= HISTORIAL =================")
    print(f"Chat: {chat['_id']} | {chat.get('title','(sin título)')}")
    for m in chat.get("messages", []):
        who = "TÚ" if m["role"] == "user" else ("Barista Bot" if m["role"] == "assistant" else "TOOL")
        when = m.get("ts")
        when_str = when.strftime("%Y-%m-%d %H:%M:%S UTC") if when else ""
        print(f"\n[{who}] {when_str}\n{m['content']}")
    print("============================================\n")

# ---------------------------
# Agente “Barista Bot”
# ---------------------------

def main():
    """
    Coffee Expert Agent con persistencia de chats en MongoDB.
    Comandos de consola:
      :new              -> crear un chat nuevo
      :list             -> listar últimos chats
      :load <_id>       -> cargar chat por _id
      :exit / :quit     -> salir
    """
    # 1) ENV
    load_dotenv()
    google_api_key = os.getenv('GOOGLE_API_KEY')
    mongo_conn_string = os.getenv('MONGO_CONNECTION_STRING')
    google_cse_id = os.getenv('GOOGLE_CSE_ID')

    if not all([google_api_key, mongo_conn_string, google_cse_id]):
        raise ValueError("Please set GOOGLE_API_KEY, MONGO_CONNECTION_STRING, and GOOGLE_CSE_ID in the .env file.")

    os.environ['GOOGLE_API_KEY'] = google_api_key
    os.environ['GOOGLE_CSE_ID'] = google_cse_id

    # 2) MODELOS + VECTOR STORE
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.4, convert_system_message_to_human=True)
    embedding_function = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    DB_NAME = "coffee_db"
    COLLECTION_NAME = "knowledge_collection"
    INDEX_NAME = "coffee_index"

    client = MongoClient(mongo_conn_string)
    collection = client[DB_NAME][COLLECTION_NAME]

    # Colección de chats para historial
    chats_col = client[DB_NAME]["chats"]
    ensure_indexes(chats_col)

    vector_store = MongoDBAtlasVectorSearch(
        collection=collection,
        embedding=embedding_function,
        index_name=INDEX_NAME
    )
    retriever = vector_store.as_retriever(search_kwargs={'k': 5})

    # 3) RAG
    rag_prompt = PromptTemplate(
        template="Answer the user's question based ONLY on the following context:\n\n{context}\n\nQuestion: {input}",
        input_variables=["context", "input"],
    )
    rag_chain_base = create_stuff_documents_chain(llm, rag_prompt)
    rag_chain = create_retrieval_chain(retriever, rag_chain_base)

    def run_rag_tool(query_dict_or_str: dict | str) -> str:
        query = query_dict_or_str[list(query_dict_or_str.keys())[0]] if isinstance(query_dict_or_str, dict) else query_dict_or_str
        result = rag_chain.invoke({"input": query})
        return result.get('answer', "I couldn't find a specific answer in my knowledge base.")

    rag_tool = Tool(
        name="coffee_knowledge_base",
        func=run_rag_tool,
        description="Your primary tool. Use this to answer any questions about coffee brewing, origins, processing, history, roasting, and techniques. This is your own expert knowledge."
    )

    # 4) Google Search (solo si lo amerita, tema de café)
    search_wrapper = GoogleSearchAPIWrapper()
    web_search_tool = Tool(
        name="google_search",
        func=search_wrapper.run,
        description="A web search tool. Use this to find current or specific information on COFFEE-RELATED topics that are not in the knowledge base, like latest coffee prices, a specific new cafe, or recent competition results."
    )

    # 5) Agente
    tools = [rag_tool, web_search_tool]

    agent_prompt_template = """
    You are a world-class Coffee Master and sommelier named 'Barista Bot'.
    You are an expert on all things coffee and your goal is to provide accurate, helpful information about coffee with a friendly, educational, and passionate tone.

    You have access to two tools:
    1. `coffee_knowledge_base`: Your internal, expert knowledge base. Always try this tool FIRST for any coffee-related question.
    2. `Google Search`: A web search tool.

    Here are your strict rules:
    - **Rule 1:** For any question you receive, you MUST use the `coffee_knowledge_base` tool first.
    - **Rule 2:** If the `coffee_knowledge_base` tool provides a complete answer, use it directly.
    - **Rule 3:** If the `coffee_knowledge_base` tool does not provide a sufficient answer BUT the question is CLEARLY ABOUT COFFEE, you are then allowed to use `Google Search` to find the missing information.
    - **Rule 4:** If the user's question is NOT related to coffee (e.g., "what is the capital of Colombia?", "who won the soccer match?"), you MUST politely decline. You must say: "As a Coffee Master, my focus is solely on the fascinating world of coffee. I can't answer questions on that topic." Do NOT use the google_search tool for these questions.

    Think step-by-step and provide a final, comprehensive answer to the user.

    Begin!

    Question: {input}
    Thought: {agent_scratchpad}
    """
    agent_prompt = PromptTemplate.from_template(agent_prompt_template)
    agent = create_tool_calling_agent(llm, tools, agent_prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

    # 6) Loop con persistencia
    print("--------------------------------------------------")
    print("Barista Bot (Agent Mode) con historial persistente listo!")
    print("Comandos: :new | :list | :load <_id> | :exit")
    print("--------------------------------------------------")

    # Chat activo
    active_chat_id = str(create_chat(chats_col, title="Conversación de café"))
    print(f"Nuevo chat creado. _id = {active_chat_id}")

    while True:
        user_query = input("\nTú: ")

        # Comandos de control
        if user_query.strip().lower() in [":exit", ":quit", "exit", "quit", "salir"]:
            print("\nBarista Bot: ¡Un placer! ¡Hasta la próxima taza!")
            break

        if user_query.strip().lower() == ":new":
            active_chat_id = str(create_chat(chats_col, title="Conversación de café"))
            print(f"Nuevo chat creado. _id = {active_chat_id}")
            continue

        if user_query.strip().lower() == ":list":
            items = list_chats(chats_col, limit=10)
            print("\n=== Últimos chats ===")
            for it in items:
                print(f"- {it['_id']} | {it.get('title','(sin título)')} | updated_at: {it.get('updated_at')}")
            continue

        if user_query.strip().lower().startswith(":load"):
            parts = user_query.split()
            if len(parts) != 2:
                print("Uso: :load <_id>")
                continue
            try:
                # Verifica y carga
                _id = parts[1]
                chat = get_chat(chats_col, _id)
                if not chat:
                    print("No se encontró un chat con ese _id.")
                    continue
                active_chat_id = str(chat["_id"])
                print_chat(chat)
            except Exception as e:
                print(f"Error al cargar: {e}")
            continue

        # Guardar turno del usuario
        append_message(chats_col, active_chat_id, "user", user_query)

        # Retries por errores 500 temporales
        max_retries = 4
        retry_delay = 10  # segundos

        for attempt in range(max_retries):
            try:
                response = agent_executor.invoke({"input": user_query})
                bot_text = response.get("output", "")
                print(f"\nBarista Bot: {bot_text}")

                # Guardar turno del asistente
                append_message(chats_col, active_chat_id, "assistant", bot_text)
                # Si querés, podrías persistir también tool traces como role="tool"

                break

            except InternalServerError:
                print(f"\n[Agent Status] Error 500 temporal. Reintentando en {retry_delay}s... (Intento {attempt + 1}/{max_retries})")
                if attempt == max_retries - 1:
                    msg = "Estoy teniendo problemas para conectar con mi inteligencia. Inténtalo más tarde."
                    print(f"\nBarista Bot: {msg}")
                    append_message(chats_col, active_chat_id, "assistant", msg)
                    break
                time.sleep(retry_delay)

            except Exception as e:
                err_msg = f"Ocurrió un error inesperado: {e}. Por favor, intenta reformular."
                print(f"\nBarista Bot: {err_msg}")
                append_message(chats_col, active_chat_id, "assistant", err_msg)
                break

if __name__ == "__main__":
    main()
