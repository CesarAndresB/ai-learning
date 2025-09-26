# AI Learning – Coffee Chat App

This repository contains two projects:
- backend: FastAPI server exposing a coffee RAG + agent (Gemini) with MongoDB Atlas Vector Search, plus SSE streaming.
- chat-ui: Angular 20 app that provides the chat interface (history, new chat, temperature slider, and streaming view).

Below are end-to-end instructions to run both locally on macOS.

## Prerequisites

- Node.js 20+ and npm 10+ (for the Angular app)
- Python 3.11 – 3.13 (project tested with 3.13) and venv
- MongoDB Atlas cluster with a database `coffee_db` and a collection `knowledge_collection` plus a vector index (see notes)
- Google API key with access to Gemini models
- Google Custom Search Engine (CSE) ID for the web search tool

## 1) Backend (FastAPI)

Location: `backend/`

### Environment

Create a `backend/.env` with the following variables:

```
GOOGLE_API_KEY=your_google_api_key
MONGO_CONNECTION_STRING=your_mongodb_connection_string
GOOGLE_CSE_ID=your_google_cse_id
```

Notes:
- The Mongo connection string must allow access to your cluster and the database used here.
- GOOGLE_CSE_ID is required for the GoogleSearchAPIWrapper tool.

### Install and Run

Create and activate a virtual environment, then install dependencies and start the server.

```bash
# from the repo root
python3 -m venv venv
source venv/bin/activate

# install backend deps
pip install -r backend/requirements.txt

# start API (reload for dev)
cd backend
uvicorn api:app --reload --port 8000
```

The API will run at http://localhost:8000.

### API Endpoints (summary)

- GET /health → { status: "ok" }
- POST /chats → { chat_id }
- GET /chats?limit=N → recent chats
- GET /chats/{chat_id} → full chat detail
- POST /chats/{chat_id}/messages { content } → { reply }
- GET /chats/{chat_id}/stream?content=... → Server-Sent Events stream with:
  - event: verbose → step/status messages
  - event: delta → token/partial text
  - event: done → final signal
- GET /settings → { temperature }
- POST /settings { temperature } → { temperature } (rebuilds agent)

### Knowledge base and vector index

- The app expects a MongoDB Atlas Vector Search index named `coffee_index` on the `knowledge_collection` collection in the `coffee_db` database.
- Populate `knowledge_collection` with your coffee documents and ensure the embeddings model matches in code (`models/embedding-001`).

## 2) Frontend (Angular)

Location: `chat-ui/`

### Install and Run

```bash
# from the repo root (optionally keep the same shell with the backend venv activated)
cd chat-ui
npm install
npm start
```

The UI will run at http://localhost:4200 and is configured to call the backend at http://localhost:8000.

### Features

- Chat history sidebar with latest chats, new chat button, and chat selection
- Streaming assistant responses with inline verbose status (tool usage, analysis, finishing)
- Temperature slider (updates backend agent on change)
- Automatic conversation title after the first exchange

## Troubleshooting

- If the backend reloads with IndentationError or import issues, ensure you've updated `backend/api.py` fully and restarted Uvicorn.
- CORS: For development, CORS allows http://localhost:4200. Adjust for production.
- MongoDB network access must allow your IP. Verify your Atlas connection string and firewall rules.
- Google APIs: Ensure your GOOGLE_API_KEY has access to Gemini and your CSE ID is correct.

## Project layout

```
backend/
  api.py
  chatbot.py
  requirements.txt
  coffee_knowledge.txt (optional source material)
chat-ui/
  package.json
  src/
    app/
      ...
```

## Scripts quick reference

- Backend: uvicorn api:app --reload --port 8000
- Frontend: npm start (Angular dev server)

## License

This project is for learning purposes. Review API and data source licenses before production use.

##Example

<img width="1256" height="755" alt="image" src="https://github.com/user-attachments/assets/7c7313a8-7e97-41fd-a061-f86b42aeac26" />
