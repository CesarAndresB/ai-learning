# Coffee Chat AI API

Minimal FastAPI server exposing your Barista Bot via REST so the Angular app can talk to it.

## Setup

1. Create and activate a virtualenv (optional but recommended)
2. Install dependencies
3. Ensure `.env` contains:

```
GOOGLE_API_KEY=...
MONGO_CONNECTION_STRING=...
GOOGLE_CSE_ID=...
```

## Run

```
uvicorn api:app --reload --port 8000
```

Then open the Angular app (http://localhost:4200) and send a message.

## Endpoints

- GET /health
- POST /chats -> { chat_id }
- GET /chats -> list recent
- GET /chats/{chat_id} -> full chat detail
- POST /chats/{chat_id}/messages { content } -> { reply }
