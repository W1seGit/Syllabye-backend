# Calendar AI Backend

A small FastAPI backend that exposes:

- JWT-based auth (`/auth/register`, `/auth/login`)
- A LangChain-powered calendar/chat agent (`/chat`)
- CRUD endpoints for events (`/events`)

The agent uses OpenAI via `langchain-openai` and a SQLite database (`app.db`) via SQLAlchemy.

## Requirements

- Python 3.11 (recommended)
- An OpenAI API key

## Setup

1. **Clone / open this folder**

   ```bash
   cd syllabye-backend
   ```

2. **Create and activate a virtual environment**

   ```bash
   python -m venv venv
   # Windows PowerShell
   .\venv\Scripts\Activate.ps1
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**

   Create a `.env` file (not committed to git) in this folder:

   ```ini
   OPENAI_API_KEY=your-openai-key-here
   # Optional overrides
   # DATABASE_URL=sqlite:///./app.db
   # SECRET_KEY=change-me
   ```

   The app uses `python-dotenv` to load `.env` on startup.

## Running the API

Run the FastAPI app with Uvicorn:

```bash
uvicorn app.main:app --reload
```

The API will be available at `http://127.0.0.1:8000`.

- Interactive docs: `http://127.0.0.1:8000/docs`
- Health check: `GET /health`

## Auth & Events

- `POST /auth/register` – create a new user
- `POST /auth/login` – obtain a JWT access token
- All `/events` and `/chat` requests require `Authorization: Bearer <token>`

Event endpoints:

- `POST /events` – create event
- `GET /events` – list events (optionally by date)
- `GET /events/{id}` – get single event
- `PATCH /events/{id}` – update event
- `DELETE /events/{id}` – delete event

## Chat UI (local helper)

This repo includes a simple browser-based helper page `chat.html` for local development. It is **ignored by git** and not meant for production deployment.

Usage:

1. Start the backend (`uvicorn app.main:app --reload`).
2. Open `chat.html` in a browser (double-click or `start .\chat.html`).
3. Enter `http://127.0.0.1:8000` as the API base URL.
4. Log in with a valid username/password.
5. Chat with the calendar agent.

The agent:

- Uses OpenAI via `ChatOpenAI`.
- Has short-term per-user memory (recent conversation turns).
- Manages events only for the authenticated user.

## Database

- Default: SQLite file `app.db` in this folder.
- It is ignored by git and can be safely deleted/reset in development.

## Notes

- `.env` and other secrets are ignored by git via `.gitignore`.
- `app.db` and `chat.html` are local artifacts and are also ignored.
