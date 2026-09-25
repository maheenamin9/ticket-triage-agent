import sqlite3
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from langgraph.checkpoint.sqlite import SqliteSaver

from app.config import get_settings
from app.graph.build_graph import build_graph
from app.routers.tickets_router import router as tickets_router

load_dotenv()
get_settings()

CHECKPOINT_DB_PATH = "./data/checkpoints.db"


@asynccontextmanager
async def lifespan(app: FastAPI):
    Path(CHECKPOINT_DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(CHECKPOINT_DB_PATH, check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    app.state.graph = build_graph(checkpointer)

    yield

    conn.close()


app = FastAPI(title="Ticket Triage Agent", lifespan=lifespan)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


app.include_router(tickets_router)
