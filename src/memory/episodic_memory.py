"""Episodic memory: approved experiences kept OUTSIDE the model and looked up
at question time. The model's weights never change in this approach.

    approved experiences (SQLite) -> embed -> retrieve top-k -> paste into prompt
"""
import sqlite3

from src.memory.experience_repository import ExperienceRepository
from src.memory.memory_retriever import MemoryRetriever
from src.utils.config_loader import load_config


class EpisodicMemory:
    def __init__(self, conn: sqlite3.Connection, retriever: MemoryRetriever | None = None,
                 include_personal: bool | None = None):
        cfg = load_config("system_config")["episodic_memory"]
        self.repo = ExperienceRepository(conn)
        self.retriever = retriever or MemoryRetriever(
            cfg["embedding_model"], cfg["top_k"], cfg["min_similarity"])
        self.include_personal = cfg["include_personal"] if include_personal is None else include_personal

    def refresh(self) -> int:
        """Re-read APPROVED experiences from SQLite and re-embed them. Returns how many."""
        rows = [r for r in self.repo.live_statements() if r["status"] == "APPROVED"
                and (self.include_personal or r["category"] != "personal")]
        self.retriever.index([{"experience_id": r["experience_id"], "statement": r["statement"]}
                              for r in rows])
        return len(rows)

    def context_for(self, question: str) -> tuple[str | None, list[dict]]:
        """Return (text to put in the prompt or None, the retrieved facts)."""
        hits = self.retriever.retrieve(question)
        if not hits:
            return None, []
        lines = "\n".join(f"- {h['statement']}" for h in hits)
        return f"Known facts from memory (use them if they are relevant):\n{lines}", hits
