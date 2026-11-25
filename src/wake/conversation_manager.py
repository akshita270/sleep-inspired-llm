"""The Wake Phase conversation loop.

For each user message:
    classify it -> if it is a question, ask the model -> otherwise acknowledge it
    -> if it is a fact/correction/feedback, store a PENDING experience.

The model's own weights are never touched here. Statements are only *recorded*.
"""
import uuid

from src.model.inference import ask
from src.model.model_loader import ModelBundle
from src.wake import interaction_classifier as ic
from src.wake.experience_collector import ExperienceCollector


class ConversationManager:
    def __init__(self, bundle: ModelBundle, collector: ExperienceCollector | None = None,
                 system_state=None):
        self.bundle = bundle
        self.session_id = uuid.uuid4().hex[:8]
        self.collector = collector or ExperienceCollector(self.session_id)
        self.system_state = system_state   # shared AWAKE/SLEEPING switch (SQLite-backed)
        self.last_question: str | None = None
        self.last_answer: str | None = None

    def handle(self, message: str) -> dict:
        """Process one user message. Returns what to show and what was recorded."""
        status = self.system_state.get() if self.system_state else "AWAKE"
        if status != "AWAKE":
            return {"reply": f"I am {status.lower()} right now and cannot chat. Please try again soon.",
                    "label": None, "is_personal": False, "experience": None}

        c = ic.classify(message)
        experience = None

        if c.label == ic.QUESTION:
            out = ask(self.bundle, c.text)
            self.last_question, self.last_answer = c.text, out["answer"]
            reply = out["answer"]
        else:
            experience = self.collector.collect(
                c, message, self.last_question, self.last_answer)
            reply = self._acknowledge(c, experience)

        return {"reply": reply, "label": c.label, "is_personal": c.is_personal,
                "experience": experience}

    @staticmethod
    def _acknowledge(c: ic.Classification, exp) -> str:
        """Fixed replies (no model call): honest about what has and has not happened."""
        if exp is None:
            return "Hello! Ask me something, or tell me something new."
        if c.label == ic.CORRECTION:
            reply = ("Thanks. I have noted your correction for review. "
                     "I have not learned it yet, and I cannot assume it is correct.")
        elif c.label == ic.NEW_FACT:
            reply = "Understood. I have noted that for review. I have not learned it yet."
        else:
            reply = "Thanks for the feedback. I have noted it."
        if exp.requires_explicit_approval:
            reply += " It looks personal, so it will not be used for training without your explicit approval."
        return reply
