"""Terminal chat for the Wake Phase.  Run from the project root:

    python -m scripts.chat

Type 'quit' to exit. Each line shows how your message was classified.
"""
from src.memory.database import get_connection
from src.model.inference import ask
from src.sleep.sleep_state import SystemState
from src.model.model_loader import load_model
from src.wake.conversation_manager import ConversationManager


def main() -> None:
    bundle = load_model()
    ask(bundle, "Say hello.")  # warm-up
    chat = ConversationManager(bundle, system_state=SystemState(get_connection()))
    print(f"Awake. Session {chat.session_id}. Model: {bundle.version}. Type 'quit' to exit.\n")

    while True:
        try:
            message = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not message:
            continue
        if message.lower() in ("quit", "exit"):
            break
        result = chat.handle(message)
        if result["label"]:
            print(f"  [{result['label']}{' / personal' if result.get('is_personal') else ''}]")
        print(f"Assistant: {result['reply']}\n")

    print(f"Session ended. {len(chat.collector.collected)} pending experience(s) saved to data/raw/.")


if __name__ == "__main__":
    main()
