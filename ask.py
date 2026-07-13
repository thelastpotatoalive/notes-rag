"""
ask.py — Ask questions about your notes.
Run this after ingest.py has processed your PDFs.
Usage: python ask.py
"""

from rag import answer


def ask_one(question):
    """Asks a single question and prints the answer with sources."""
    print("\nThinking...")

    answer_text, sources = answer(question)

    print("\n" + "=" * 50)
    print("ANSWER:")
    print(answer_text)
    print("\nSOURCES:", ", ".join(sources))
    print("=" * 50)

def main():
    """Runs an interactive loop — ask questions until you type 'quit'."""
    print("=" * 50)
    print("Ask questions about your notes!")
    print("Type 'quit' or 'exit' to stop.")
    print("=" * 50)

    while True:
        question = input("\nYour question: ").strip()

        # Let the user leave
        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        # Skip empty input
        if not question:
            continue

        ask_one(question)


if __name__ == "__main__":
    main()