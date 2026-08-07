
import os
os.environ["HF_HUB_OFFLINE"] = "1"

import itertools
import re
import sys
import threading
import time
import agent
import config
import ollama


def clean_answer(text: str) -> str:
    # strip markdown code fences (```json ... ``` or ``` ... ```)
    text = re.sub(r"```[\w]*\n?.*?```", "", text, flags=re.DOTALL)
    # strip bare JSON tool-call objects {"name": ..., "parameters": ...}
    text = re.sub(r"\{[^{}]*\"name\"\s*:\s*\"search_knowledge_base\"[^{}]*\}", "", text, flags=re.DOTALL)
    # collapse extra blank lines left behind
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def spinner(stop_event):
    for ch in itertools.cycle("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"):
        if stop_event.is_set():
            break
        sys.stdout.write(f"\rThinking {ch}")
        sys.stdout.flush()
        time.sleep(0.1)
    sys.stdout.write("\r" + " " * 20 + "\r")


def main():

    print("\n" + "─" * 60)
    print("  Doc-RAG Assistant  |  type 'quit' to exit")
    print("─" * 60 + "\n")

    while True:

        usr_input = input("Q: ").strip()

        if not usr_input:
            continue

        if usr_input.lower() == "quit":
            print("Unloading model...")
            ollama.generate(model=config.MODEL, prompt="", keep_alive=0)
            break

        stop = threading.Event()
        t = threading.Thread(target=spinner, args=(stop,))
        t.start()

        answer = agent.run_agent(usr_input)

        stop.set()
        t.join()

        print("\nA:", clean_answer(answer))
        print("\n" + "─" * 60 + "\n")

if __name__ == "__main__":
    main()