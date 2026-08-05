
import chromadb
import itertools
import sys
import threading
import time
from sentence_transformers import SentenceTransformer
import config
import agent


def spinner(stop_event):
    for ch in itertools.cycle("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"):
        if stop_event.is_set():
            break
        sys.stdout.write(f"\rThinking {ch}")
        sys.stdout.flush()
        time.sleep(0.1)
    sys.stdout.write("\r" + " " * 20 + "\r")


def main():

    client = chromadb.PersistentClient(path="chroma_db")
    collection = client.get_or_create_collection(config.COLLECTION)

    encoder = SentenceTransformer(config.ENCODER)

    print("\nInput 'quit' to end program\n")

    while True:

        usr_input = input("Question: ")

        if (usr_input.strip().lower()) == "quit":
            break

        stop = threading.Event()
        t = threading.Thread(target=spinner, args=(stop,))
        t.start()

        answer, sources = agent.run_agent(usr_input, collection, encoder)

        stop.set()
        t.join()

        print(f"{answer}\n\n")

if __name__ == "__main__":
    main()