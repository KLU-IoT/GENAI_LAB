import os
import queue
import sys
import json
import time
from vosk import Model, KaldiRecognizer
import sounddevice as sd
from gtts import gTTS
import playsound
import re
import ollama

# --- Configuration ---
VOSK_MODEL_PATH = "models/vosk-model-small-en-us-0.15"
RAG_SOURCE_FILE = "klu_iot.txt"
OLLAMA_MODEL = "phi3"

# --- Verify Vosk Model ---
if not os.path.exists(VOSK_MODEL_PATH):
    print(f"Vosk model not found at {VOSK_MODEL_PATH}")
    sys.exit(1)


# --- Load RAG Source ---
def load_rag_source(filename):
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return f.readlines()
    except FileNotFoundError:
        print(f"RAG source file '{filename}' not found. Please ensure it's in the correct directory.")
        sys.exit(1)


rag_source_lines = load_rag_source(RAG_SOURCE_FILE)


def retrieve_info_from_rag(query, source_lines, num_sentences=5):
    relevant_sentences = []
    query_words = set(query.lower().split())

    cited_sentences = {}
    for i, line in enumerate(source_lines):
        match = re.match(r'\\s*(.*)', line)
        if match:
            source_nums_str = match.group(1)
            content = match.group(2).strip()
            source_nums = [int(s.strip()) for s in source_nums_str.split(',')]
        else:
            content = line.strip()
            source_nums = []

        if not content:
            continue

        if any(word in content.lower() for word in query_words):
            if content not in cited_sentences:
                cited_sentences[content] = set()
            cited_sentences[content].update(source_nums)

    for sentence_content, source_numbers in cited_sentences.items():
        if source_numbers:
            citation_str = ", ".join(map(str, sorted(list(source_numbers))))
            relevant_sentences.append(f"{sentence_content}")
        else:
            relevant_sentences.append(sentence_content)

    return "\n".join(relevant_sentences[:num_sentences])


# Vosk Setup
vosk_model = Model(VOSK_MODEL_PATH)
recognizer = KaldiRecognizer(vosk_model, 16000)
q = queue.Queue()


def audio_callback(indata, frames, time_info, status):
    if status:
        print("Audio input status:", status, file=sys.stderr)
    q.put(bytes(indata))


def chat_with_ollama_rag(user_query):
    # 1. Retrieve relevant information from the RAG source
    retrieved_context = retrieve_info_from_rag(user_query, rag_source_lines)

    # 2. Construct the prompt for Ollama
    if retrieved_context:
        full_prompt = (
            f"Based on the following information, answer the question accurately and concisely. "
            f"Cite the source numbers in brackets at the end of each sentence if directly from the information. "
            f"For example: 'The B.Tech IOT Program was established in 2020[cite: 1].'\n\n"  # CORRECTED LINE AGAIN
            f"Information:\n{retrieved_context}\n\n"
            f"Question: {user_query}"
        )
    else:
        full_prompt = user_query

    # 3. Get response from Ollama
    try:
        response = ollama.chat(model=OLLAMA_MODEL, messages=[{'role': 'user', 'content': full_prompt}])
        reply = response['message']['content'].strip()
        return reply
    except Exception as e:
        print(f"Ollama Error: {e}")
        return "I'm sorry, I couldn't connect to the language model. Please ensure Ollama is running."


def speak(text):
    try:
        tts = gTTS(text=text, lang='en')
        filename = "temp_response.mp3"
        tts.save(filename)
        playsound.playsound(filename)
        if os.path.exists(filename):
            os.remove(filename)
    except Exception as e:
        print(f"TTS Error: {e}")
        print("Could not play audio. Check if 'playsound' and your audio setup are correct.")


def main():
    print("Voice assistant is running. Speak into your microphone.")
    with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16',
                           channels=1, callback=audio_callback):
        print("#" * 80)
        print("Press Ctrl+C to stop the program.")
        print("#" * 80)

        while True:
            data = q.get()

            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                text = result.get("text", "").strip()

                if text:
                    print("\nYou:", text)
                    reply = chat_with_ollama_rag(text)
                    print("Bot:", reply)
                    speak(reply)
            else:
                pass

            time.sleep(0.05)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting voice assistant. Goodbye!")
    except Exception as e:
        print(f"\nFatal error: {e}")
        sys.exit(1)