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
VOSK_MODEL_PATH = "models/vosk-model-small-en-us-0.15"  # Path to your Vosk model
RAG_SOURCE_FILE = "klu_iot.txt"  # Your provided text file
OLLAMA_MODEL = "phi3"  # Your local Ollama Phi-3 model

# --- Verify Vosk Model ---
if not os.path.exists(VOSK_MODEL_PATH):
    print(f"Vosk model not found at {VOSK_MODEL_PATH}")
    sys.exit(1)


# --- Load RAG Source ---
def load_rag_source(filename):
    """Loads the text file content line by line."""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return f.readlines()
    except FileNotFoundError:
        print(f"RAG source file '{filename}' not found. Please ensure it's in the correct directory.")
        sys.exit(1)


rag_source_lines = load_rag_source(RAG_SOURCE_FILE)


def retrieve_info_from_rag(query, source_lines, num_sentences=5):
    """
    A simple keyword-based retrieval for RAG.
    In a real-world scenario, you'd use embeddings and a vector database for better relevance.
    """
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
    """This is called (from a separate thread) for each audio block."""
    if status:
        print("Audio input status:", status, file=sys.stderr)
    q.put(bytes(indata))


def chat_with_ollama_rag(user_query):
    """
    Generates a response using Ollama's Phi-3 model,
    augmented with retrieved information from the RAG source.
    """
    # 1. Retrieve relevant information from the RAG source
    retrieved_context = retrieve_info_from_rag(user_query, rag_source_lines)

    # 2. Construct the prompt for Ollama - MODIFIED FOR BREVITY
    if retrieved_context:
        full_prompt = (
            f"Based on the following information, answer the question accurately and very concisely. "  # Added "very concisely"
            f"Limit your response to 1-2 sentences. "  # Added sentence limit instruction
            f"Cite the source numbers in brackets at the end of each sentence if directly from the information. "
            f"For example: 'The B.Tech IOT Program was established in 2020.'\n\n"
            f"Information:\n{retrieved_context}\n\n"
            f"Question: {user_query}"
        )
    else:
        full_prompt = user_query + ". Please answer very concisely in 1-2 sentences."  # Added for cases without RAG

    # 3. Get response from Ollama
    try:
        response = ollama.chat(model=OLLAMA_MODEL, messages=[{'role': 'user', 'content': full_prompt}])
        reply = response['message']['content'].strip()
        return reply
    except Exception as e:
        print(f"Ollama Error: {e}")
        return "I'm sorry, I couldn't connect to the language model. Please ensure Ollama is running."


def speak(text):
    """Converts text to speech and plays it."""
    try:
        tts = gTTS(text=text, lang='en', slow=False)  # Changed slow=False for faster speech
        filename = "temp_response.mp3"
        tts.save(filename)
        playsound.playsound(filename)
        if os.path.exists(filename):
            os.remove(filename)  # Delete the temporary audio file
    except Exception as e:
        print(f"TTS Error: {e}")
        print("Could not play audio. Check if 'playsound' and your audio setup are correct.")


def main():
    print("Voice assistant is running.")
    print("#" * 80)
    print("Type your query and press Enter, or type 'voice' to switch to voice input.")
    print("Press Ctrl+C to stop the program.")
    print("#" * 80)

    stream_active = False  # Flag to manage stream state
    audio_stream = None  # Initialize audio_stream to None

    while True:
        user_input = input("\nYou: ").strip()

        if user_input.lower() == "exit":
            print("Exiting voice assistant. Goodbye!")
            # Ensure the audio stream is properly closed on exit
            if stream_active and audio_stream:
                audio_stream.stop()
                audio_stream.close()
            break
        elif user_input.lower() == "voice":
            print("Switching to voice input. Speak now...")
            if not stream_active:
                # Start the audio stream only if not already active
                audio_stream = sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16',
                                                 channels=1, callback=audio_callback)
                audio_stream.start()
                stream_active = True

            # Listen for voice input for a certain duration or until recognized
            voice_text = ""
            start_time = time.time()
            # Give a fixed duration for voice input, e.g., 10 seconds
            while (time.time() - start_time) < 10:
                try:
                    data = q.get(timeout=1)  # Get with timeout to avoid blocking indefinitely
                    if recognizer.AcceptWaveform(data):
                        result = json.loads(recognizer.Result())
                        recognized_text = result.get("text", "").strip()
                        if recognized_text:
                            voice_text = recognized_text
                            print(f"Recognized: {voice_text}")
                            break  # Break loop if voice recognized
                except queue.Empty:
                    # No audio data in queue, continue waiting or timeout
                    pass

            if voice_text:
                text_to_process = voice_text
            else:
                print("No voice detected after timeout. Please type your query or 'voice' again.")
                # Ensure stream is closed if voice input was attempted but failed/timed out
                if stream_active and audio_stream:
                    audio_stream.stop()
                    audio_stream.close()
                    stream_active = False
                    while not q.empty():  # Clear queue
                        q.get_nowait()
                continue  # Go back to asking for text input if no voice was recognized

            if stream_active:
                audio_stream.stop()
                audio_stream.close()
                stream_active = False
                # Clear the queue in case of leftover audio data
                while not q.empty():
                    q.get_nowait()
        else:
            text_to_process = user_input

        if text_to_process:
            print("Bot is thinking...")
            reply = chat_with_ollama_rag(text_to_process)
            print("Bot:", reply)
            speak(reply)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting voice assistant. Goodbye!")
    except Exception as e:
        print(f"\nFatal error: {e}")
        sys.exit(1)