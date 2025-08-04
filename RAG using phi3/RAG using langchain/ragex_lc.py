import os
import queue
import sys
import json
import time
import re

# Vosk and Sounddevice (Existing STT/TTS components)
from vosk import Model, KaldiRecognizer
import sounddevice as sd
from gtts import gTTS
import playsound

# LangChain Imports - UPDATED TO RESOLVE DEPRECATION WARNINGS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# Changed import path for SentenceTransformerEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings  # New import path
from langchain_community.vectorstores import Chroma
# Changed import path for ChatOllama
from langchain_ollama import ChatOllama  # New import path
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# --- Configuration ---
VOSK_MODEL_PATH = "models/vosk-model-small-en-us-0.15"
RAG_SOURCE_FILE = "klu_iot.txt"
OLLAMA_MODEL = "phi3"
CHROMA_DB_DIRECTORY = "./chroma_db"

# --- Verify Vosk Model ---
if not os.path.exists(VOSK_MODEL_PATH):
    print(f"Vosk model not found at {VOSK_MODEL_PATH}")
    sys.exit(1)

# --- LangChain RAG Setup (Happens once at startup) ---
print("Setting up LangChain RAG components...")
try:
    # 1. Load the document
    loader = TextLoader(RAG_SOURCE_FILE, encoding='utf-8')
    documents = loader.load()

    # 2. Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)

    # 3. Create Embeddings - Using the updated import
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 4. Create or load the Vector Store
    vectorstore = Chroma.from_documents(
        docs,
        embeddings,
        persist_directory=CHROMA_DB_DIRECTORY
    )
    # 5. Remove manual persist() call as it's deprecated and automatic
    # vectorstore.persist() # REMOVED THIS LINE

    print(f"ChromaDB initialized/loaded at {CHROMA_DB_DIRECTORY}")

    # 6. Create a Retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

except Exception as e:
    print(f"Error setting up LangChain RAG: {e}")
    print("Please ensure you have installed all LangChain dependencies and the RAG_SOURCE_FILE exists.")
    sys.exit(1)

# --- LangChain LLM and Chain Setup ---
# Using the updated import
llm = ChatOllama(model=OLLAMA_MODEL, temperature=0.1)
# You can add Ollama options here, e.g., for faster generation:
# llm = ChatOllama(model=OLLAMA_MODEL, temperature=0.1, num_predict=50)


# Define the prompt template for the LLM
prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a helpful assistant specialized in the B.Tech IOT Program. "
        "Based on the following context, answer the question accurately and very concisely, in 1-2 sentences. "
        "Do NOT invent or fabricate information beyond the provided context. "
        "For example: 'The B.Tech IOT Program was established in 2020.'\n\n"
        "Context: {context}"
    )),
    ("human", "{input}")
])

# Create a chain to combine documents into the prompt
document_chain = create_stuff_documents_chain(llm, prompt)

# Create the full retrieval chain
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# --- Vosk Setup (Existing) ---
vosk_model = Model(VOSK_MODEL_PATH)
recognizer = KaldiRecognizer(vosk_model, 16000)
q = queue.Queue()


def audio_callback(indata, frames, time_info, status):
    if status:
        print("Audio input status:", status, file=sys.stderr)
    q.put(bytes(indata))


# --- Chatbot Response Function (Now uses LangChain) ---
def get_chatbot_response(user_query):
    try:
        response = retrieval_chain.invoke({"input": user_query})
        reply = response.get("answer", "I couldn't find an answer based on the provided information.").strip()
        return reply
    except Exception as e:
        print(f"LangChain/Ollama Error: {e}")
        return "I'm sorry, I encountered an issue connecting to the language model or processing the request."


def speak(text):
    try:
        tts = gTTS(text=text, lang='en', slow=False)
        filename = "temp_response.mp3"
        tts.save(filename)
        playsound.playsound(filename)
        if os.path.exists(filename):
            os.remove(filename)
    except Exception as e:
        print(f"TTS Error: {e}")
        print("Could not play audio. Check if 'playsound' and your audio setup are correct.")


def main():
    print("Voice assistant is running.")
    print("#" * 80)
    print("Type your query and press Enter, or type 'voice' to switch to voice input.")
    print("Press Ctrl+C to stop the program.")
    print("#" * 80)

    stream_active = False
    audio_stream = None

    while True:
        user_input = input("\nYou: ").strip()

        if user_input.lower() == "exit":
            print("Exiting voice assistant. Goodbye!")
            if stream_active and audio_stream:
                audio_stream.stop()
                audio_stream.close()
            break
        elif user_input.lower() == "voice":
            print("Switching to voice input. Speak now...")
            if not stream_active:
                audio_stream = sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16',
                                                 channels=1, callback=audio_callback)
                audio_stream.start()
                stream_active = True

            voice_text = ""
            start_time = time.time()
            while (time.time() - start_time) < 10:
                try:
                    data = q.get(timeout=1)
                    if recognizer.AcceptWaveform(data):
                        result = json.loads(recognizer.Result())
                        recognized_text = result.get("text", "").strip()
                        if recognized_text:
                            voice_text = recognized_text
                            print(f"Recognized: {voice_text}")
                            break
                except queue.Empty:
                    pass

            if voice_text:
                text_to_process = voice_text
            else:
                print("No voice detected after timeout. Please type your query or 'voice' again.")
                if stream_active and audio_stream:
                    audio_stream.stop()
                    audio_stream.close()
                    stream_active = False
                    while not q.empty():
                        q.get_nowait()
                continue

            if stream_active:
                audio_stream.stop()
                audio_stream.close()
                stream_active = False
                while not q.empty():
                    q.get_nowait()
        else:
            text_to_process = user_input

        if text_to_process:
            print("Bot is thinking...")
            reply = get_chatbot_response(text_to_process)
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