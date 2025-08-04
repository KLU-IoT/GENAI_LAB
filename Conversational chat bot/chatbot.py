import os
import queue
import sys
import json
import time
from vosk import Model, KaldiRecognizer
import sounddevice as sd
from gtts import gTTS
import playsound
import google.generativeai as genai

# Gemini API Setup
genai.configure(api_key="AIzaSyCvZLoiQbOlbyhqpXla2iHG15H16GLP25o")  # Replace with your API key

gemini_model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=genai.types.GenerationConfig(
        temperature=0.4,
        max_output_tokens=50,
    )
)

# Vosk Setup
vosk_model_path = "models/vosk-model-small-en-us-0.15"
if not os.path.exists(vosk_model_path):
    print("Vosk model not found")
    sys.exit(1)

vosk_model = Model(vosk_model_path)
recognizer = KaldiRecognizer(vosk_model, 16000)
q = queue.Queue()

def audio_callback(indata, frames, time_info, status):
    if status:
        print("Audio input status:", status, file=sys.stderr)
    q.put(bytes(indata))

def chat_with_gemini(prompt):
    try:
        response = gemini_model.generate_content(prompt)
        reply = response.text.strip().split(".")[0] + "."
        return reply
    except Exception as e:
        print("Gemini Error:", e)
        return "Something went wrong."

def speak(text):
    try:
        tts = gTTS(text=text, lang='en')
        filename = "temp_response.mp3"
        tts.save(filename)
        playsound.playsound(filename)
        if os.path.exists(filename):
            os.remove(filename)
    except Exception as e:
        print("TTS Error:", e)

def main():
    print("Voice assistant is running.")
    with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16',
                           channels=1, callback=audio_callback):
        while True:
            print("Listening...")
            data = q.get()
            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                text = result.get("text", "").strip()
                if text:
                    print("You:", text)
                    reply = chat_with_gemini(text)
                    print("Bot:", reply)
                    speak(reply)
            time.sleep(0.1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Exiting...")
    except Exception as e:
        print("Fatal error:", e)
