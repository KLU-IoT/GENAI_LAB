import speech_recognition as sr
import requests
import os
import tempfile
import simpleaudio as sa  # For playing WAV files
import wave  # To handle WAV file headers for simpleaudio
import numpy as np  # To convert raw PCM to numpy array if needed
import base64  # Needed for base64 decoding of audio data

# Import the Google Generative AI client library
import google.generativeai as genai
from google.generativeai.types import GenerationConfig  # Keep GenerationConfig import

# --- Configuration ---
# IMPORTANT: Your Gemini API Key for both translation and TTS
GEMINI_API_KEY = "AIzaSyCvZLoiQbOlbyhqpXla2iHG15H16GLP25o"
GEMINI_TRANSLATION_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
# The TTS API is typically accessed via the client library with a specific model
GEMINI_TTS_MODEL = "gemini-2.5-flash-preview-002"  # Model known to support TTS

# Configure the Gemini API client
if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_GEMINI_API_KEY":
    print("Error: Gemini API Key is not set. Please update GEMINI_API_KEY in the script.")
    exit(1)
genai.configure(api_key=GEMINI_API_KEY)

# Initialize the Gemini model for TTS
tts_model = None  # Initialize as None
try:
    tts_model = genai.GenerativeModel(GEMINI_TTS_MODEL)
    print(f"Gemini TTS model '{GEMINI_TTS_MODEL}' initialized successfully.")
except Exception as e:
    print(f"ERROR: Could not initialize Gemini TTS model: {e}")
    print("Ensure the Gemini TTS model is enabled for your API key and region.")
    # Do not exit here, allow the rest of the script to run, but TTS will be disabled


# --- Helper Functions ---

def get_gemini_response(prompt_text):
    """
    Sends a prompt to the Gemini API for translation and returns the generated text.
    """
    headers = {
        "Content-Type": "application/json"
    }
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt_text}]
            }
        ]
    }

    try:
        response = requests.post(f"{GEMINI_TRANSLATION_API_URL}?key={GEMINI_API_KEY}", headers=headers, json=payload)
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)
        result = response.json()

        if result.get("candidates") and result["candidates"][0].get("content") and result["candidates"][0][
            "content"].get("parts"):
            return result["candidates"][0]["content"]["parts"][0].get("text")  # Use .get() for safety
        else:
            print("Gemini API translation response did not contain expected content.")
            return None
    except requests.exceptions.HTTPError as errh:
        print(f"HTTP Error during translation: {errh}")
        print(f"Response body: {errh.response.text}")
        return None
    except requests.exceptions.ConnectionError as errc:
        print(f"Error Connecting for translation: {errc}")
        return None
    except requests.exceptions.Timeout as errt:
        print(f"Timeout Error for translation: {errt}")
        return None
    except requests.exceptions.RequestException as err:
        print(f"Something went wrong with the translation request: {err}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during translation API call: {e}")
        return None


def speak_text(text):
    """
    Converts text to speech using Gemini API's speech generation and plays it.
    """
    print(f"Bot: {text}")  # Print the bot's response to the console

    if not tts_model:
        print(f"WARNING: Gemini TTS model not initialized. Cannot speak: {text}")
        return

    # Determine language for TTS based on content heuristic
    # Telugu Unicode range: U+0C00 to U+0C7F
    if any(0x0C00 <= ord(char) <= 0x0C7F for char in text):
        lang_code = 'te-IN'  # Telugu (India)
        # For Telugu, let the API pick a default voice for the language code.
        # Specific voice names like 'te-IN-Wavenet-A' might exist but need to be verified in docs.
        voice_name = None  # Let API choose default voice for te-IN
    else:
        lang_code = 'en-US'  # English (United States)
        # A common English voice. You can experiment with others if available.
        # Examples: "en-US-Standard-C", "en-US-Wavenet-F"
        voice_name = "en-US-Standard-C"  # Use a specific English voice for consistency

    try:
        print(f"Generating speech audio via Gemini TTS API (Language: {lang_code})...")

        # Corrected structure for speech_config
        # language_code and voice_name go inside prebuilt_voice_config
        # which is nested under voice_config
        speech_config_dict = {
            "voice_config": {
                "prebuilt_voice_config": {
                    "language_code": lang_code,
                }
            }
        }
        if voice_name:
            speech_config_dict["voice_config"]["prebuilt_voice_config"]["voice_name"] = voice_name

        generation_config_dict = {
            "response_modalities": ["AUDIO"],
            "speech_config": speech_config_dict
        }

        # Make the generate_content call using the client library
        response = tts_model.generate_content(
            contents=[{"role": "user", "parts": [{"text": text}]}],
            generation_config=generation_config_dict  # Pass dictionary here
        )

        # The audio data is in response.candidates[0].content.parts[0].inline_data.data
        if response.candidates and response.candidates[0].content and \
                response.candidates[0].content.parts and response.candidates[0].content.parts[0].inline_data:
            audio_bytes_base64 = response.candidates[0].content.parts[0].inline_data.data
            audio_bytes = base64.b64decode(audio_bytes_base64)  # Decode base64 audio

            print("Audio bytes received. Playing...")

            # Save to a temporary WAV file and play
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as fp:
                temp_audio_file = fp.name
                fp.write(audio_bytes)

            # Play the temporary WAV file using simpleaudio
            wave_obj = sa.WaveObject.from_file(temp_audio_file)
            play_obj = wave_obj.play()
            play_obj.wait_done()  # Wait until sound has finished playing

            os.remove(temp_audio_file)  # Clean up the temporary file
            print("Speech finished and temporary file deleted.")
        else:
            print("Gemini TTS response did not contain valid audio data.")

    except Exception as e:
        print(f"ERROR: An unexpected error occurred during Gemini TTS or audio playback: {e}")
        print(
            "Ensure simpleaudio and its system dependencies are correctly installed, and check Gemini API logs for more details.")


def main():
    """
    Main function to run the conversational translator.
    """
    r = sr.Recognizer()
    mic = sr.Microphone()

    print("--- Voice Translator (Telugu <-> English) ---")
    print("Say 'stop' to end the conversation.")
    print("Initializing microphone...")

    # Adjust for ambient noise once at the start
    try:
        with mic as source:
            print("Adjusting for ambient noise... (please be quiet)")
            r.adjust_for_ambient_noise(source)
        print("Microphone ready. Start speaking!")
    except Exception as e:
        print(f"ERROR: Could not access microphone or adjust for ambient noise: {e}")
        print("Please ensure your microphone is connected, enabled, and Python has permissions to access it.")
        print("Exiting...")
        return  # Exit if microphone cannot be initialized

    while True:
        print("\nListening...")
        try:
            with mic as source:
                audio = r.listen(source)

            print("Processing speech...")
            # Use Google Web Speech API for speech-to-text
            # It can often detect language automatically, but we'll still prompt Gemini for explicit detection.
            recognized_text = r.recognize_google(audio)
            print(f"You: {recognized_text}")  # User's input is still printed for clarity

            if recognized_text.lower() == "stop":
                print("Stopping the translator. Goodbye!")
                break

            # Prompt for Gemini to detect language and translate
            prompt = (
                f"Detect the language of the following text: '{recognized_text}'. "
                f"If it is English, translate it to Telugu. "
                f"If it is Telugu, translate it to English. "
                f"If it is neither English nor Telugu, respond with 'Unsupported language'. "
                f"Provide only the translated text, or 'Unsupported language'."
            )

            translated_text = get_gemini_response(prompt)

            if translated_text:
                if translated_text.lower() == 'unsupported language':
                    print("Bot: Unsupported language detected. Please speak in English or Telugu.")
                else:
                    speak_text(translated_text)
            else:
                print("Bot: Could not get a translation. Please try again.")

        except sr.UnknownValueError:
            print("Could not understand audio. Please speak clearly.")
        except sr.RequestError as e:
            print(f"Could not request results from Google Speech Recognition service; {e}")
            print("Please check your internet connection and Gemini API key.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            print("If you are having microphone issues, ensure it's properly configured.")


if __name__ == "__main__":
    main()
