Here is a professional and detailed `README.md` for your **Chat Bot** project that uses **Vosk** for speech recognition and **Gemini (via Google Generative AI API)** for generating responses.

---

# Chat Bot – Voice Assistant using Vosk + Gemini

A simple voice assistant that listens to your speech, sends your query to **Gemini (via Google Generative AI API)**, and speaks back the generated response using **gTTS**.

---

## Table of Contents

* [Features](#features)
* [Requirements](#requirements)
* [Installation](#installation)
* [Project Structure](#project-structure)
* [Configuration](#configuration)
* [Usage](#usage)
* [How It Works](#how-it-works)
* [Troubleshooting](#troubleshooting)
* [Notes](#notes)

---

## Features

* **Speech-to-Text:** Uses [Vosk](https://alphacephei.com/vosk/) for real-time voice transcription.
* **Generative AI Integration:** Sends user input to **Gemini 1.5 Flash** via Google's Generative AI API.
* **Text-to-Speech:** Uses [gTTS](https://pypi.org/project/gTTS/) to convert responses into spoken output.
* **Offline Speech Recognition:** Vosk operates fully offline for STT.
* **Custom Prompting:** Modify the prompt or Gemini model parameters for different use-cases.

---

## Requirements

* Python 3.8 or later
* Internet connection (for Gemini API and gTTS)
* Microphone and speaker access
* Google API Key for Gemini

---

## Installation

1. **Clone this repository:**

```bash
git clone --branch feature/audio https://github.com/KLU-IoT/GENAI_LAB.git
cd "GENAI_LAB/Conversational chat bot"
```

2. **Create a virtual environment:**

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

3. **Install dependencies:**

```bash
pip install -r requirements.txt
```

**requirements.txt**

```
vosk
sounddevice
gTTS
playsound
google-generativeai
```

4. **Download and place the Vosk model:**

Download the model [here](https://alphacephei.com/vosk/models) and extract it to:

```
models/vosk-model-small-en-us-0.15/
```

---

## Project Structure

```
chat-bot/
├── chatbot.py               # Main Python script
├── requirements.txt         # Python dependencies
├── README.md                # This file
└── models/
    └── vosk-model-small-en-us-0.15/  # Vosk English STT model
```

---

## Configuration

In `chatbot.py`, set your **Gemini API key**:

```python
genai.configure(api_key="your-api-key-here")
```

Ensure this key is valid and has access to `gemini-1.5-flash`.

---

## Usage

Activate the virtual environment and run:

```bash
python chatbot.py
```

Once started:

* Speak into your microphone.
* The bot will print your transcribed input.
* Gemini generates a response.
* The response is printed and spoken aloud.

To exit, press `Ctrl+C`.

---

## How It Works

1. **Listen and Transcribe:**
   Captures audio from your mic and uses Vosk's `KaldiRecognizer` to convert it to text.

2. **Send Prompt to Gemini:**
   The transcribed text is passed to Gemini (via Google Generative AI SDK) for a short, concise answer.

3. **Speak the Response:**
   The bot speaks the response aloud using `gTTS` and `playsound`.

---

## Troubleshooting

### Vosk Model Not Found

Make sure you downloaded and extracted the Vosk model to the correct directory:

```
models/vosk-model-small-en-us-0.15/
```

### Gemini API Issues

* Ensure your API key is valid.
* You must have access to the Gemini 1.5 Flash model.
* Your API key must not exceed quota limits.

### No Audio Output

* Ensure your system audio is working.
* gTTS requires internet access.
* `playsound` may have issues on some platforms; try replacing it with another library if needed.

---

## Notes

* The Gemini model used here (`gemini-1.5-flash`) is optimized for quick, lightweight responses. You can change this to `gemini-1.5-pro` if you need more in-depth responses.
* Modify `max_output_tokens` and `temperature` in the `GenerationConfig` to tune response length and creativity.
* gTTS uses Google’s servers, so an internet connection is required even though STT runs offline.

---


