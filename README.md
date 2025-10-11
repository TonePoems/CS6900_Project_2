# CS6900_Project_2
Camera application : Selfie Helper for Visually Impaired Users
This is a desktop application written in Python that enables visually impaired users to take selfie images using only speech-based commands.

## Features

* **Voice-Controlled Positioning:** Users can specify where they want their face (e.g., "top left," "center") using voice commands.
* **Live Voice Guidance:** The application provides real-time audio feedback to guide the user into the correct position and to level their head.
* **Automatic Photo Capture:** Once the user is in the correct position with their head straight, the application automatically takes a picture.
* **High-Quality Text-to-Speech:** Uses a modern AI model from Hugging Face (`microsoft/speecht5_tts`) to provide clear, natural-sounding, and non-freezing voice prompts.
* **Background Listening:** The application listens for the initial command in the background, allowing the video feed to run smoothly without interruption.

## Usage
Create a Virtual Environment
It is highly recommended to use a Python virtual environment.


# On Windows
python -m venv .venv
.\.venv\Scripts\Activate


# Install PyTorch
This project uses the PyTorch library to run the AI model. Install it separately using the official command from their website for the most stable version. For a standard CPU-only installation, use:

pip install torch torchvision torchaudio


# Install Other Dependencies
Install the remaining required packages using the requirements.txt file.

pip install -r requirements.txt 


# How to Run
Make sure your virtual environment is activated.
Run the main application script:

python camera_app.py


Note: The very first time you run the script, it will need to download the Text-to-Speech AI models from Hugging Face. This is a one-time download of a few hundred megabytes and may take a few minutes. Subsequent startups will be much faster.

---



