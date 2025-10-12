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


## Setup
With the repo cloned, run the following commands to set up the virtual environment.

```
python -m venv .venv
source .venv/Scripts/Activate
pip install -r requirements.txt
```


## Running
This tool can be run with the following command

python camera_app.py
```
Note: The very first time you run the script, it will need to download the Text-to-Speech AI models from Hugging Face. This is a one-time download of a few hundred megabytes and may take a few minutes. Subsequent startups will be much faster.



While in the application you can use the following commands:
- `q`: Quit the application
- `d`: Debug mode will draw bounding boxes for the face, eyes, and the angle of the head

## Logic Explainers
Explainers for how we approached several open ended portions of the project.

### Command Detection
To determine potential valid commands from any speech to text input, the following logic was used.

Valid commands are "top left", "top right", "bottom left", "bottom, right", and "center". 

A first pass applies translations from synonyms to make the tool easier to use. For example, the command will interpret either "top" or "upper" as a valid indicator of the top row. This valid translations are stored in the `aliasDict`. 

The code then traverses backwards through the spoken words to determine the last command spoken. This allows a user to backtrack if they start saying a different location (ex. "lower, no, upper right"). If "center" is specified, this one-word command is valid and is returned. If it finds a command that fills in a row or column value, it will store that command and move to the prior word. Once it has found both a row and column command, it concats them and returns that well formed string.


### Timings
Current guidance provides a new navigation command every 2 seconds.


### Rotation Calculation
To calculate the roll of the face, we run an eye classifier on the face subset of the overall image. To remove false detections, the roll is only calculated when exactly two eyes are detected. The inverse tangent of the line between the eyes gives us the roll of the face. For displaying the roll value in debug mode, it is more intuitive to show the line as dividing the face in two vertically, so we add 90 to the deg.

