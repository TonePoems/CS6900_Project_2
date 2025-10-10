# CS6900_Project_2
Camera application for visually impaired individuals


## Setup
With the repo cloned, run the following commands to set up the virtual environment.

```
python -m venv .venv
source .venv/Scripts/Activate
pip install -r requirements.txt
```


## Running
This tool can be run with the following command

```
python camera_app.py
```

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

