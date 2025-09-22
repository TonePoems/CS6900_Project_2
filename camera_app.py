import cv2
import pyttsx3
import speech_recognition as sr


# create a speech recognition object
r = sr.Recognizer()

# Use microphone to get input from a user
# Optional arg for duration of audio length
# Returns string of user's input
def speechToText(dur=3):
    with sr.Microphone() as source:
        # read the audio data from the default microphone
        audio_data = r.record(source, duration=dur)  # TODO: Adjust timing depending on flow of program
        # convert speech to text
        text = r.recognize_google(audio_data)
        return text

# Take some string input and return a valid command for the camera app
# top_left
# top_right
# bottom_left
# bottom_right
# center
def textToCommand(text):
    # The user will specify the position where they want their face using
    # commands such as: "top left", "top right", "bottom left", "bottom
    # right", and "center".
    
    text = text.lower()

    aliasDict = {
        "top": ["top", "upper"],
        "bottom": ["bottom", "lower"],
        "left": ["left"],
        "right": ["right"],
        "center": ["center", "central"]
        }
    
    potential_command = []
    # remove possible aliases
    for word in text.split():
        for key in aliasDict:
            if word in aliasDict[key]:
                potential_command.append(key)
        
    row, col = "", ""
    for i in reversed(potential_command):  # reversed to give last commands priority
        if i == "center":  # center is own command
            return i
        if (row == "") and ((i == "top") or (i == "bottom")):  # if no row has been selected and command specifies a row
            row = i
        if (col == "") and ((i == "left") or (i == "right")):  # if no col has been selected and command specifies a col
            col = i

        if (not row == "") and (not col == ""):  # if a row and column have been selected, return those values
            return row + "_" + col

    return ""  # error case


# print("top_left: " + textToCommand("top left"))
# print("top_left: " + textToCommand("left top"))
# print("center: " + textToCommand("center"))
# print("center: " + textToCommand("central"))
# print("bottom_right: " + textToCommand("lower right"))
# print("top_left: " + textToCommand("right left bottom top"))  # should take the last command in each axis
#print("top_left: " + textToCommand("right no left and bottom no top"))  # should take the last command in each axis
# print("center: " + textToCommand("top center"))

# Demo speechToText capability, along with textToCommand
# speechInput = speechToText()
# commandInput = textToCommand(speechInput)
# print(speechInput)
# print(commandInput)


def textToSpeech(text):
    # Turned into function to possibly use different, higher quality voice later
    
    pyttsx3.speak(text)  # Can be modified like https://pypi.org/project/pyttsx3/

# Example of text to speech
# textToSpeech("Testing the speech to text")

face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

video_capture = cv2.VideoCapture(0)  # 0 is default camera


def detect_bounding_box(vid):
    gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)

    # TODO: Have to pare down to just the one face we want to detect for simplifying control logic.
    # Decide how to pick the most prominent face
    faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
    for (x, y, w, h) in faces:
        cv2.rectangle(vid, (x, y), (x + w, y + h), (0, 255, 0), 4)
    return faces


# TODO: Insert program control flow
while True:

    result, video_frame = video_capture.read()  # read frames from the video
    if result is False:
        break  # terminate the loop if the frame is not read successfully

    faces = detect_bounding_box(
        video_frame
    )  # apply the function we created to the video frame

    cv2.imshow(
        "Camera Application", video_frame
    )  # display the processed frame in a window named "My Face Detection Project"

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()

