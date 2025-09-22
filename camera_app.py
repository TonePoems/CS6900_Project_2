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
def textToCommand(text):
    # TODO: Do any necessary processing to get text down to acceptable commands
    command = text.lower()
    return command

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

