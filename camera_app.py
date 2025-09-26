import cv2
import math
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


face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eyes_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

video_capture = cv2.VideoCapture(0)  # 0 is default camera


def detect_face(img, debug=False):
    #gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)

    # TODO: Have to pare down to just the one face we want to detect for simplifying control logic.
    # Decide how to pick the most prominent face
    faces = face_classifier.detectMultiScale(img, 1.1, 5, minSize=(40, 40))
    if debug:
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)
    return faces


# detect the eyes of a given image and face
def detect_eyes(img, face, debug=False):
    (x, y, w, h) = face

    eyes = eyes_classifier.detectMultiScale(img[y:y + h, x:x + w], scaleFactor=1.1, minNeighbors=5)
    if debug:
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(img, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (255, 255, 255), 1)
    return eyes


def draw_top_left_box(img):
    w = img.shape[1]
    h = img.shape[0]
    center_x = int(w/2)
    center_y = int(h/2)
    cv2.rectangle(img, (0, 0), (center_x, center_y), (255, 255, 255), 2)


def draw_top_right_box(img):
    w = img.shape[1]
    h = img.shape[0]
    center_x = int(w/2)
    center_y = int(h/2)
    cv2.rectangle(img, (center_x, 0), (w, center_y), (255, 255, 255), 2)


def draw_bottom_left_box(img):
    w = img.shape[1]
    h = img.shape[0]
    center_x = int(w/2)
    center_y = int(h/2)
    cv2.rectangle(img, (0, center_y), (center_x, h), (255, 255, 255), 2)


def draw_bottom_right_box(img):
    w = img.shape[1]
    h = img.shape[0]
    center_x = int(w/2)
    center_y = int(h/2)
    cv2.rectangle(img, (center_x, center_y), (w, h), (255, 255, 255), 2)


def draw_center_box(img):
    w = img.shape[1]
    h = img.shape[0]
    center_x = int(w/2)
    center_y = int(h/2)
    quarter_x = int(center_x/2)
    quarter_y = int(center_y/2)
    cv2.rectangle(img, (quarter_x,quarter_y), (center_x+quarter_x, center_y+quarter_y), (255, 255, 255), 2)


debug = True  # TODO: Make more formal debug or tie into verbal commands to turn on/off

# TODO: Insert program control flow
while True:

    result, video_frame = video_capture.read()  # read frames from the video
    if result is False:
        break  # terminate the loop if the frame is not read successfully

    faces = detect_face(video_frame, debug)
    # TODO: Could filter to the highest confidence/largest face to get only one face

    if (len(faces) > 0):  # only get eyes if there is a face detected
        face = faces[0]  # pare down to the first face
        (x, y, w, h) = face
        eyes = detect_eyes(video_frame, face, debug)  

        if (len(eyes) == 2):
            left_eye, right_eye = (eyes[0], eyes[1]) if eyes[0][0] > eyes[1][0] else (eyes[1], eyes[0])  # ensure consistant order of eyes
            deg = math.atan2((left_eye[1] - right_eye[1]), (left_eye[0] - right_eye[0]))

            if debug:  # Draw facial rotation
                center_x, center_y = (x + w/2), (y + h/2)  # position at center of face

                x_diff = h/2 * math.cos(deg+90 * math.pi / 180.0)  # h (height of face) as length and add 90 to get vertical line 
                y_diff = h/2 * math.sin(deg+90 * math.pi / 180.0)
                # Draw out lines up and down from the center point
                p1_x = center_x + x_diff
                p1_y = center_y + y_diff
                p2_x = center_x - x_diff
                p2_y = center_y - y_diff
                #print(f'({p1_x},{p1_y}),({p2_x},{p2_y})')
                cv2.line(video_frame,(int(p1_x),int(p1_y)),(int(p2_x),int(p2_y)),(255,0,0),5)
    

    draw_top_left_box(video_frame)
    draw_top_right_box(video_frame)
    draw_bottom_left_box(video_frame)
    draw_bottom_right_box(video_frame)
    draw_center_box(video_frame)

    cv2.imshow("Camera Application", video_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()

