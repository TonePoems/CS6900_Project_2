import copy
import cv2
import datetime
import math
import pyttsx3
import speech_recognition as sr
import time # We'll need this to avoid spamming the user with guidance

# Create a single, shared text-to-speech engine for the whole app
engine = pyttsx3.init()

# create a speech recognition object
r = sr.Recognizer()

# Use microphone to get input from a user
# Optional arg for duration of audio length
# Returns string of user's input
def speechToText(dur=3):
    with sr.Microphone() as source:
        print("Listening for command...")
        r.adjust_for_ambient_noise(source, duration=1)
        try:
            # Use listen instead of record for better silence detection
            audio_data = r.listen(source, timeout=dur, phrase_time_limit=dur)
            print("Recognizing...")
            text = r.recognize_google(audio_data)
            return text
        except sr.UnknownValueError:
            # This is the safety net that catches the error
            print("Google Speech Recognition could not understand audio")
            return "" # Return an empty string instead of crashing
        except sr.RequestError as e:
            print(f"Could not request results from Google Speech Recognition service; {e}")
            return ""
        except sr.WaitTimeoutError:
            print("Listening timed out while waiting for phrase to start")
            return ""


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
    """Converts text to speech using the shared engine, preventing conflicts."""
    print(f"SPEAKING: {text}")
    try:
        # Stop any speech that's currently happening
        engine.stop()
        # Queue up the new text to be spoken
        engine.say(text)
        # Process the speech command and wait for it to finish
        engine.runAndWait()
    except RuntimeError:
        # This can happen in rare cases, just ignore it to prevent a crash
        pass
    #pyttsx3.speak(text)  # Can be modified like https://pypi.org/project/pyttsx3/

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


# boxes with fake overlap for cleaner overlay
def draw_all_boxes(img):
    w = img.shape[1]
    h = img.shape[0]
    center_x = int(w/2)
    center_y = int(h/2)
    quarter_x = int(center_x/2)
    quarter_y = int(center_y/2)
    cv2.rectangle(img, (quarter_x,quarter_y), (center_x+quarter_x, center_y+quarter_y), (255, 255, 255), 2)  # center box
    cv2.line(img,(center_x, 0),(center_x, quarter_y),(255, 255, 255), 2)  # top line
    cv2.line(img,(center_x, h),(center_x, center_y+quarter_y),(255, 255, 255), 2)  # bottom line
    cv2.line(img,(0, center_y),(quarter_x, center_y),(255, 255, 255), 2)  # left line
    cv2.line(img,(w, center_y),(center_x+quarter_x, center_y),(255, 255, 255), 2)  # right line


def save_photo(img, aux_text):
    date_time = datetime.datetime.now().strftime('%Y%m%d_%H-%M-%S')
    title = f'./output/{date_time}_{aux_text}.png'
    #print(title)
    cv2.imwrite(title, img)





# We need a global variable to hold the command from the background thread
target_command = ""

def command_callback(recognizer, audio):
    """
    This function is called in the background when speech is detected.
    """
    global target_command
    try:
        # We got audio, now we recognize it
        speech = recognizer.recognize_google(audio)
        print(f"I heard you say: '{speech}'")
        command = textToCommand(speech)
        if command:
            target_command = command # Set the global command variable
    except sr.UnknownValueError:
        textToSpeech("I could not understand that. Please say a command again")
    except sr.RequestError as e:
        textToSpeech(f"Sorry the Speech service is unavailable; {e}")


def main_application():
    """This function runs the main application workflow with background listening."""
    global target_command
    
    # 1. SETUP AND INITIALIZATION
    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        textToSpeech("Error: Cannot open camera.")
        return
    
    time.sleep(2)

    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # (quadrants and draw_box_functions dictionaries remain the same)
    quadrants = { 
        "top_left": (0, 0, width // 2, height // 2), 
        "top_right": (width // 2, 0, width, height // 2), 
        "bottom_left": (0, height // 2, width // 2, height), 
        "bottom_right": (width // 2, height // 2, width, height), 
        "center": (width // 4, height // 4, width * 3 // 4, height * 3 // 4) 
        }
    draw_box_functions = { 
        "top_left": draw_top_left_box, 
        "top_right": draw_top_right_box, 
        "bottom_left": draw_bottom_left_box, 
        "bottom_right": draw_bottom_right_box, 
        "center": draw_center_box 
        }

    # 2. START BACKGROUND LISTENING
    microphone = sr.Microphone()
    # Adjust for ambient noise once, then let the background listener take over
    with microphone as source:
        r.adjust_for_ambient_noise(source, duration=1)
    
    # This starts a separate thread that now has full control of the microphone
    stop_listening = r.listen_in_background(microphone, command_callback)

    textToSpeech("The camera is on. Please say a command like 'center' or 'top left'.")
    
    # 3. MAIN APPLICATION LOOP
    last_guidance_time = 0
    
    while True:
        result, video_frame = video_capture.read()
        if not result:
            break
            
        video_frame = cv2.flip(video_frame, 1)
        original_frame_for_photo = copy.deepcopy(video_frame)

        # The loop now checks if the background listener has set a command
        if not target_command:
            # STATE 1: WAITING FOR A COMMAND (but video is running)
            draw_all_boxes(video_frame) 
            cv2.putText(video_frame, "Listening for a command...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            # STATE 2: GUIDING THE USER
            stop_listening(wait_for_stop=False) # Stop the background listener  # stop listening for speech

            if 'first_guidance' not in locals():
                textToSpeech(f"Great. Moving to the {target_command.replace('_', ' ')} position.")
                first_guidance = True # Ensure this welcome message is only spoken once

            # (The rest of the guidance logic is the same as before)
            target_rect = quadrants[target_command]
            draw_box_functions[target_command](video_frame)
            faces = detect_face(video_frame)
            if len(faces) > 0:
                face = faces[0]
                is_in_position = (target_rect[0] < (face[0] + face[2] // 2) < target_rect[2]) and (target_rect[1] < (face[1] + face[3] // 2) < target_rect[3])
                eyes = detect_eyes(video_frame, face)
                is_face_straight = False
                if len(eyes) == 2:
                    left_eye, right_eye = (eyes[0], eyes[1]) if eyes[0][0] > eyes[1][0] else (eyes[1], eyes[0])  # Get consistent order of eyes to not switch sign of roll
                    deg = math.atan2((left_eye[1] - right_eye[1]), (left_eye[0] - right_eye[0]))
                    if abs(deg) < 0.2:
                        is_face_straight = True
                
                if is_in_position and is_face_straight:
                    textToSpeech("Perfect, hold still!")
                    save_photo(original_frame_for_photo, target_command)
                    textToSpeech("Photo taken!")
                    break
                
                elif time.time() - last_guidance_time > 2:
                    guidance_message = ""
                    if not is_in_position:
                         if (face[1] + face[3] // 2) < target_rect[1]: guidance_message += "Move down. "
                         elif (face[1] + face[3] // 2) > target_rect[3]: guidance_message += "Move up. "
                         if (face[0] + face[2] // 2) < target_rect[0]: guidance_message += "Move to your right. "
                         elif (face[0] + face[2] // 2) > target_rect[2]: guidance_message += "Move to your left. "
                    elif not is_looking_forward: guidance_message = "Please face the camera."
                    elif not is_face_straight: guidance_message = "Please level your head."
                    if guidance_message:
                        textToSpeech(guidance_message)
                        last_guidance_time = time.time()

        cv2.imshow("Selfie Helper", video_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # CLEANUP
    video_capture.release()
    cv2.destroyAllWindows()

# This is needed to run the main application
if __name__ == "__main__":
    main_application()




