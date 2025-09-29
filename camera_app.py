import copy
import cv2
import datetime
import math
import pyttsx3
import speech_recognition as sr
import time # We'll need this to avoid spamming the user with guidance

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
    # with sr.Microphone() as source:
        # read the audio data from the default microphone
       # audio_data = r.record(source, duration=dur)  # TODO: Adjust timing depending on flow of program
        # convert speech to text
        # text = r.recognize_google(audio_data)
       # return text #



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



def main_application():
    """This function runs the main application workflow."""
    
    # 1. SETUP AND INITIALIZATION 
    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        print("Error: Cannot open camera.")
        return
        
    # Gives the camera a moment to initialize
    time.sleep(2)
    
    # Get the actual width and height from the camera
    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Define the target quadrants using coordinates: (x1, y1, x2, y2)
    quadrants = {
        "top_left": (0, 0, width // 2, height // 2),
        "top_right": (width // 2, 0, width, height // 2),
        "bottom_left": (0, height // 2, width // 2, height),
        "bottom_right": (width // 2, height // 2, width, height),
        "center": (width // 4, height // 4, width * 3 // 4, height * 3 // 4)
    }
    
    # Map command names to the functions that draw their boxes
    draw_box_functions = {
        "top_left": draw_top_left_box,
        "top_right": draw_top_right_box,
        "bottom_left": draw_bottom_left_box,
        "bottom_right": draw_bottom_right_box,
        "center": draw_center_box
    }
    
    # 2. GET THE USER'S TARGET POSITION 
    target_command = ""
    while not target_command:
        textToSpeech("Where would you like your face? For example: say top left, or center.")
        user_speech = speechToText()
        print(f"I heard you say: '{user_speech}'")
        target_command = textToCommand(user_speech)
        
        if not target_command:
            textToSpeech("I did not understand. Please try again.")

    textToSpeech(f"Okay, let's get you to the {target_command.replace('_', ' ')} position.")
    target_rect = quadrants[target_command]
    
    # 3. START THE GUIDANCE AND CAPTURE LOOP
    last_guidance_time = 0
    while True:
        result, video_frame = video_capture.read()
        if not result:
            break
            
        # Flip the frame to act like a mirror, which is more intuitive
        video_frame = cv2.flip(video_frame, 1)
        original_frame_for_photo = copy.deepcopy(video_frame)

        # Display ONLY the correct box for the chosen command
        draw_box_functions[target_command](video_frame)

        faces = detect_face(video_frame)
        
        if len(faces) > 0:
            # Focus on the first face found
            face = faces[0]
            (x, y, w, h) = face
            face_center_x = x + w // 2
            face_center_y = y + h // 2

            # Check if the user is facing the camera (requires 2 eyes)
            eyes = detect_eyes(video_frame, face)
            is_facing_forward = len(eyes) >= 2

            # Check if the face's center is inside the target box
            (x1, y1, x2, y2) = target_rect
            is_in_position = (x1 < face_center_x < x2) and (y1 < face_center_y < y2)

            #  SUCCESS CONDITION 
            if is_in_position and is_facing_forward:
                textToSpeech("Perfect, hold still!")
                save_photo(original_frame_for_photo, target_command)
                textToSpeech("Photo taken! You can now close the window.")
                time.sleep(2) # Give user time to hear the message
                break # Exit the loop!
            
            #  GUIDANCE LOGIC 
            # Only give a new instruction every 2 seconds to avoid spam
            elif time.time() - last_guidance_time > 2:
                guidance_message = ""
                if face_center_y < y1: guidance_message += "Move down. "
                elif face_center_y > y2: guidance_message += "Move up. "
                
                if face_center_x < x1: guidance_message += "Move to your right. "
                elif face_center_x > x2: guidance_message += "Move to your left. "

                if not is_facing_forward:
                    guidance_message = "Please face the camera directly."

                if guidance_message:
                    textToSpeech(guidance_message)
                    last_guidance_time = time.time()
        
        else:
            # Give feedback if no face is found
            if time.time() - last_guidance_time > 5:
                 textToSpeech("I can't see your face. Please move in front of the camera.")
                 last_guidance_time = time.time()

        cv2.imshow("Selfie Helper", video_frame)

        # Allow user to quit manually by pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # 4. CLEANUP 
    video_capture.release()
    cv2.destroyAllWindows()


# This line makes sure the main_application() function runs when you execute the script
if __name__ == "__main__":
    # Initialize the engine once for the initial prompt
    pyttsx3.init().runAndWait() 
    main_application()

