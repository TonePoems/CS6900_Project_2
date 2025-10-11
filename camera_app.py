import copy
import cv2
import datetime
import math
import speech_recognition as sr
import time
import torch
from transformers import pipeline
import sounddevice as sd


# NEW HUGGING FACE SETUP 
# Load the Text-to-Speech pipeline from Hugging Face
print("Loading Text-to-Speech model...")
tts_pipeline = pipeline("text-to-speech", model="microsoft/speecht5_tts", device="cuda" if torch.cuda.is_available() else "cpu")

# Create a consistent, random speaker embedding ONCE.
print("Creating a consistent speaker voice...")
speaker_embedding = torch.randn((1, 512))
print("Model and voice loaded.")

# create a speech recognition object
r = sr.Recognizer()

# Global variable to hold the command from the background thread
target_command = ""
last_speech_time = 0
stop_listening = None

# Use microphone to get input from a user
# Optional arg for duration of audio length
# Returns string of user's input
def speechToText(dur=3):
    with sr.Microphone() as source:
        # read the audio data from the default microphone
        audio_data = r.record(source, duration=dur)  
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


## NEW TEXT TO SPEECH FUNCTION
def textToSpeech(text, wait=False):
    """
    Uses a Hugging Face model to generate and play speech without freezing the app.
    """
    global last_speech_time
    print(f"SAYING: {text}")

    # Generate the audio from the text using the AI model
    speech = tts_pipeline(text, forward_params={"speaker_embeddings": speaker_embedding})

    # Play the generated audio. 
    sd.play(speech["audio"], samplerate=speech["sampling_rate"])
    if wait:
        sd.wait() 

    # Record the time we started speaking to prevent feedback loops
    last_speech_time = time.time()

def command_callback(recognizer, audio):
    #"""This function is called in the background when speech is detected."""
    global target_command, last_speech_time
    
    # If the app has spoken in the last 5 seconds, ignore any audio (it's an echo)
    if time.time() - last_speech_time < 5:
        return

    if target_command: return
    try:
        speech = recognizer.recognize_google(audio)
        print(f"I heard you say: '{speech}'")
        command = textToCommand(speech)
        if command:
            target_command = command
            if stop_listening:
                stop_listening(wait_for_stop=False)
    except sr.UnknownValueError:
        textToSpeech("I did not understand that. Please say a command again.")
    except sr.RequestError:
        textToSpeech("Sorry, the speech service is unavailable.")


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


# MODIFIED MAIN LOOP
def main_application():
    """This function runs the main application workflow with all features combined."""
    global target_command, stop_listening

    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        textToSpeech("Error: Cannot open camera."); return
        
    time.sleep(1)
    
    width, height = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    quadrants = { "top_left": (0, 0, width // 2, height // 2), "top_right": (width // 2, 0, width, height // 2), "bottom_left": (0, height // 2, width // 2, height), "bottom_right": (width // 2, height // 2, width, height), "center": (width // 4, height // 4, width * 3 // 4, height * 3 // 4) }
    draw_box_functions = { "top_left": draw_top_left_box, "top_right": draw_top_right_box, "bottom_left": draw_bottom_left_box, "bottom_right": draw_bottom_right_box, "center": draw_center_box }

    microphone = sr.Microphone()
    with microphone as source:
        print("Adjusting for ambient noise...")
        r.adjust_for_ambient_noise(source, duration=1)
        print(f"Ambient noise adjustment complete.")

    stop_listening = r.listen_in_background(microphone, command_callback)
    textToSpeech("The camera is on. Please say a command like top left or center.")

    last_guidance_time = time.time()  # Start a timer to create a "quiet period" for the welcome message
    debug = True
    last_face_position = None # To store the (x, y) of the face from the last frame
    face_still_start_time = None # To track how long the face has been still
    
    while True:
        result, video_frame = video_capture.read()
        if not result: break
            
        video_frame = cv2.flip(video_frame, 1)
        original_frame_for_photo = copy.deepcopy(video_frame)

        if not target_command:
            # STATE 1: WAITING FOR A COMMAND
            draw_all_boxes(video_frame) 
        
            
        else:
            # STATE 2: GUIDING THE USER
            if 'first_guidance' not in locals():
                textToSpeech(f"Great. Moving to the {target_command.replace('_', ' ')} position.")
                last_guidance_time = time.time()
                first_guidance = True

            target_rect = quadrants[target_command]
            draw_box_functions[target_command](video_frame)
            
            # YOUR ORIGINAL LOGIC STARTS HERE
            faces = detect_face(video_frame, debug) # Using your original function call
            

            if len(faces) > 0:
                face = faces[0]; (x, y, w, h) = face
                current_face_center = (x + w // 2, y + h // 2)
                
                is_in_position = (target_rect[0] < current_face_center[0] < target_rect[2]) and (target_rect[1] < current_face_center[1] < target_rect[3])
                eyes = detect_eyes(video_frame, face, debug)
                is_face_straight = False
                if len(eyes) >= 2:
                    left_eye, right_eye = (eyes[0], eyes[1]) if eyes[0][0] > eyes[1][0] else (eyes[1], eyes[0])
                    deg = math.atan2((left_eye[1] - right_eye[1]), (left_eye[0] - right_eye[0]))
                    if abs(deg) < 0.25: #Increased from 0.2
                        is_face_straight = True
                
                # NEW MOVEMENT DETECTION LOGIC
                movement_threshold = 10 # Pixels
                
                if last_face_position is not None:
                    # Calculate the distance the face has moved since the last frame
                    distance_moved = math.sqrt((current_face_center[0] - last_face_position[0])**2 + (current_face_center[1] - last_face_position[1])**2)
                    
                    if distance_moved < movement_threshold:
                        # If the face is still, start or continue the stillness timer
                        if face_still_start_time is None:
                            face_still_start_time = time.time()
                    else:
                        # If the face is moving, reset the stillness timer
                        face_still_start_time = None
                
                # Update the last position for the next frame's calculation
                last_face_position = current_face_center
                
                
                # Check if the face has been still for over a second
                if face_still_start_time is not None and (time.time() - face_still_start_time > 1.0):
                    # Check if enough time has passed since the LAST command to avoid spamming
                    if time.time() - last_guidance_time > 3:
                        # SUCCESS CONDITION 
                        if is_in_position and is_face_straight:
                            textToSpeech("Perfect, hold still!", wait=True) # Add wait=True
                            save_photo(original_frame_for_photo, target_command)
                            textToSpeech("Photo taken!", wait=True) # Add wait=True
                            break
                        
                        # GUIDANCE LOGIC 
                        guidance_message = ""
                        if not is_in_position:
                             if current_face_center[1] < target_rect[1]: guidance_message += "Move down. "
                             elif current_face_center[1] > target_rect[3]: guidance_message += "Move up. "
                             if current_face_center[0] < target_rect[0]: guidance_message += "Move to your right. "
                             elif current_face_center[0] > target_rect[2]: guidance_message += "Move to your left. "
                        elif not is_face_straight: guidance_message = "Please level your head."
                        elif len(eyes) < 2: guidance_message = "Please face the camera."
                        
                        if guidance_message:
                            textToSpeech(guidance_message)
                            last_guidance_time = time.time()
                            # Reset stillness timer after speaking to wait for the next stop
                            face_still_start_time = None 
            else:
                # If the face is lost from the frame, reset the tracking variables
                last_face_position = None
                face_still_start_time = None
                if time.time() - last_guidance_time > 5:
                    textToSpeech("I can't see your face.")
                    last_guidance_time = time.time()
            

        cv2.imshow("Selfie Helper", video_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"): break

    if stop_listening:
        stop_listening(wait_for_stop=True)
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main_application()


 
