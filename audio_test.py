import pyttsx3

print("Initializing speech engine...")
engine = pyttsx3.init(driverName='sapi5')

print("Attempting to speak...")
engine.say("If you can hear this, the speech engine is working.")
engine.runAndWait()

print("Test complete.")