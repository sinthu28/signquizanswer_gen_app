import pyttsx3

class TextToVoiceSystem:
    def __init__(self, rate=150, volume=1.0):
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', rate)
        self.engine.setProperty('volume', volume)

    def speak(self, text):
        self.engine.say(text)
        self.engine.runAndWait()

# Example Usage
# tts_system = TextToVoiceSystem()
# tts_system.speak("This is the recommended answer.")