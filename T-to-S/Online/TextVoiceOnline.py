from gtts import gTTS
import os

class TextToVoiceSystem:
    def __init__(self, lang='en'):
        self.lang = lang

    def speak(self, text):
        tts = gTTS(text=text, lang=self.lang)
        tts.save("output.mp3")
        os.system("mpg321 output.mp3")

# Example Usage
# tts_system = TextToVoiceSystem()
# tts_system.speak("This is the recommended answer.")