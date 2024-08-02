import speech_recognition as sr
import numpy as np
import av

class AudioProcessor(AudioProcessor):
    def __init__(self):
        self.recorder = sr.Recognizer()
        self.audio_data = None

    def recv_audio(self, frame: av.AudioFrame) -> av.AudioFrame:
        audio = frame.to_ndarray()
        self.audio_data = np.frombuffer(audio, np.int16).tobytes()
        return frame

    def get_text(self):
        if self.audio_data:
            audio = sr.AudioData(self.audio_data, sample_rate=frame.sample_rate, sample_width=2)
            try:
                text = self.recorder.recognize_google(audio, language="pt-BR")
                return text
            except sr.UnknownValueError:
                return "Não foi possível reconhecer o áudio"
            except sr.RequestError as e:
                return f"Erro no serviço de reconhecimento: {e}"
        return ""